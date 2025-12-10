from glob import glob
from tqdm import tqdm
import os
from os.path import join, basename
import re
import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd
import numpy as np
import argparse

from PIL import Image
import SimpleITK as sitk
import torch
import torch.multiprocessing as mp
from sam2.build_sam import build_sam2_video_predictor_npz
import SimpleITK as sitk
from skimage import measure, morphology

torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

parser = argparse.ArgumentParser()

parser.add_argument(
    '--checkpoint',
    type=str,
    default="checkpoints/MedSAM2_latest.pt",
    help='checkpoint path',
)
parser.add_argument(
    '--cfg',
    type=str,
    default="E:\\Hadraba\\MedSAM2\\sam2\\configs\\sam2.1_hiera_t512.yaml",
    help='model config',
)

parser.add_argument(
    '-i',
    '--img_path',
    type=str,
    default="kurz_femur_202.nii",
    help='imgs path',
)
parser.add_argument(
    '--gts_path',
    default=None,
    help='simulate prompts based on ground truth',
)
parser.add_argument(
    '-o',
    '--pred_save_dir',
    type=str,
    default="./test_femur_results",
    help='path to save segmentation results',
)
# add option to propagate with either box or mask
parser.add_argument(
    '--propagate_with_box',
    default=True,
    action='store_true',
    help='whether to propagate with box'
)

args = parser.parse_args()
checkpoint = args.checkpoint
model_cfg = args.cfg
img_path = args.img_path
gts_path = args.gts_path
pred_save_dir = args.pred_save_dir
os.makedirs(pred_save_dir, exist_ok=True)
propagate_with_box = args.propagate_with_box

def getLargestCC(segmentation):
    labels = measure.label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def dice_multi_class(preds, targets):
    smooth = 1.0
    assert preds.shape == targets.shape
    labels = np.unique(targets)[1:]
    dices = []
    for label in labels:
        pred = preds == label
        target = targets == label
        intersection = (pred * target).sum()
        dices.append((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))
    return np.mean(dices)

def show_mask(mask, ax, mask_color=None, alpha=0.5):
    """
    show mask on the image

    Parameters
    ----------
    mask : numpy.ndarray
        mask of the image
    ax : matplotlib.axes.Axes
        axes to plot the mask
    mask_color : numpy.ndarray
        color of the mask
    alpha : float
        transparency of the mask
    """
    if mask_color is not None:
        color = np.concatenate([mask_color, np.array([alpha])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, edgecolor='blue'):
    """
    show bounding box on the image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    ax : matplotlib.axes.Axes
        axes to plot the bounding box
    edgecolor : str
        color of the bounding box
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))     


def resize_grayscale_to_rgb_and_resize(array, image_size):
    """
    Resize a 3D grayscale NumPy array to an RGB image and then resize it.
    
    Parameters:
        array (np.ndarray): Input array of shape (d, h, w).
        image_size (int): Desired size for the width and height.
    
    Returns:
        np.ndarray: Resized array of shape (d, 3, image_size, image_size).
    """
    d, h, w = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size))
    
    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)  # (3, image_size, image_size)
        resized_array[i] = img_array
    
    return resized_array

def mask2D_to_bbox(gt2D, max_shift=20):
    y_indices, x_indices = np.where(gt2D > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    H, W = gt2D.shape
    bbox_shift = np.random.randint(0, max_shift + 1, 1)[0]
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W-1, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H-1, y_max + bbox_shift)
    boxes = np.array([x_min, y_min, x_max, y_max])
    return boxes

def mask3D_to_bbox(gt3D, max_shift=20):
    z_indices, y_indices, x_indices = np.where(gt3D > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    z_min, z_max = np.min(z_indices), np.max(z_indices)
    D, H, W = gt3D.shape
    bbox_shift = np.random.randint(0, max_shift + 1, 1)[0]
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W-1, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H-1, y_max + bbox_shift)
    z_min = max(0, z_min)
    z_max = min(D-1, z_max)
    boxes3d = np.array([x_min, y_min, z_min, x_max, y_max, z_max])
    return boxes3d


class BoundingBoxSelector:
    def __init__(self, image):
        self.image = image
        self.bbox = None
        self.start_point = None
        self.rect = None
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))
        
    def select_bbox(self):
        self.ax.imshow(self.image, cmap='gray')
        self.ax.set_title('Click and drag to draw a bounding box.')
        self.ax.axis('off')
        
        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        plt.tight_layout()
        plt.show()
        
        return self.bbox
    
    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.start_point = (event.xdata, event.ydata)
        
        # Remove previous rectangle if exists
        if self.rect is not None:
            self.rect.remove()
            self.rect = None
    
    def on_motion(self, event):
        if self.start_point is None or event.inaxes != self.ax:
            return
        
        # Remove previous rectangle
        if self.rect is not None:
            self.rect.remove()
        
        # Draw new rectangle
        x0, y0 = self.start_point
        width = event.xdata - x0
        height = event.ydata - y0
        
        self.rect = plt.Rectangle((x0, y0), width, height, 
                                   edgecolor='red', facecolor='none', 
                                   linewidth=2, linestyle='--')
        self.ax.add_patch(self.rect)
        self.fig.canvas.draw()
    
    def on_release(self, event):
        if self.start_point is None or event.inaxes != self.ax:
            return
        
        x0, y0 = self.start_point
        x1, y1 = event.xdata, event.ydata
        
        # Ensure x0 < x1 and y0 < y1
        x_min = int(min(x0, x1))
        x_max = int(max(x0, x1))
        y_min = int(min(y0, y1))
        y_max = int(max(y0, y1))
        
        # Store bbox in format [x_min, y_min, x_max, y_max]
        self.bbox = [x_min, y_min, x_max, y_max]
        
        # Update rectangle to final position
        if self.rect is not None:
            self.rect.remove()
        
        self.rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                   edgecolor='green', facecolor='none', 
                                   linewidth=2)
        self.ax.add_patch(self.rect)
        self.ax.set_title(f'Bounding box selected: [{x_min}, {y_min}, {x_max}, {y_max}]\nClose window to continue.')
        self.fig.canvas.draw()
        
        self.start_point = None


predictor = build_sam2_video_predictor_npz(model_cfg, checkpoint)

# get corresponding case info
nii_image = sitk.ReadImage(img_path)
nii_image_data = sitk.GetArrayFromImage(nii_image)


seg_3D = np.zeros(nii_image_data.shape, dtype=np.uint8)

# get the key slice info
nii_image_data_pre = (nii_image_data - np.min(nii_image_data))/(np.max(nii_image_data)-np.min(nii_image_data))*255.0
nii_image_data_pre = np.uint8(nii_image_data_pre)
key_slice_idx = nii_image_data.shape[0] // 2  # use the middle slice as key slice for demo

key_slice_img = nii_image_data_pre[key_slice_idx, :,:]

# Use GUI to select bounding box
print(f"Please select a bounding box on slice {key_slice_idx}")
bbox_selector = BoundingBoxSelector(key_slice_img)
bbox_coords = bbox_selector.select_bbox()

if bbox_coords is None:
    raise ValueError("No bounding box was selected. Please run again and draw a bounding box.")

print(f"Selected bounding box: {bbox_coords}")
bbox = np.array(bbox_coords)  # [x_min, y_min, x_max, y_max]

img_3D_ori = nii_image_data_pre
assert np.max(img_3D_ori) < 256, f'input data should be in range [0, 255], but got {np.unique(img_3D_ori)}'

video_height = key_slice_img.shape[0]
video_width = key_slice_img.shape[1]
img_resized = resize_grayscale_to_rgb_and_resize(img_3D_ori, 512)
img_resized = img_resized / 255.0
img_resized = torch.from_numpy(img_resized).cuda()
img_mean=(0.485, 0.456, 0.406)
img_std=(0.229, 0.224, 0.225)
img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None].cuda()
img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None].cuda()
img_resized -= img_mean
img_resized /= img_std
z_mids = []

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    inference_state = predictor.init_state(img_resized, video_height, video_width)
    if propagate_with_box:
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                                            inference_state=inference_state,
                                            frame_idx=key_slice_idx,
                                            obj_id=1,
                                            box=bbox,
                                        )
    else: # gt
        pass

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        seg_3D[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
    predictor.reset_state(inference_state)
    if propagate_with_box:
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                                            inference_state=inference_state,
                                            frame_idx=key_slice_idx,
                                            obj_id=1,
                                            box=bbox,
                                        )
    else: # gt
        pass

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
        seg_3D[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
    predictor.reset_state(inference_state)

    if np.max(seg_3D) > 0:
        seg_3D = getLargestCC(seg_3D)
        seg_3D = np.uint8(seg_3D) 
    
    # Save results
    sitk_image = sitk.GetImageFromArray(img_3D_ori)
    sitk_image.CopyInformation(nii_image)
    sitk_mask = sitk.GetImageFromArray(seg_3D)
    sitk_mask.CopyInformation(nii_image)
    
    # Generate output filename based on input
    nii_fname = basename(img_path)
    save_seg_name = nii_fname.replace('.nii', f'_k{key_slice_idx}_mask.nii')
    save_img_name = nii_fname.replace('.nii', '_img.nii')
    
    sitk.WriteImage(sitk_image, os.path.join(pred_save_dir, save_img_name))
    sitk.WriteImage(sitk_mask, os.path.join(pred_save_dir, save_seg_name))
    
    print(f"Segmentation saved to {os.path.join(pred_save_dir, save_seg_name)}")
    print(f"Image saved to {os.path.join(pred_save_dir, save_img_name)}")



