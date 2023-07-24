import os
from os import path
from argparse import ArgumentParser
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from inference.data.test_datasets import LongTestDataset, DAVISTestDataset, YouTubeVOSTestDataset
from inference.data.mask_mapper import MaskMapper
from model.network import XMem
from inference.inference_core import InferenceCore

from progressbar import progressbar

torch.set_grad_enabled(False)

if torch.cuda.is_available():
  print('Using GPU')
  device = 'cuda'
else:
  print('CUDA not available. Please connect to a GPU instance if possible.')
  device = 'cpu'

# default configuration
config = {
    'top_k': 30,
    'mem_every': 5,
    'deep_update_every': -1,
    'enable_long_term': True,
    'enable_long_term_count_usage': True,
    'num_prototypes': 128,
    'min_mid_term_frames': 5,
    'max_mid_term_frames': 10,
    'max_long_term_elements': 10000,
}

network = XMem(config, './saves/XMem.pth').eval().to(device)

# video_name = 'video_sample.mp4'
# mask_name = 'first_frame_sample.png'

video_name = 'original_video.mp4'
mask_name = '00.pred.png'

mask = np.array(Image.open(mask_name))
print(np.unique(mask))

copy_mask = np.copy(mask)
w, h, d = copy_mask.shape

copy_mask = copy_mask.reshape((w * h, d))

mask_dict = {}
mask_idx = 0
for i in range(w*h):
    if str(copy_mask[i]) in mask_dict:
        continue
    else:
        mask_dict[str(copy_mask[i])] = mask_idx
        mask_idx += 1

new_mask = np.zeros(w*h)
for i in range(w*h):
    new_mask[i] = mask_dict[str(copy_mask[i])]
    
mask = new_mask.reshape((w, h))
print(np.unique(mask))

num_objects = len(np.unique(mask)) - 1


import cv2
from inference.interact.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask, overlay_davis

torch.cuda.empty_cache()

processor = InferenceCore(network, config=config)
processor.set_all_labels(range(1, num_objects+1)) # consecutive labels
cap = cv2.VideoCapture(video_name)

# You can change these two numbers
frames_to_propagate = 200
visualize_every = 1

current_frame_index = 0

with torch.cuda.amp.autocast(enabled=True):
  while (cap.isOpened()):
    # load frame-by-frame
    _, frame = cap.read()
    if frame is None or current_frame_index > frames_to_propagate:
      break

    # convert numpy array to pytorch tensor format
    frame_torch, _ = image_to_torch(frame, device=device)
    if current_frame_index == 0:
      # initialize with the mask
      mask_torch = index_numpy_to_one_hot_torch(mask, num_objects+1).to(device)
      # the background mask is not fed into the model
      prediction = processor.step(frame_torch, mask_torch[1:])
    else:
      # propagate only
      prediction = processor.step(frame_torch)

    # argmax, convert to numpy
    prediction = torch_prob_to_numpy_mask(prediction)

    if current_frame_index % visualize_every == 0:
        visualization = overlay_davis(frame, prediction, alpha=0.2)
        rgb_visualization = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
        masked_image = Image.fromarray(rgb_visualization)
        masked_image.save("{}.jpg".format(current_frame_index))

    current_frame_index += 1