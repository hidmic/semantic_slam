#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'include'))
import cv2
import numpy as np
import time
from skimage.transform import resize
import torch
from ptsemseg.models import get_model
from ptsemseg.utils import convert_state_dict

def color_map(N=256, normalized=False):
    """
    Return Color Map in PASCAL VOC format (rgb)
    \param N (int) number of classes
    \param normalized (bool) whether colors are normalized (float 0-1)
    \return (Nx3 numpy array) a color map
    """
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)
    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3
        cmap[i] = np.array([r, g, b])
    cmap = cmap/255.0 if normalized else cmap
    return cmap

def decode_segmap(temp, n_classes, cmap):
    """
    Given an image of class predictions, produce an bgr8 image with class colors
    \param temp (2d numpy int array) input image with semantic classes (as integer)
    \param n_classes (int) number of classes
    \cmap (Nx3 numpy array) input color map
    \return (numpy array bgr8) the decoded image with class colors
    """
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = cmap[l,0]
        g[temp == l] = cmap[l,1]
        b[temp == l] = cmap[l,2]
    bgr = np.zeros((temp.shape[0], temp.shape[1], 3))
    bgr[:, :, 0] = b
    bgr[:, :, 1] = g
    bgr[:, :, 2] = r
    return bgr.astype(np.uint8)

class seg_test:
    def __init__(self):
        self.img_width, self.img_height = 640, 480
        print('Setting up CNN model...')
        # Set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPU: device=cuda
        dataset = 'ade20k'
        model_name = 'pspnet'
        model_path = '/home/yubao/data/Dataset/semantic_slam/pspnet_50_ade20k.pth'

        if dataset == 'sunrgbd':  # If use version fine tuned on sunrgbd dataset
            self.n_classes = 38  # Semantic class number
            self.model = get_model(model_name, self.n_classes, version='sunrgbd_res50')
            state = torch.load(model_path, map_location='cuda:0')
            self.model.load_state_dict(state)
            self.cnn_input_size = (321, 321)
            self.mean = np.array([104.00699, 116.66877, 122.67892])  # Mean value of dataset
        elif dataset == 'ade20k':
            self.n_classes = 150  # Semantic class number
            self.model = get_model(model_name, self.n_classes, version='ade20k')
            state = torch.load(model_path)
            self.model.load_state_dict(convert_state_dict(state['model_state']))  # Remove 'module' from dictionary keys
            self.cnn_input_size = (473, 473)
            self.mean = np.array([104.00699, 116.66877, 122.67892])  # Mean value of dataset
        self.model = self.model.to(self.device)
        self.model.eval()
        self.cmap = color_map(N=self.n_classes, normalized=False)  # Color map for semantic classes

    def predict(self, img):
        """
        Do semantic segmantation
        \param img: (numpy array bgr8) The input cv image
        """
        img = img.copy()  # Make a copy of image because the method will modify the image
        # orig_size = (img.shape[0], img.shape[1]) # Original image size
        # Prepare image: first resize to CNN input size then extract the mean value of SUNRGBD dataset. No normalization
        img = resize(img, self.cnn_input_size, mode='reflect', anti_aliasing=True, preserve_range=True)  # Give float64
        img = img.astype(np.float32)
        img -= self.mean
        # Convert HWC -> CHW
        img = img.transpose(2, 0, 1)
        # Convert to tensor
        img = torch.tensor(img, dtype=torch.float32)
        img = img.unsqueeze(0)  # Add batch dimension required by CNN
        with torch.no_grad():
            img = img.to(self.device)
            # Do inference
            since = time.time()
            outputs = self.model(img)  # N,C,W,H
            # Apply softmax to obtain normalized probabilities
            outputs = torch.nn.functional.softmax(outputs, 1)
            return outputs

if __name__ == '__main__':
    segnet = seg_test()
    color_img = cv2.imread('walk_rgb_1.png')

    color_img = cv2.resize(color_img, (segnet.img_width, segnet.img_height), interpolation=cv2.INTER_NEAREST)

    # Do semantic segmantation
    class_probs = segnet.predict(color_img)
    confidence, label = class_probs.max(1)
    confidence, label = confidence.squeeze(0).cpu().numpy(), label.squeeze(0).cpu().numpy()
    label = resize(label, (segnet.img_height, segnet.img_width), order=0, mode='reflect', anti_aliasing=False,
                   preserve_range=True)  # order = 0, nearest neighbour
    label_int = label.astype(np.int)

    # Add semantic class colors
    decoded = decode_segmap(label_int, segnet.n_classes, segnet.cmap)  # Show input image and decoded image

    class_labels = np.unique(label_int)
    print(class_labels)

    confidence = resize(confidence, (segnet.img_height, segnet.img_width), mode='reflect', anti_aliasing=True,
                        preserve_range=True)
    cv2.imshow('Camera image', color_img)
    cv2.imshow('confidence', confidence)
    cv2.imshow('Semantic segmantation', decoded)
    cv2.waitKey(0)
