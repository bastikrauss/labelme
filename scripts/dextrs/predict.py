import os
import sys
import torch
from collections import OrderedDict
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2

from torch.nn.functional import upsample

import dextrs.networks.deeplab_resnet as resnet
from dextrs.mypath import Path
from dextrs.dataloaders import helpers as helpers

class DextrModel():

    def __init__(self):
        modelPath = 'C:/Users/krseba/Documents/labelme/scripts/dextrs/models/dextr_pascal-sbd.pth'
        self.pad = 50
        self.thres = 0.8
        gpu_id = 0
        self.device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
        print("Device: {}".format(self.device))
        #  Create the network and load the weights
        self.net = resnet.resnet101(1, nInputChannels=4, classifier='psp')
        print("Initializing weights from: {}".format(modelPath))
        state_dict_checkpoint = torch.load(modelPath,
                                        map_location=lambda storage, loc: storage)
        # Remove the prefix .module from the model when it is trained using DataParallel
        if 'module.' in list(state_dict_checkpoint.keys())[0]:
            new_state_dict = OrderedDict()
            for k, v in state_dict_checkpoint.items():
                name = k[7:]  # remove `module.` from multi-gpu training
                new_state_dict[name] = v
        else:
            new_state_dict = state_dict_checkpoint
        self.net.load_state_dict(new_state_dict)
        self.net.eval()
        self.net.to(self.device)

    def dextrPrediction(self, filename, extreme_points_ori):            
        #  Read image
        image = np.array(Image.open(filename))
        results = []

        #all operations have no gradients
        with torch.no_grad():
            #  Crop image to the bounding box from the extreme points and resize
            bbox = helpers.get_bbox(image, points=extreme_points_ori, pad=self.pad, zero_pad=True)
            crop_image = helpers.crop_from_bbox(image, bbox, zero_pad=True)
            resize_image = helpers.fixed_resize(crop_image, (512, 512)).astype(np.float32)
            print("image resized")

            #  Generate extreme point heat map normalized to image values
            extreme_points = extreme_points_ori - [np.min(extreme_points_ori[:, 0]), np.min(extreme_points_ori[:, 1])] + [self.pad,
                                                                                                                        self.pad]
            extreme_points = (512 * extreme_points * [1 / crop_image.shape[1], 1 / crop_image.shape[0]]).astype(np.int)
            extreme_heatmap = helpers.make_gt(resize_image, extreme_points, sigma=10)
            extreme_heatmap = helpers.cstm_normalize(extreme_heatmap, 255)

            #  Concatenate inputs and convert to tensor
            input_dextr = np.concatenate((resize_image, extreme_heatmap[:, :, np.newaxis]), axis=2)
            inputs = torch.from_numpy(input_dextr.transpose((2, 0, 1))[np.newaxis, ...])

            # Run a forward pass
            inputs = inputs.to(self.device)
            outputs = self.net.forward(inputs)
            outputs = upsample(outputs, size=(512, 512), mode='bilinear', align_corners=True)
            outputs = outputs.to(torch.device('cpu'))

            pred = np.transpose(outputs.data.numpy()[0, ...], (1, 2, 0))
            pred = 1 / (1 + np.exp(-pred))
            pred = np.squeeze(pred)
            result = helpers.crop2fullmask(pred, bbox, im_size=image.shape[:2], zero_pad=True, relax=self.pad) > self.thres

            results.append(result)

            npresults = np.array(results)
            intresults = npresults.astype(np.uint8)
            sqresults=np.squeeze(intresults, axis=0)

            # convert mask into contour points
            contour, hierarchy = cv2.findContours(sqresults, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1 )
            cnt = contour[0]
            approxFactor = 0.01  #adjust the factor for approximation (smaller factor -> more points)
            epsilon = approxFactor*cv2.arcLength(cnt,True)
            # approximate contour
            approx = cv2.approxPolyDP(cnt,epsilon,True)

            return approx
