# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 10:45:06 2018

@author: kowns
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys
import random
import math
import re
import time
import numpy as np
import cv2
# import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from config import Config
import utils
import model as modellib
import visualize
from tqdm import tqdm
from model import log
from utils import resize_image, resize_mask

from keras.utils import Progbar
from skimage.io import imread, imshow, imread_collection, concatenate_images, imsave
from visualize import display_images
from keras import backend as K
from scipy import ndimage
import kaggle_util

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

FULLDATA = 0

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

TRAIN_PATH = '/home/kownse/code2/kaggle/ship/input/train_v2/'
VALID_PATH = '/home/kownse/code2/kaggle/ship/input/train_v2/'
TEST_PATH = '/home/kownse/code2/kaggle/ship/input/test_v2/'

train_ids = next(os.walk(TRAIN_PATH))[1]
#test_ids = next(os.walk(TEST_PATH))[1]
total_size = len(train_ids)

IMG_SIZE = 768


def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction
    
class ShipConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "ship"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 shapes
    BACKBONE = 'resnet50'
    
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_PADDING = True
    LEARNING_RATE = 0.002

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 128
    DETECTION_MAX_INSTANCES = 32
    MAX_GT_INSTANCES = 32
    RPN_TRAIN_ANCHORS_PER_IMAGE = 32
    #BACKBONE = "resnet50"

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50
    ARGUMENT = 1
    LEARNING_RATE = 0.001

class CoCoShipConfig(ShipConfig):
    NAME = "coco_ship_"
    IMAGES_PER_GPU = 2
    STEPS_PER_EPOCH = 10000
    VALIDATION_STEPS = 1000
    TRAIN_FROM_COCO = 1

    IMAGE_CHANEL = 3
    
class InferenceConfig(CoCoShipConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
class ShipConfig_101(CoCoShipConfig):
    NAME = "res101_"
    BACKBONE = 'resnet101'
    STEPS_PER_EPOCH = 10000
    TRAIN_FROM_COCO = 0
    
class InferenceConfig_101(ShipConfig_101):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
class ShipOnly(CoCoShipConfig):
    NAME = "headonly_"
    STEPS_PER_EPOCH = 3000
    TRAIN_FROM_COCO = 1
    IMAGES_PER_GPU = 4

class InferenceConfig_ShipOnly(ShipOnly):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
train_config = ShipOnly
test_config = InferenceConfig_ShipOnly

# def get_ax(rows=1, cols=1, size=8):
#     """Return a Matplotlib Axes array to be used in
#     all visualizations in the notebook. Provide a
#     central point to control graph sizes.
    
#     Change the default size attribute to control the size
#     of rendered images
#     """
#     _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
#     return ax

class ShapesDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    scale_ = 1.0
    padding_ = None
    segment = pd.read_csv('/home/kownse/code2/kaggle/ship/input/train_ship_segmentations_v2.csv.zip')

    exclude_list = ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg', 
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg']
    
    def load_imgs(self, basepath):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "ship")
        
        folds = pd.read_csv('../input/folds.csv')
        if not FULLDATA:
            folds = folds.loc[folds.has_ship > 0]
        
        print('self.fold', self.fold, 'self.valid', self.valid)
        if self.fold >= 0:
            if self.valid == 0:
                folds = folds.loc[folds.fold != self.fold]
            else:
                folds = folds.loc[folds.fold == self.fold]
        print(folds.shape)
        if self.holdout >= 0:
            folds = folds.loc[folds.holdout == self.valid]
            
        print(basepath, 'total imgs', len(folds))
        prog = tqdm(total = len(folds))
        for idx, row in folds.iterrows():
            id_ = row.ImageId
            if id_ in self.exclude_list:
                continue
            
            img_path = basepath + id_
            if not os.path.isfile(img_path):
                continue

            shapes = []
            self.add_image("shapes", image_id=id_, path = img_path,
                           width=None, height=None, shapes=shapes)
            prog.update(1)
            
    def load_test_imgs(self, basepath):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "ship")
        
        imgs = os.listdir(basepath)
        print(basepath, 'total imgs', len(imgs))
        prog = tqdm(total = len(imgs))
        for idx, id_ in enumerate(imgs):
            img_path = basepath + id_
            if not os.path.isfile(img_path):
                continue

            shapes = []
            self.add_image("shapes", image_id=id_, path = img_path,
                           width=None, height=None, shapes=shapes)
            prog.update(1)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        path = info['path']
        img = imread(path)
        if img.shape[0] == 0:
            img = np.zeros((IMG_SIZE,IMG_SIZE,3))
        img = img.astype(np.uint8)
        
        info['width'] = img.shape[0]
        info['height'] = img.shape[1]
        
        return img

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def masks_as_image(self, rle, IMG_WIDTH, IMG_HEIGHT):
        # Take the individual ship masks and create a single mask array for all ships
        cnt = len(rle)
        all_masks = np.zeros((IMG_WIDTH, IMG_HEIGHT, cnt), dtype = np.int8)
        for idx, mask in enumerate(rle):
            if isinstance(mask, str):
                all_masks[:,:,idx] += rle_decode(mask)
        return all_masks

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        #print(info['id'], np.in1d(info['id'], self.segment['EncodedPixels']))
        rle = self.segment.loc[self.segment.ImageId==info['id'], 'EncodedPixels']
        #print('rle shape', rle.shape, rle)
        masks = self.masks_as_image(rle, info['width'], info['height'])
        class_ids = np.array([self.class_names.index('ship') for i in range(len(rle))])
        return masks, class_ids.astype(np.int32)
    
def continueTrainingCoCo(epochs):
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "model", "mask_rcnn_coco.h5")
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    config = train_config()
    config.display()

    dataset = ShapesDataset(-1, 0, -1)
    dataset.load_imgs(TRAIN_PATH)
    dataset.prepare()

    #print("Image Count: {}".format(len(dataset.image_ids)))
    #print("Class Count: {}".format(dataset.num_classes))
    for i, info in enumerate(dataset.class_info):
        print("{:3}. {:50}".format(i, info['name']))

    dataset_valid = ShapesDataset(-1, 1, 1)
    dataset_valid.load_imgs(VALID_PATH)
    dataset_valid.prepare()

    model = modellib.MaskRCNN(mode="training", config=config,
                            model_dir=MODEL_DIR)

    model_path = model.find_last()[1]
    if model_path:
        print("Loading weights from " + model_path)
        model.load_weights(model_path, by_name=True)
    elif config.TRAIN_FROM_COCO == 1:
        print('train from CoCo')
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
    else:
        print('train from scrash')

    print('start training');
    model.train(dataset, dataset_valid, 
                learning_rate=config.LEARNING_RATE,
                epochs=epochs, 
                #layers="heads",
                layers="all",
                argument = config.ARGUMENT)
        
def continueTrainingCoCo_kfold(epochs):
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "model", "mask_rcnn_coco.h5")
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    train_fold = [0, 1]
    for i in range(3):
        if i not in train_fold:
            print('skip fold', i)
            continue
        
        config = train_config()
        if FULLDATA:
            config.STEPS_PER_EPOCH = 10000
            config.LEARNING_RATE = config.LEARNING_RATE / 4
        config.NAME = '{}{}'.format(config.NAME, i)
        config.display()

        dataset = ShapesDataset(i, 0, -1)
        dataset.load_imgs(TRAIN_PATH)
        dataset.prepare()

        #print("Image Count: {}".format(len(dataset.image_ids)))
        #print("Class Count: {}".format(dataset.num_classes))

        dataset_valid = ShapesDataset(i, 1, 1)
        dataset_valid.load_imgs(VALID_PATH)
        dataset_valid.prepare()
        
        model = modellib.MaskRCNN(mode="training", config=config,
                                model_dir=MODEL_DIR)
        
        search_dir = os.path.join(MODEL_DIR, config.NAME)
        if not os.path.exists(search_dir):
            os.mkdir(search_dir)
        print('search_dir', search_dir)
        model_path = model.find_last(search_dir)[1]
        if model_path:
            print("Loading weights from " + model_path)
            model.load_weights(model_path, by_name=True)
        elif config.TRAIN_FROM_COCO == 1:
            print('train from CoCo')
            model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
        else:
            print('train from scrash')

        print('start training', i);
        model.train(dataset, dataset_valid, 
                    learning_rate=config.LEARNING_RATE * 2,
                    epochs=5, 
                    layers="heads",
                    argument = config.ARGUMENT)

        model_path = model.find_last(search_dir)[1]
        print("Loading weights from " + model_path)
        model.load_weights(model_path, by_name=True)
        stage1_path = os.path.join(
            model.log_dir,'{}_{}.h5'.format(config.NAME.lower(), 'heads'))


        model_path = model.find_last(search_dir)[1]
        if model_path:
            print("Loading weights from " + model_path)
            model.load_weights(model_path, by_name=True)
            
        model.train(dataset, dataset_valid, 
                    learning_rate=config.LEARNING_RATE,
                    epochs=15, 
                    layers="all",
                    argument = config.ARGUMENT)
        
        model_path = model.find_last(search_dir)[1]
        if model_path:
            print("Loading weights from " + model_path)
            model.load_weights(model_path, by_name=True)
            
        model.train(dataset, dataset_valid, 
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=50, 
                    layers="all",
                    argument = config.ARGUMENT)
        
        from keras import backend as K
        K.clear_session()
        
def loadModelPredictCoCo_kfold(valpath, plot, send):
    train_fold = [1,2]
    for i in range(3):
        if i not in train_fold:
            print('skip fold', i)
            continue
            
        inference_config = test_config()
        inference_config.display()
        inference_config.NAME = '{}{}'.format(inference_config.NAME, i)
        search_dir = os.path.join(MODEL_DIR, inference_config.NAME)
        model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
                              model_dir=MODEL_DIR)
        model_path = model.find_last()[1]
        print(model_path)
        model.load_weights(model_path, by_name=True)
        predictTests(model, inference_config, valpath, plot, 'sub_{}'.format(i), send)
        
def loadModelPredictCoCo_oof_kfold(valpath, plot):
    train_fold = [0, 1,2]
    for i in range(3):
        if i not in train_fold:
            print('skip fold', i)
            continue
            
        inference_config = test_config()
        inference_config.display()
        inference_config.NAME = '{}{}'.format(inference_config.NAME, i)
        search_dir = os.path.join(MODEL_DIR, inference_config.NAME)
        model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
                              model_dir=MODEL_DIR)
        model_path = model.find_last()[1]
        print(model_path)
        model.load_weights(model_path, by_name=True)
        predictTests(model, inference_config, valpath, plot, 'oof_{}'.format(i), False, i, 1)

def simpleValidation():
    
    inference_config = test_config()
    model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
                              model_dir=MODEL_DIR)
    
    model_path = model.find_last()[1]
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    
    dataset_valid = ShapesDataset()
    dataset_valid.load_imgs(VALID_PATH)
    dataset_valid.prepare()
    
    image_id = dataset_valid.image_ids[3]
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_valid, inference_config, 
                               image_id, use_mini_mask=True, augment=False)
    
    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)
    
    results = model.detect([original_image], verbose=1)
    
    r = results[0]
    
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                                dataset_valid.class_names, r['scores'], ax=get_ax())
    
# Basic NMS on boxes, https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python
# Malisiewicz et al.
def non_max_suppression_fast(boxes, masks, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the integer data type
    #return boxes[pick].astype("int"), pick
    return pick

def filter_lowscores(scores, limit):
    if len(scores) == 0:
        return []  
    
    pick = []
    for idx, score in enumerate(scores):
        if score >= limit:
            pick.append(idx)
    return pick

def non_max_suppression_mask_fast(masks, overlapThresh):
    if len(masks) == 0:
        #print('no mask')
        return []  
    
    if masks.shape[2] > 50:
        print('too many masks, skip non_max')
        return list(range(masks.shape[2]))
                    
    pick = []
    area = []
    for i in range(masks.shape[2]):
        area.append(np.sum(masks[:,:,i]))
    idxs = np.argsort(area)
    
    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        idxs = idxs[:-1]

        # find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
        remove_idx = []
        for j in range(len(idxs)):
            if idxs[j] == i:
                continue
            
            small_area = area[idxs[j]]
            intersection = np.sum(np.logical_and(masks[...,i], masks[...,idxs[j]]))
            overratio = intersection / small_area
            #if overratio > 0:
            #    print(overratio)
            if overratio > overlapThresh:
                remove_idx.append(j)
        
        if len(remove_idx) > 0:
            idxs = [i for j, i in enumerate(idxs) if j not in remove_idx]

    return pick


# Compute NMS (i.e. select only one box when multiple boxes overlap) for across models.
def models_cv_masks_boxes_nms(models_cv_masks_boxes, masks, threshold=0.3):
    #boxes = np.concatenate(models_cv_masks_boxes).squeeze()
    pick = non_max_suppression_fast(models_cv_masks_boxes, threshold)
    return pick

def pick_by_idx(r, pick):
    r['rois'] = r['rois'][pick]
    r['masks'] = r['masks'][:,:,pick]
    r['class_ids'] = r['class_ids'][pick]
    r['scores'] = r['scores'][pick]
    return r

def predictOne(model, dataset, config, id, plot = True):
    image_id = dataset.image_ids[id]
    scaled_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset, config, 
                               image_id, use_mini_mask=True, augment=False)
    
    file_id = dataset.image_info[image_id]['id']
    original_shape = (image_meta[1], image_meta[2])
    #print('\n{}\t{}\t'.format(file_id, original_shape))
    
    results = model.detect([scaled_image], verbose=0)
    
    r = results[0]
    
    score_pick = filter_lowscores(r['scores'], 0.8)
    r = pick_by_idx(r, score_pick)
    
    boxes = r['rois']
    masks = r['masks']
    pick = non_max_suppression_mask_fast(r['masks'], 0.2)
    
    pick_rois = r['rois'][pick]
    pick_masks = r['masks'][:,:,pick]
    pick_class_ids = r['class_ids'][pick]
    pick_scores = r['scores'][pick]
    
    # if plot: 
    #     print('non_max_suppression [{}] to [{}]'.format(boxes.shape[0], len(pick)))
    #     if boxes.shape[0] != len(pick):
    #         ax = get_ax(rows=1, cols=2, size=5)
    #         visualize.display_instances(scaled_image, r['rois'], r['masks'], r['class_ids'], 
    #                             dataset.class_names, r['scores'], ax=ax[0])
    #         visualize.display_instances(scaled_image, pick_rois, pick_masks, pick_class_ids, 
    #                             dataset.class_names, pick_scores, ax=ax[1])
    #         plt.show()
    #     else:
    #         visualize.display_instances(scaled_image, pick_rois, pick_masks, pick_class_ids, 
    #                             dataset.class_names, pick_scores, ax=get_ax())
    #         plt.show()
    
        
    r['rois'] = pick_rois
    r['masks'] = pick_masks
    r['class_ids'] = pick_class_ids
    r['scores'] = pick_scores
    
    
    #from scipy.misc import imresize
    from skimage.transform import resize
    
    rles = []
    
    window = image_meta[4:8]
    original_img = scaled_image[window[0]:window[2], window[1]:window[3], :]
    original_img = resize(original_img, original_shape)
    
    if min(r['masks'].shape) > 0:
        submasks = np.zeros((image_meta[1], image_meta[2], r['masks'].shape[2]))
        mask_record = np.ones(original_shape)
        for i in range(r['masks'].shape[2]):
            submask = r['masks'][window[0]:window[2], window[1]:window[3], i]
            area = np.sum(submask)
            
            submask = resize(submask, original_shape)
            submask = np.logical_and(submask, mask_record)
            
            if submask.shape != original_shape:
                print(submask.shape)
            mask_record = np.logical_and(mask_record, np.logical_not(submask))
            #plt.figure()
            #plt.imshow(mask_record)
            submasks[:,:,i] = submask
            rle = rle_encoding(submask)
            if len(rle) > 0:
                rles.append(rle)
            else:
                print('error')
    else:
        rles = [[]]
    
    
    return file_id, rles

def convert_result(scaled_image, r, image_meta, dataset, plot):
    boxes = r['rois']
    masks = r['masks']
    pick = non_max_suppression_mask_fast(r['masks'], 0.6)
    #print('non_max_suppression [{}] to [{}]'.format(boxes.shape[0], len(pick)))
    
    pick_rois = r['rois'][pick]
    pick_masks = r['masks'][:,:,pick]
    pick_class_ids = r['class_ids'][pick]
    pick_scores = r['scores'][pick]
    
    if plot: 
        if boxes.shape[0] != len(pick):
            ax = get_ax(rows=1, cols=2, size=5)
            visualize.display_instances(scaled_image, r['rois'], r['masks'], r['class_ids'], 
                                dataset.class_names, r['scores'], ax=ax[0])
            visualize.display_instances(scaled_image, pick_rois, pick_masks, pick_class_ids, 
                                dataset.class_names, pick_scores, ax=ax[1])
            plt.show()
        else:
            visualize.display_instances(scaled_image, pick_rois, pick_masks, pick_class_ids, 
                                dataset.class_names, pick_scores, ax=get_ax())
            plt.show()
            
    r['rois'] = pick_rois
    r['masks'] = pick_masks
    r['class_ids'] = pick_class_ids
    r['scores'] = pick_scores
    
    from skimage.transform import resize
    
    rles = []
    
    window = image_meta[4:8]
    original_img = scaled_image[window[0]:window[2], window[1]:window[3], :]
    original_shape = (image_meta[1], image_meta[2])
    original_img = resize(original_img, original_shape)
    
    if min(r['masks'].shape) > 0:
        submasks = np.zeros((image_meta[1], image_meta[2], r['masks'].shape[2]))
        mask_record = np.ones(original_shape)
        for i in range(r['masks'].shape[2]):
            submask = r['masks'][window[0]:window[2], window[1]:window[3], i]
            area = np.sum(submask)
            
            submask = resize(submask, original_shape)
            submask = np.logical_and(submask, mask_record)
            
            if submask.shape != original_shape:
                print(submask.shape)
            mask_record = np.logical_and(mask_record, np.logical_not(submask))
            #plt.figure()
            #plt.imshow(mask_record)
            submasks[:,:,i] = submask
            rle = rle_encoding(submask)
            if len(rle) > 0:
                rles.append(rle)
            else:
                print('error')
    else:
        rles = [[0, 0]]
    
    
    return rles

def predictTests_Bulk(model, inference_config, validpath, plot, cnt = None):
    print('BATCH SIZE', inference_config.IMAGES_PER_GPU)
    dataset_valid = ShapesDataset()
    dataset_valid.load_imgs(validpath)
    dataset_valid.prepare()
    
    new_test_ids = []
    rles = []

    tot = len(dataset_valid.image_ids)
    prog = tqdm(total = tot)
    for i in range(0, tot, inference_config.IMAGES_PER_GPU):
        if cnt is not None and i > cnt:
            break
        imgs = []
        img_metas = []
        file_ids = []
        for j in range(inference_config.IMAGES_PER_GPU):
            id = i + j
            image_id = dataset_valid.image_ids[id]
            file_id = dataset_valid.image_info[image_id]['id']
            scaled_image, image_meta, _, _, _ =\
                modellib.load_image_gt(dataset_valid, inference_config, 
                                       image_id, use_mini_mask=True, augment=False)
            file_ids.append(file_id)
            imgs.append(scaled_image)
            img_metas.append(image_meta)

        results = model.detect(imgs, verbose=0)
        for i in range(len(results)):
            rle = convert_result(imgs[i], results[i], img_metas[i], dataset_valid, plot)
            file_id = file_ids[i]

            rles.extend(rle)
            new_test_ids.extend([file_id] * len(rle))
            
        prog.update(inference_config.IMAGES_PER_GPU)

    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv('../result/{}_org.csv'.format(inference_config.NAME), index=False)
    kaggle_util.save_result(sub, '../result/{}.csv'.format(inference_config.NAME), competition = 'airbus-ship-detection', send = True, index = False)
    return sub

def predictTests(model, config, validpath, plot, flex = 'sub', send = False, fold = -1, valid = -1):
    print('fold', fold, 'valid', valid)
    dataset_valid = ShapesDataset(fold,valid, -1)
    if fold >= 0 and valid == 1:
        dataset_valid.load_imgs(validpath)
    else:
        dataset_valid.load_test_imgs(validpath)
    dataset_valid.prepare()

    #file_id, rle = predictOne(model, dataset_valid, inference_config, 1)

    new_test_ids = []
    rles = []
        
    for i in tqdm(range(len(dataset_valid.image_ids))):
        file_id, rle = predictOne(model, dataset_valid, config, i, 
                                  plot = plot)
        rles.extend(rle)
        new_test_ids.extend([file_id] * len(rle))
        
        #if i > 10:
        #    break
            
    from keras import backend as K
    K.clear_session()
        
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
        
    sub.to_csv('../result/{}_{}.csv'.format(flex, config.NAME), index=False)
    kaggle_util.save_result(sub, '../result/{}_{}.csv'.format(config.NAME, flex), competition = 'airbus-ship-detection', send = send, index = False)
    return sub

def loadModelPredict():

    inference_config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
                              model_dir=MODEL_DIR)
    
    model_path = model.find_last()[1]
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    
    predictTests(model, inference_config, TEST_PATH)
    #predictTests_Bulk(model, inference_config, TEST_PATH, False)

def loadModelPredictCoCo(valpath, plot, flex, send, model_path = None):
    
    inference_config = test_config()
    inference_config.display()
    #inference_config = CoCoInferenceConfig()
    model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
                              model_dir=MODEL_DIR)
    
    if model_path is None:
        model_path = model.find_last()[1]
        print(model_path)
        assert model_path != "", "Provide path to trained weights"
        print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    
    return predictTests(model, inference_config, valpath, plot, flex, send)
    #return predictTests_Bulk(model, inference_config, TEST_PATH, False)
    
def checkHandyMask():
    dataset = ShapesDataset()
    dataset.load_imgs('input/stage2_new/')
    dataset.prepare()

    for i in range(len(dataset.image_ids)):
        id = dataset.image_ids[i]
        image = dataset.load_image(id)
        masks = dataset.load_mask(id)[0] > 0
        for i in range(masks.shape[-1]):
            plt.imshow(masks[...,i])
            plt.show()
    
"""
image_id = np.random.choice(dataset.image_ids, 1)[0]
image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
    dataset, config, image_id, augment=True, use_mini_mask=True)
log("mask", mask)
display_images([image]+[mask[:,:,i] for i in range(min(mask.shape[-1], 7))])
"""
if __name__ == "__main__":
    #continueTraining(30)
    #loadModelPredict()

#     loadModelPredictCoCo_oof_kfold(TRAIN_PATH, False)
#     loadModelPredictCoCo_kfold(TEST_PATH, False, True)
    
    continueTrainingCoCo_kfold(100)
#     loadModelPredictCoCo(TEST_PATH, False, 'sub', True)
    #with tf.device('/cpu:0'):
    
    #loadModelPredictCoCo(TEST_PATH, False, 'sub', True)

    #checkHandyMask()
