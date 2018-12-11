# -*- coding: utf-8 -*-
from os import path, mkdir
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import timeit
import cv2
from tqdm import tqdm
from skimage import measure
from multiprocessing import Pool
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.neighbors import KDTree
from skimage.morphology import watershed
from skimage.morphology import square, dilation
import pandas as pd
import math
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook

def get_mask(img_id, df):
    shape = (768,768)
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    masks = df.loc[img_id]['EncodedPixels']
    if(type(masks) == float): return img.reshape(shape)
    if(type(masks) == str): masks = [masks]
    for mask in masks:
        s = mask.split()
        for i in range(len(s)//2):
            start = int(s[2*i]) - 1
            length = int(s[2*i+1])
            img[start:start+length] = 1
    return img.reshape(shape).T

def extend_neighbor_features(inp, cur_prop, pred_props, neighbors, med_area, max_area):
    inp.append(neighbors.shape[0])
    median_area = med_area
    maximum_area = max_area
    if neighbors.shape[0] > 0:
        neighbors_areas = np.asarray([pred_props[j].area for j in neighbors])
        median_area = np.median(neighbors_areas)
        maximum_area = np.max(neighbors_areas)
    inp.append(median_area)
    inp.append(maximum_area)
    inp.append(cur_prop.area / median_area)
    inp.append(cur_prop.area / maximum_area)
    return inp

def get_inputs(imageid, pre_df, seg_df = None):
    inputs = []    
    
    pred_msk = get_mask(imageid, pre_df)
    y_pred = measure.label(pred_msk, neighbors=8, background=0)
    ship_pro = pre_df.loc[imageid, 'p_ship'].mean()
    
    props = measure.regionprops(y_pred)
    for i in range(len(props)):
        if props[i].area < 10:
            y_pred[y_pred == i+1] = 0
    pred_labels = measure.label(y_pred, neighbors=8, background=0)
    pred_props = measure.regionprops(y_pred)
    init_count = len(pred_props)

    coords = [pr.centroid for pr in pred_props]
#     print('len(coords)', len(coords))
    if len(coords) > 0:
#         print('make neighbors')
        t = KDTree(coords)
        neighbors100 = t.query_radius(coords, r=50)
        neighbors200 = t.query_radius(coords, r=100)
        neighbors300 = t.query_radius(coords, r=150)
        neighbors400 = t.query_radius(coords, r=200)
        areas = np.asarray([pr.area for pr in props])
        med_area = np.median(areas)
        max_area = np.max(areas)
    
    for i in range(len(pred_props)):
        cur_prop = pred_props[i]
        is_on_border = 1 * ((cur_prop.bbox[0] <= 1) | (cur_prop.bbox[1] <= 1) | (cur_prop.bbox[2] >= y_pred.shape[0] - 1) | (cur_prop.bbox[3] >= y_pred.shape[1] - 1))
  
        msk_reg = pred_labels[cur_prop.bbox[0]:cur_prop.bbox[2], cur_prop.bbox[1]:cur_prop.bbox[3]] == i+1
        pred_reg = y_pred[cur_prop.bbox[0]:cur_prop.bbox[2], cur_prop.bbox[1]:cur_prop.bbox[3]]
        
        contours = cv2.findContours((msk_reg * 255).astype(dtype=np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours[1]) > 0:
            cnt = contours[1][0]
            min_area_rect = cv2.minAreaRect(cnt)
        
        inp = []
        inp.append(ship_pro)
        inp.append(cur_prop.area)
        inp.append(cur_prop.area / med_area)
        inp.append(cur_prop.area / max_area)
        if len(contours[1]) > 0:
            inp.append(cv2.isContourConvex(cnt) * 1.0)
            inp.append(min(min_area_rect[1]))
            inp.append(max(min_area_rect[1]))
            if max(min_area_rect[1]) > 0:
                inp.append(min(min_area_rect[1]) / max(min_area_rect[1]))
            else:
                inp.append(0)
            inp.append(min_area_rect[2])
        else:
            inp.append(0)
            inp.append(0)
            inp.append(0)
            inp.append(0)
            inp.append(0)
        inp.append(cur_prop.convex_area)
        inp.append(cur_prop.solidity)
        inp.append(cur_prop.eccentricity)
        inp.append(cur_prop.extent)
        inp.append(cur_prop.perimeter)
        inp.append(cur_prop.major_axis_length)
        inp.append(cur_prop.minor_axis_length)
        if(cur_prop.minor_axis_length > 0):
            inp.append(cur_prop.minor_axis_length / cur_prop.major_axis_length)
        else:
            inp.append(0)
            
        inp.append(cur_prop.euler_number)
        inp.append(cur_prop.equivalent_diameter)
        inp.append(cur_prop.perimeter ** 2 / (4 * cur_prop.area * math.pi))
        
        inp.append(is_on_border)        
        inp.append(init_count)
        inp.append(med_area)
        inp.append(cur_prop.area / med_area)

        inp = extend_neighbor_features(inp, cur_prop, pred_props, neighbors100[i], med_area, max_area)
        inp = extend_neighbor_features(inp, cur_prop, pred_props, neighbors200[i], med_area, max_area)
        inp = extend_neighbor_features(inp, cur_prop, pred_props, neighbors300[i], med_area, max_area)
        inp = extend_neighbor_features(inp, cur_prop, pred_props, neighbors400[i], med_area, max_area)
        
        
        inputs.append(np.asarray(inp))
        
    inputs = np.asarray(inputs)
    if seg_df is None:
        return inputs, pred_labels
    else:
        outputs = []
        truth_labels = get_mask(imageid, seg_df)
        truth_labels = measure.label(truth_labels, neighbors=8, background=0)
        truth_props = measure.regionprops(truth_labels)
        
        m = np.zeros((len(pred_props), len(truth_props)))
        
        for x in range(pred_labels.shape[1]):
            for y in range(pred_labels.shape[0]):
                if pred_labels[y, x] > 0 and truth_labels[y, x] > 0:
                    m[pred_labels[y, x]-1, truth_labels[y, x]-1] += 1
                    
        truth_used = set([])
        for i in range(len(pred_props)): 
            max_iou = 0
            for j in range(len(truth_props)):
                if m[i, j] > 0:
                    iou = m[i, j] / (pred_props[i].area + truth_props[j].area - m[i, j])
                    if iou > max_iou:
                        max_iou = iou
                    if iou > 0.5:
                        truth_used.add(j)
            if max_iou <= 0.5:
                max_iou = 0
            outputs.append(max_iou)
            
        outputs = np.asarray(outputs)
        fn = len(truth_props) - len(truth_used)
        
        return inputs, pred_labels, outputs, fn