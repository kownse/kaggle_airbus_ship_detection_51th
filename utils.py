from os import path, mkdir
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import tensorflow as tf
tf.set_random_seed(1)
import timeit
import cv2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler #, TensorBoard
from models import get_densenet121_unet_softmax, dice_coef_rounded_ch0, dice_coef_rounded_ch1, schedule_steps, softmax_dice_loss
import keras.backend as K
import pandas as pd
from tqdm import tqdm
from transforms import aug_mega_hardcore
from keras import metrics
from abc import abstractmethod
from keras.preprocessing.image import Iterator
import time
from skimage import measure
from skimage.morphology import square, erosion, dilation, watershed
from skimage.filters import median
import os
# from keras.applications.inception_resnet_v2 import preprocess_input

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

images_folder = '../../input/all/'
labels_folder = '../../input/labels_all'
masks_folder = '../../input/masks_all'
test_folder = '../../input/test'
models_folder = 'nn_models'

model_path_root = 'inception_softmax_meaninputs'

#IMG_SIZE = 384
#input_shape = (IMG_SIZE, IMG_SIZE)

exclude_list = ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg', 
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg']

from fastai.dataset import *

class Gaussian(Transform):
    def __init__(self, prob, strength, tfm_y=TfmType.NO):
        super().__init__(tfm_y)
        self.prob = prob
        self.strength = strength
        self.store.apply_transform = False

    def set_state(self):
        self.store.apply_transform = random.random() < self.prob
        self.store.strength = random.randint(0, self.strength)

    def do_transform(self, image, is_y):
        if self.store.apply_transform:
            for i in range(image.shape[2]):
                image[...,i] += cv2.randu(np.zeros(image.shape[:-1], dtype=np.int16), 
                                          (0), 
                                          (self.store.strength)).astype(np.uint8)
        return image

def preprocess_input(x):
    x = np.asarray(x, dtype='float32')
    x /= 127.5
    x -= 1.
    return x

def bgr_to_lab(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(17, 17))
    lab = clahe.apply(lab[:, :, 0])
    if lab.mean() > 127:
        lab = 255 - lab
    return lab[..., np.newaxis]

def create_mask(labels):
    labels = measure.label(labels, neighbors=8, background=0)
    tmp = dilation(labels > 0, square(9))    
    tmp2 = watershed(tmp, labels, mask=tmp, watershed_line=True) > 0
    tmp = tmp ^ tmp2
    tmp = dilation(tmp, square(7))
    msk = (255 * tmp).astype('uint8')
    
    props = measure.regionprops(labels)
    msk0 = 255 * (labels > 0)
    msk0 = msk0.astype('uint8')
    
    msk1 = np.zeros_like(labels, dtype='bool')
    
    max_area = np.max([p.area for p in props])
    
    for y0 in range(labels.shape[0]):
        for x0 in range(labels.shape[1]):
            if not tmp[y0, x0]:
                continue
            if labels[y0, x0] == 0:
                if max_area > 4000:
                    sz = 6
                else:
                    sz = 3
            else:
                sz = 3
                if props[labels[y0, x0] - 1].area < 300:
                    sz = 1
                elif props[labels[y0, x0] - 1].area < 2000:
                    sz = 2
            uniq = np.unique(labels[max(0, y0-sz):min(labels.shape[0], y0+sz+1), max(0, x0-sz):min(labels.shape[1], x0+sz+1)])
            if len(uniq[uniq > 0]) > 1:
                msk1[y0, x0] = True
                msk0[y0, x0] = 0
    
    msk1 = 255 * msk1
    msk1 = msk1.astype('uint8')
    
    msk2 = np.zeros_like(labels, dtype='uint8')
    msk = np.stack((msk0, msk1, msk2))
    msk = np.rollaxis(msk, 0, 3)
    return msk

class BaseMaskDatasetIterator(Iterator):
    def __init__(self,
                 image_table,
                 input_shape,
                 random_transformers=None,
                 batch_size=8,
                 shuffle=True,
                 seed=None,
                 ):
        self.input_shape = input_shape
        self.image_table = image_table
        self.random_transformers = random_transformers
        if seed is None:
            seed = np.uint32(time.time() * 1000)

        super(BaseMaskDatasetIterator, self).__init__(len(self.image_table), batch_size, shuffle, seed)

    @abstractmethod
    def transform_mask(self, mask, image):
        raise NotImplementedError

    def transform_batch_y(self, batch_y):
        return batch_y

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = []
        batch_y = []
        
        #print(index_array)

        for batch_index, image_index in enumerate(index_array):
            row = self.image_table.iloc[image_index]
            img_id = row.ImageId
            
            if img_id in exclude_list:
                continue
            
            img0 = cv2.imread(path.join(images_folder, '{0}'.format(img_id)), cv2.IMREAD_COLOR)
            mask_path = path.join(masks_folder, '{0}.png'.format(img_id[:-4]))
            if not os.path.exists(mask_path):
                msk0 = np.zeros((768,768,3))
                lbl0 = np.zeros((768,768))
            else:
                msk0 = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                lbl0 = cv2.imread(path.join(labels_folder, '{0}.tif'.format(img_id[:-4])), cv2.IMREAD_UNCHANGED)
                
                if img0.shape[0] == 0 or msk0.shape[0] == 0 or lbl0.shape[0] == 0:
                    img0 = np.zeros((768, 768,3))
                    msk0 = np.zeros((768, 768,3))
                    lbl0 = np.zeros((768, 768))
#             print('mask_path', mask_path)
#             print('img_shape', img0.shape, 'mask_shape', msk0.shape)
            tmp = np.zeros_like(msk0[..., 0], dtype='uint8')
            tmp[1:-1, 1:-1] = msk0[1:-1, 1:-1, 0]
            good4copy = list(set(np.unique(lbl0[lbl0 > 0])).symmetric_difference(np.unique(lbl0[(lbl0 > 0) & (tmp == 0)])))
    
            x0 = random.randint(0, img0.shape[1] - self.input_shape[1])
            y0 = random.randint(0, img0.shape[0] - self.input_shape[0])
            img = img0[y0:y0+self.input_shape[0], x0:x0+self.input_shape[1], :]
            msk = msk0[y0:y0+self.input_shape[0], x0:x0+self.input_shape[1], :]
            
            if len(good4copy) > 0 and random.random() > 0.75:
                num_copy = random.randrange(1, min(6, len(good4copy)+1))
                lbl_max = lbl0.max()
                for i in range(num_copy):
                    lbl_max += 1
                    l_id = random.choice(good4copy)
                    lbl_msk = lbl0 == l_id
                    row, col = np.where(lbl_msk)
                    y1, x1 = np.min(np.where(lbl_msk), axis=1)
                    y2, x2 = np.max(np.where(lbl_msk), axis=1)
                    lbl_msk = lbl_msk[y1:y2+1, x1:x2+1]
                    lbl_img = img0[y1:y2+1, x1:x2+1, :]
                    if random.random() > 0.5:
                        lbl_msk = lbl_msk[:, ::-1, ...]
                        lbl_img = lbl_img[:, ::-1, ...]
                    rot = random.randrange(4)
                    if rot > 0:
                        lbl_msk = np.rot90(lbl_msk, k=rot)
                        lbl_img = np.rot90(lbl_img, k=rot)
                    x1 = random.randint(max(0, x0 - lbl_msk.shape[1] // 2), min(img0.shape[1] - lbl_msk.shape[1], x0 + self.input_shape[1] - lbl_msk.shape[1] // 2))
                    y1 = random.randint(max(0, y0 - lbl_msk.shape[0] // 2), min(img0.shape[0] - lbl_msk.shape[0], y0 + self.input_shape[0] - lbl_msk.shape[0] // 2))
                    tmp = erosion(lbl_msk, square(5))
                    lbl_msk_dif = lbl_msk ^ tmp
                    tmp = dilation(lbl_msk, square(5))
                    lbl_msk_dif = lbl_msk_dif | (tmp ^ lbl_msk)
                    lbl0[y1:y1+lbl_msk.shape[0], x1:x1+lbl_msk.shape[1]][lbl_msk] = lbl_max
                    img0[y1:y1+lbl_msk.shape[0], x1:x1+lbl_msk.shape[1]][lbl_msk] = lbl_img[lbl_msk]
                    full_diff_mask = np.zeros_like(img0[..., 0], dtype='bool')
                    full_diff_mask[y1:y1+lbl_msk.shape[0], x1:x1+lbl_msk.shape[1]] = lbl_msk_dif
                    img0[..., 0][full_diff_mask] = median(img0[..., 0], mask=full_diff_mask)[full_diff_mask]
                    img0[..., 1][full_diff_mask] = median(img0[..., 1], mask=full_diff_mask)[full_diff_mask]
                    img0[..., 2][full_diff_mask] = median(img0[..., 2], mask=full_diff_mask)[full_diff_mask]
                img = img0[y0:y0+self.input_shape[0], x0:x0+self.input_shape[1], :]
                lbl = lbl0[y0:y0+self.input_shape[0], x0:x0+self.input_shape[1]]
                msk = create_mask(lbl)
                
#             return img, msk
            data = self.random_transformers[0](image=img[..., ::-1], mask=msk)
                
            img = data['image'][..., ::-1]
            msk = data['mask']
            
            msk = msk.astype('float')
            msk[..., 0] = (msk[..., 0] > 127) * 1
            msk[..., 1] = (msk[..., 1] > 127) * (msk[..., 0] == 0) * 1
            msk[..., 2] = (msk[..., 1] == 0) * (msk[..., 0] == 0) * 1
            otp = msk

            img = np.concatenate([img, bgr_to_lab(img)], axis=2)
            batch_x.append(img)
            batch_y.append(otp)
        batch_x = np.array(batch_x, dtype="float32")
        batch_y = np.array(batch_y, dtype="float32")
        batch_x = preprocess_input(batch_x)
        return self.transform_batch_x(batch_x), self.transform_batch_y(batch_y)

    def transform_batch_x(self, batch_x):
        return batch_x

    def next(self):

        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)    
    
def align_img(img):
    x0 = 16
    y0 = 16
    x1 = 16
    y1 = 16
    if (img.shape[1] % 32) != 0:
        x0 = int((32 - img.shape[1] % 32) / 2)
        x1 = (32 - img.shape[1] % 32) - x0
        x0 += 16
        x1 += 16
    if (img.shape[0] % 32) != 0:
        y0 = int((32 - img.shape[0] % 32) / 2)
        y1 = (32 - img.shape[0] % 32) - y0
        y0 += 16
        y1 += 16
    img0 = np.pad(img, ((y0,y1), (x0,x1), (0, 0)), 'symmetric')
    img0 = np.concatenate([img0, bgr_to_lab(img0)], axis=2)
    return img0

def train_data_generator(val_df, batch_size, input_shape):
    aug_tfms = [
            RandomFlip(),
            RandomDihedral(tfm_y=TfmType.NO),
#             RandomBlur(5, probability=0.5,tfm_y=TfmType.NO),
#             RandomLighting(0.1, 0.5),
#             Gaussian(0.5, 50),
        ]
    
    while True:
        inputs = []
        outputs = []
        step_id = 0
        cnt = 0
        for idx, row in val_df.iterrows():
            img_id = row.ImageId
            imgpath = os.path.join(images_folder, '{0}'.format(img_id))
#             print(imgpath)
            img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
            
            mask_path = os.path.join(masks_folder, '{0}.png'.format(img_id[:-4]))
            if not os.path.exists(mask_path):
                msk = np.zeros((input_shape[0],input_shape[1],3))
            else:
                msk = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                
            img = cv2.resize(img, input_shape)
            msk = cv2.resize(msk, input_shape)
            
            img = np.concatenate([img, bgr_to_lab(img)], axis=2)
            
            rimg = img
            rmask = msk
            for trans in aug_tfms:
                trans.set_state()
                trans.apply_transform = True
                # print(rimg.dtype, rimg.max(), rimg.min())
                if type(trans) == RandomLighting:
                    rimg = rimg.astype(np.float32) / 255

                rimg = trans.do_transform(rimg, False)

                if type(trans) == RandomLighting:
                    rimg = (rimg * 255).astype(np.int16)

                #print(rimg.dtype, rimg.max(), rimg.min())
                y_trans = [Cutout, RandomRotate, RandomDihedral, RandomFlip]
                if type(trans) in y_trans:
                    #print('transform rotation')
                    rmask = trans.do_transform(rmask, True)
            img = rimg
            msk = rmask
            img = preprocess_input(img)
#             print(img.min(), img.max(), img.shape)
            
            msk = msk.astype('float')
            msk[..., 0] = (msk[..., 0] > 127) * 1
            msk[..., 1] = (msk[..., 1] > 127) * (msk[..., 0] == 0) * 1
            msk[..., 2] = (msk[..., 1] == 0) * (msk[..., 0] == 0) * 1
            otp = msk
            
            inputs.append(img)
            outputs.append(otp)
         
            if len(inputs) == batch_size:
                step_id += 1
                inputs = np.asarray(inputs)
                outputs = np.asarray(outputs, dtype='float')
#                 for img in inputs:
#                     print(img.min(), img.max(), img.shape)
                yield inputs, outputs
                inputs = []
                outputs = []
    
def val_data_generator(val_df, batch_size, validation_steps, input_shape):
    while True:
        inputs = []
        outputs = []
        step_id = 0
        for idx, row in val_df.iterrows():
            img_id = row.ImageId
            imgpath = os.path.join(images_folder, '{0}'.format(img_id))
#             print(imgpath)
            img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
            
            mask_path = os.path.join(masks_folder, '{0}.png'.format(img_id[:-4]))
            if not os.path.exists(mask_path):
                msk = np.zeros((input_shape[0], input_shape[0],3))
            else:
                msk = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                
            img = cv2.resize(img, input_shape)
            msk = cv2.resize(msk, input_shape)
            
            msk = msk.astype('float')
            msk[..., 0] = (msk[..., 0] > 127) * 1
            msk[..., 1] = (msk[..., 1] > 127) * (msk[..., 0] == 0) * 1
            msk[..., 2] = (msk[..., 1] == 0) * (msk[..., 0] == 0) * 1
            otp = msk
            
            img = np.concatenate([img, bgr_to_lab(img)], axis=2)
            img = preprocess_input(img)
#             print(img.min(), img.max(), img.shape)
            
            inputs.append(img)
            outputs.append(otp)
            
            if len(inputs) == batch_size:
                step_id += 1
                inputs = np.asarray(inputs)
                outputs = np.asarray(outputs, dtype='float')
                yield inputs, outputs
                inputs = []
                outputs = []
                if step_id == validation_steps:
                    break
                

def is_grayscale(image):
    return np.allclose(image[..., 0], image[..., 1], atol=0.001) and np.allclose(image[..., 1], image[..., 2], atol=0.001)


def train_data_generator_sigmoid(val_df, batch_size, input_shape):
    aug_tfms = [
            RandomFlip(),
            RandomDihedral(tfm_y=TfmType.NO),
#             RandomBlur(5, probability=0.5,tfm_y=TfmType.NO),
#             RandomLighting(0.1, 0.5),
#             Gaussian(0.5, 50),
        ]
    
    while True:
        inputs = []
        outputs = []
        step_id = 0
        cnt = 0
        for idx, row in val_df.iterrows():
            img_id = row.ImageId
            imgpath = os.path.join(images_folder, '{0}'.format(img_id))
#             print(imgpath)
            img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
            
            mask_path = os.path.join(masks_folder, '{0}.png'.format(img_id[:-4]))
            if not os.path.exists(mask_path):
                msk = np.zeros(input_shape)
            else:
                msk = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)[...,0]
                
            img = cv2.resize(img, input_shape)
            msk = cv2.resize(msk, input_shape)
            
            msk = msk[..., np.newaxis]
            
            
            img = np.concatenate([img, bgr_to_lab(img)], axis=2)
            
            rimg = img
            rmask = msk
            for trans in aug_tfms:
                trans.set_state()
                trans.apply_transform = True
                # print(rimg.dtype, rimg.max(), rimg.min())
                if type(trans) == RandomLighting:
                    rimg = rimg.astype(np.float32) / 255

                rimg = trans.do_transform(rimg, False)

                if type(trans) == RandomLighting:
                    rimg = (rimg * 255).astype(np.int16)

                #print(rimg.dtype, rimg.max(), rimg.min())
                y_trans = [Cutout, RandomRotate, RandomDihedral, RandomFlip]
                if type(trans) in y_trans:
                    #print('transform rotation')
                    rmask = trans.do_transform(rmask, True)
            img = rimg
            msk = rmask
            img = preprocess_input(img)
#             print(img.min(), img.max(), img.shape)
            
            msk = msk.astype('float')
            msk = (msk > 127) * 1
            otp = msk
            
            inputs.append(img)
            outputs.append(otp)
         
            if len(inputs) == batch_size:
                step_id += 1
                inputs = np.asarray(inputs)
                outputs = np.asarray(outputs, dtype='float')
#                 for img in inputs:
#                     print(img.min(), img.max(), img.shape)
                yield inputs, outputs
                inputs = []
                outputs = []
    
def val_data_generator_sigmoid(val_df, batch_size, validation_steps, input_shape):
    while True:
        inputs = []
        outputs = []
        step_id = 0
        for idx, row in val_df.iterrows():
            img_id = row.ImageId
            imgpath = os.path.join(images_folder, '{0}'.format(img_id))
#             print(imgpath)
            img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
            
            mask_path = os.path.join(masks_folder, '{0}.png'.format(img_id[:-4]))
            if not os.path.exists(mask_path):
                msk = np.zeros(input_shape)
            else:
                msk = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)[...,0]
                
            img = cv2.resize(img, input_shape)
            msk = cv2.resize(msk, input_shape)
            
            msk = msk[..., np.newaxis]
            
            msk = msk.astype('float')
            msk = (msk > 127) * 1
            otp = msk
            
            img = np.concatenate([img, bgr_to_lab(img)], axis=2)
            img = preprocess_input(img)
#             print(img.min(), img.max(), img.shape)
            
            inputs.append(img)
            outputs.append(otp)
            
            if len(inputs) == batch_size:
                step_id += 1
                inputs = np.asarray(inputs)
                outputs = np.asarray(outputs, dtype='float')
                yield inputs, outputs
                inputs = []
                outputs = []
                if step_id == validation_steps:
                    break
                    
def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch]>0, B[batch]>0
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10 )/ (np.sum(union > 0) + 1e-10)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)

def my_iou_metric(label, pred):
    return tf.py_func(get_iou_vector, [label, pred>0.5], tf.float64)

def my_iou_metric_2(label, pred):
    return tf.py_func(get_iou_vector, [label, pred >0], tf.float64)

# code download from: https://github.com/bermanmaxim/LovaszSoftmax
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


# --------------------------- BINARY LOSSES ---------------------------

def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        #loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        loss = tf.tensordot(tf.nn.elu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name="loss"
                   )
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels

def lovasz_loss(y_true, y_pred):
    y_true, y_pred = K.cast(K.squeeze(y_true, -1), 'int32'), K.cast(K.squeeze(y_pred, -1), 'float32')
    #logits = K.log(y_pred / (1. - y_pred))
    logits = y_pred #Jiaxin
    loss = lovasz_hinge(logits, y_true, per_image = True, ignore = None)
    return loss

def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
    
def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)
