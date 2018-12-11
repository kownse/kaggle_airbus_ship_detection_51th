#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 11:32:42 2018

@author: kownse
"""

import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import pickle
import numpy as np
import gc

# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in tqdm(df.columns):
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    gc.collect()
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def replaceCatWithDummy(df, catname):
    dummies = pd.get_dummies(df[catname], prefix=catname)
    df = df.drop(catname, axis=1)
    gc.collect()
    return pd.concat([df, dummies], axis=1)

def replaceCatWithDummy_bulk(df, cols):
    for col in cols:
        df = replaceCatWithDummy(df, col)
        gc.collect()
    return df

def save_result(sub, filename, competition = '', send = False, index = False):
    print('save result')
    sub.to_csv(filename, index=index, float_format='%.5f')
    
    if send and len(competition) > 0:
        file_7z = '{}.7z'.format(filename)
        os.system('7z a {} {}'.format(file_7z, filename))
        
        print('upload result')
        command = 'kaggle competitions submit -c {} -f {} -m "submit"'.format(competition, file_7z)
        print('cmd: ' + command)
        os.system(command)
        
def read_result(path, idx, score_col = 'deal_probability'):
    compression = None
    if '.gz' in path:
        compression='gzip'
    return pd.read_csv('../result/' + path, compression=compression).rename(columns={score_col: 'p{}'.format(idx)})
    
def ensemble(result_list, send, competition = '', score_col = 'deal_probability', prefix = 'ensemble'):
    print('score_col ', score_col)
    sub = read_result(result_list[0][0], 0, score_col = score_col)
    sub['p0'] *= result_list[0][1]
    for i in tqdm(range(1, len(result_list))):
        res = read_result(result_list[i][0], 0, score_col = score_col)
        sub['p{}'.format(i)] = res['p0'] * result_list[i][1]
    
    print(sub.corr())
    #return

    sub[score_col] = 0 #(sub['p0'] + sub['p1'] + sub['p2']) / 3
    for i in tqdm(range(len(result_list))):
        sub[score_col] += sub['p{}'.format(i)]
        sub.drop('p{}'.format(i), axis = 1, inplace = True)

    str_now = datetime.now().strftime("%m-%d-%H-%M")
    filename = '../result/{}_{}.csv'.format(prefix, str_now)
    save_result(sub, filename, competition = competition, send = send)
    
def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
        
def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def drop_constant_feature(df):
    feats_counts = df.nunique(dropna = False)
    const_feats = feats_counts.loc[feats_counts<=1].index.tolist()
    df = df.drop(const_feats, axis = 1)
    return df, const_feats

def remove_duplicate(df_ori):
    cats = list(df_ori.select_dtypes(include=['object']).columns)
    df = df_ori[cats].fillna('NaN')
    train_enc = pd.DataFrame(index = df.index)
    
    for col in tqdm(df.columns):
        df[col] = df[col].factorize()[0]
        
    dup_cols = {}
    for i, c1 in enumerate(tqdm(df.columns)):
        for c2 in df.columns[i + 1:]:
            if c2 not in dup_cols and np.all(df[c1] == df[c2]):
                dup_cols[c2] = c1

    return df_ori.drop(dup_cols.keys(), axis=1), dup_cols

def get_cat_num(df):
    cat_cols = list(df.select_dtypes(include=['object']).columns)
    num_cols = list(df.select_dtypes(exclude=['object']).columns)
    return cat_cols, num_cols

def autolabel(arrayA):
    ''' label each colored square with the corresponding data value. 
    If value > 20, the text is in black, else in white.
    '''
    arrayA = np.array(arrayA)
    for i in range(arrayA.shape[0]):
        for j in range(arrayA.shape[1]):
                plt.text(j,i, "%.2f"%arrayA[i,j], ha='center', va='bottom',color='w')

def hist_it(feat):
    plt.figure(figsize=(16,4))
    feat[Y==0].hist(bins=range(int(feat.min()),int(feat.max()+2)),normed=True,alpha=0.8)
    feat[Y==1].hist(bins=range(int(feat.min()),int(feat.max()+2)),normed=True,alpha=0.5)
    plt.ylim((0,1))
    
def gt_matrix(feats,sz=16):
    a = []
    for i,c1 in enumerate(feats):
        b = [] 
        for j,c2 in enumerate(feats):
            mask = (~train[c1].isnull()) & (~train[c2].isnull())
            if i>=j:
                b.append((train.loc[mask,c1].values>=train.loc[mask,c2].values).mean())
            else:
                b.append((train.loc[mask,c1].values>train.loc[mask,c2].values).mean())

        a.append(b)

    plt.figure(figsize = (sz,sz))
    plt.imshow(a, interpolation = 'None')
    _ = plt.xticks(range(len(feats)),feats,rotation = 90)
    _ = plt.yticks(range(len(feats)),feats,rotation = 0)
    autolabel(a)
    
def get_zscore(df, col, log = False):
    values = np.log1p(df[col]) if log else df[col]
    mean = values.mean()
    std = values.std()
    return (values - mean) / std

def fix_outlier(df, zscore_limit = 5):
    num_cols = list(df.select_dtypes(exclude=['object']).columns)
    num_cols = [col for col in num_cols if df[col].nunique() > 1000 
                and df[col].max() > 1000 
                and df[col].skew() > 100]
    
    for idx, col in enumerate(num_cols):
        zscore = get_zscore(df, col, True )
        if zscore.max() > zscore_limit:
            limit = df[col][zscore<=zscore_limit].max()
            print(col, (zscore>zscore_limit).sum(), )
            df[col][zscore > zscore_limit] = limit
    return df

def masks_as_color(in_mask_list):
    # Take the individual ship masks and create a color mask array for each ships
    all_masks = np.zeros((768, 768), dtype = np.float)
    scale = lambda x: (len(in_mask_list)+x+1) / (len(in_mask_list)*2) ## scale the heatmap image to shift 
    for i,mask in enumerate(in_mask_list):
        if isinstance(mask, str):
            all_masks[:,:] += scale(i) * rle_decode(mask)
            
    #print(in_mask_list, len(in_mask_list), 'ships')
    return all_masks

def show_test_result_by_imgid(imgid, test_set):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))
    c_img = cv2.imread(os.path.join(TEST_PATH, imgid))
    #print(c_img.shape)
    c_img = np.expand_dims(c_img, 0)/255.0
    ax1.imshow(c_img[0])
    ax1.set_title('Image: ' + imgid)
    ax2.imshow(masks_as_color(test_set.query('ImageId==\"{}\"'.format(imgid))['EncodedPixels']))
    
def show_test_result_bulk(test_set, imglst, TOP_PREDICTIONS = 2):
    for imgid in imglst[:TOP_PREDICTIONS]:
        show_test_result_by_imgid(imgid, test_set)
        
def merge_ship(df, df_new, hascol = 'hasship'):
    df_new[hascol] = df_new.EncodedPixels.notna().astype(np.int8)
    
    ship_cnt = df_new.groupby('ImageId')[hascol].agg('sum')
    
    test_hasship = df_new[['ImageId', hascol]].drop_duplicates()
    test_hasship.set_index('ImageId', inplace = True)
    test_hasship['{}_cnt'.format(hascol)] = ship_cnt
    test_hasship = test_hasship.reset_index()
    
    return pd.merge(df, test_hasship, on='ImageId', how='left')

def kfold_mean_encoding(df, cols, tarcol, n_splits = 5):
    from sklearn.model_selection import KFold
    global_mean = df[tarcol].mean()
    for col in cols:
       df[col + 'kf_mean'] = global_mean

    folds = KFold(n_splits=n_splits, shuffle=False)
    for train_index, val_index in folds.split(df):
        X_tr, X_val = df.iloc[train_index], df.iloc[val_index]
        for col in cols:
            means = X_val[col].map(X_tr.groupby(col)[tarcol].mean())
            X_val[col + 'kf_mean'] = means
        df.iloc[val_index] = X_val
    return df

def loo_mean_encoding(df, cols, tarcol):
    global_mean = df[tarcol].mean()
    df_tot = pd.DataFrame()
    for col in cols:
        agg = all_data.groupby(col)[tarcol].agg(['sum','count', 'mean'])
        df_tot[col + '_sum'] = agg['sum']
        df_tot[col + '_cnt'] = agg['count']
        df_tot[col + '_mean'] = agg['mean']
        df_tot = df_tot.reset_index()
        df_combine = pd.merge(df, df_tot, on = col, how = 'left')
        df_combine[col + '_loo_mean'] = (df_combine[col + '_sum'] - df_combine[tarcol]) / (df_combine[col + '_cnt'] - 1)
        df_combine[col + '_loo_mean'].fillna(global_mean, inplace = True)
    return df

def smooth_mean_encoding(df, cols, tarcol, alpha = 100):
    global_mean = df[tarcol].mean()
    df_tot = pd.DataFrame()
    for col in cols:
        agg = all_data.groupby(col)[tarcol].agg(['sum','count', 'mean'])
        df_tot[col + '_sum'] = agg['sum']
        df_tot[col + '_cnt'] = agg['count']
        df_tot[col + '_mean'] = agg['mean']
        df_tot = df_tot.reset_index()
        df_combine = pd.merge(df, df_tot, on = col, how = 'left')
        df_combine[col + '_smooth_mean'] = (df[col + '_mean'] * df_combine[col + '_cnt'] + global_mean * alpha) / (df_combine[col + '_cnt'] + alpha)
        df_combine[col + '_smooth_mean'].fillna(global_mean, inplace = True)
    return df