import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image
import tensorflow_datasets as tfds


def get_colors(num_classes):
    base = (1.0 / num_classes)
    return tf.convert_to_tensor([[base * i, base * i, base * i] for i in range(num_classes)], tf.float32)

def load_images(image):
    with open(image, 'rb') as z:
        return np.array(Image.open(z).convert('P'))#.flatten()


def load_npz(ds_path):
    label_files =  sorted([os.path.join(dp, f) for dp, dn, fn in \
                    os.walk(os.path.expanduser(ds_path), followlinks=True) for f in \
                    fn if f.endswith("labels.npy.npz")])

    data_files =  sorted([os.path.join(dp, f) for dp, dn, fn in \
                    os.walk(os.path.expanduser(ds_path), followlinks=True) for f in \
                    fn if f.endswith("inputs.npy.npz")])

    with np.load(*data_files) as d:
        data = d['arr_0']
        coeffs = d['arr_1']

    label = np.load(*label_files)['arr_0']


    data = np.expand_dims(data.astype(np.float32,copy=False)/255.0, axis=3)            # normalize and add channel dimension
    label = np.expand_dims(label.astype(np.int32, copy=False), axis=3)

    with open(os.path.join(ds_path, 'metadata.json')) as json_file:
        metadata = json.load(json_file)

    nfeet = 14
    ns = 20000
    train_size_f = int(ns*0.7)
    val_size_f = int(ns*0.15)

    t_indx = np.arange(train_size_f)
    for i in range(1,14):
        t_indx = np.append(t_indx, np.arange(train_size_f)+i*ns)

    v_indx = np.arange(train_size_f, train_size_f+val_size_f)
    for i in range(1,14):
        v_indx = np.append(v_indx, np.arange(train_size_f, train_size_f+val_size_f)+i*ns)

    te_indx = np.arange(train_size_f, train_size_f+2*val_size_f)
    for i in range(1,14):
        te_indx = np.append(te_indx, np.arange(train_size_f, train_size_f+2*val_size_f)+i*ns)

    
    ds_size = np.shape(data)[0]

    train_size = int(ds_size* 0.7)
    val_size = int(ds_size * 0.15)
  
    train_ds = tf.data.Dataset.from_tensor_slices((np.take(data,t_indx,axis=0), coeffs[t_indx,0], coeffs[t_indx,1], np.take(label,t_indx,axis=0))).cache().shuffle(train_size)

    val_ds = tf.data.Dataset.from_tensor_slices((np.take(data,v_indx,axis=0), coeffs[v_indx,0], coeffs[v_indx,1], np.take(label,v_indx,axis=0)))

    test_ds = tf.data.Dataset.from_tensor_slices((np.take(data,te_indx,axis=0), coeffs[te_indx,0], coeffs[te_indx,1], np.take(label,te_indx,axis=0)))

    return (train_ds, train_size), (val_ds, val_size), (test_ds, ds_size - train_size - val_size ), metadata



def _cast(data, target):
    image = tf.cast(data, tf.float32)
    labels = tf.cast(target, tf.int32)
    return image,labels

def _convert_img(data,target):
    image = tf.image.convert_image_dtype(data, tf.float32)
    labels = tf.cast(target, tf.int32)
    return image, labels


def get_conditional_files(base_path, condition):
    return sorted([os.path.join(dp, f) for dp, dn, fn in
                   os.walk(os.path.expanduser(base_path), followlinks=True) for f in
                   fn if condition(f)])


def unison_shuffled_copies(*arrays):
    p = np.random.permutation(len(arrays[0]))
    return (a[p] for a in arrays)
