# import some common libraries
from __future__ import annotations
from tokenize import String
from unicodedata import category
from matplotlib import image
import numpy as np
import os, json, cv2, random, shutil
import torch
import random
from os.path import exists
import pandas as pd
from detectron2.structures import Boxes, pairwise_iou
import json

from tqdm import tqdm


def remove_dir(dir_name):
    try:
        shutil.rmtree(dir_name)
    except:
        pass


def create_dir(dir_name):
    try:
        os.mkdir(dir_name)
    except:
        pass

def store_file(source_img_dir,source_txt_dir,names, index, img_dir, txt_dir):
    source_img_path = os.path.join(source_img_dir, "{}.jpg".format(names[index]))
    source_txt_path = os.path.join(source_txt_dir, "{}.xml".format(names[index]))
    dest_img_path = os.path.join(img_dir, "{}.jpg".format(names[index]))
    dest_txt_path = os.path.join(txt_dir, "{}.xml".format(names[index]))
    try:
        shutil.copy(source_txt_path, dest_txt_path)
        shutil.copy(source_img_path, dest_img_path)
    except:
        pass


def split_data(source_dirs, split_cfg, source_json, data_size = 4952):
    # source_dirs = ("img", "txt")
    #  = ("pascal2007/new_query_data_img", "pascal2007/new_query_data_txt_anno")
    train_dirs = ("pascal2007/train_data_img", "pascal2007/train_data_txt_anno")
    lake_dirs = ("pascal2007/lake_data_img", "pascal2007/lake_data_txt_anno")
    test_dirs = ("pascal2007/test_data_img", "pascal2007/test_data_txt_anno")
    val_dirs = ("pascal2007/val_data_img", "pascal2007/val_data_txt_anno")

    source_img_dir, source_txt_dir = source_dirs
    with open(source_json, mode="r") as f:
        dataset = json.load(f)
    
    data_images = [x['file_name'] for x in dataset['images']]
    images = list(filter(lambda x: x in data_images, os.listdir(source_img_dir)))
    names = [f.replace(".jpg","") for f in images]
    print(len(names))
    np.random.seed(42)

    
    list_index = list(range(0, len(names)))
    random.shuffle(list_index)
    train_indices = list_index[:int(data_size * split_cfg['train_ratio'])]
    val_index = int((split_cfg['train_ratio']  + split_cfg['val_ratio']) * data_size);
    val_indices = list_index[int(data_size * split_cfg['train_ratio'] + 1) : val_index]

    test_index = int(val_index + split_cfg['test_ratio'] * data_size);
    test_indices = list_index[val_index + 1: test_index];
    
    lake_indices = list_index[test_index +1:data_size];
 
    train_img_dir, train_txt_dir = train_dirs
    
    lake_img_dir, lake_txt_dir = lake_dirs
    val_img_dir, val_txt_dir = val_dirs
    test_img_dir, test_txt_dir = test_dirs

    if(split_cfg['train_ratio']>0):
        remove_dir(train_img_dir)
        remove_dir(train_txt_dir)
    if(len(lake_indices)>0):
        remove_dir(lake_img_dir)
        remove_dir(lake_txt_dir)
    if(split_cfg['val_ratio']>0):    
        remove_dir(val_img_dir)
        remove_dir(val_txt_dir)
    if(split_cfg['test_ratio']>0):
        remove_dir(test_img_dir)
        remove_dir(test_txt_dir)


    if(split_cfg['train_ratio']>0):
        create_dir(train_img_dir)
        create_dir(train_txt_dir)
        for index in train_indices:
            if exists(os.path.join(source_txt_dir, "{}.xml".format(names[index]))):
                store_file(source_img_dir,source_txt_dir, names, index, train_img_dir, train_txt_dir)
        split_dataset(train_img_dir, source_json ,"pascal2007/train_targeted.json")

    if(len(lake_indices)>0):
        create_dir(lake_img_dir)
        create_dir(lake_txt_dir)
        for index_l in lake_indices:
            if exists(os.path.join(source_txt_dir, "{}.xml".format(names[index_l]))):
                store_file(source_img_dir,source_txt_dir, names, index_l, lake_img_dir, lake_txt_dir)
        split_dataset(lake_img_dir, source_json ,"pascal2007/lake_targeted.json")
            
    if(split_cfg['val_ratio']>0):    
        create_dir(val_img_dir)
        create_dir(val_txt_dir)
        for index_v in val_indices:
            if exists(os.path.join(source_txt_dir, "{}.xml".format(names[index_v]))):
                store_file(source_img_dir,source_txt_dir, names, index_v, val_img_dir, val_txt_dir)
        split_dataset(val_img_dir, source_json ,"pascal2007/val_targeted.json")

    if(split_cfg['test_ratio']>0):
        create_dir(test_img_dir)
        create_dir(test_txt_dir)
        for index_t in test_indices:
            if exists(os.path.join(source_txt_dir, "{}.xml".format(names[index_t]))):
                store_file(source_img_dir,source_txt_dir, names, index_t, test_img_dir, test_txt_dir)
        split_dataset(test_img_dir, source_json ,"pascal2007/test_targeted.json")


def split_dataset(img_data_dir, img_anno_file, dest_anno_file):
    with open(img_anno_file, mode="r") as f:
        dataset = json.load(f)

    data_images = dataset['images']
    list_images = os.listdir(img_data_dir)
    random.shuffle(list_images)
    images = list(filter(lambda x: x['file_name'] in list_images, data_images))
    print("filter completed")
    annotations = dataset['annotations']
    categories = dataset["categories"]

    image_ids = [id['id'] for id in images]

    create_labels(image_ids, images, annotations, categories, dest_anno_file)

def create_labels(indices, images, annotations, categories, filename):
    labels = {}
    image_list = list(filter(lambda x: x['id'] in indices, images))
    annotation_list = list(filter(lambda x: x['image_id'] in indices, annotations))
    labels['images'] = image_list
    labels['annotations'] = annotation_list
    labels['categories'] = categories

    with open(filename, "w") as f:
        json.dump(labels, f)

def create_labels_update(images, annotations, categories, filename):
    labels = {}
    labels['images'] = images
    labels['annotations'] = annotations
    labels['categories'] = categories

    with open(filename, "w") as f:
        json.dump(labels, f)

def create_pascal_db():
    split_cfg = {
                "train_ratio":0.3,
                "val_ratio":.15,
                "test_ratio":.15
    }
    source_json='pascal2007/pascal_voc/PASCAL_VOC/pascal_test2007.json'
    source_dirs = ("pascal2007/voctest_06-nov-2007/VOCdevkit/VOC2007/JPEGImages", "pascal2007/voctest_06-nov-2007/VOCdevkit/VOC2007/Annotations")
    split_data(source_dirs, split_cfg, source_json,4952)

def create_query_dataset(categories):
    category_list = {
            "aeroplane":1, "bicycle":2, "bird":3, "boat":4, "bottle":5, "bus":6, "car":7, "cat":8,
    "chair":9, "cow":10, "diningtable":11, "dog":12, "horse":13, "motorbike":14, "person":15,
    "pottedplant":16, "sheep":17, "sofa":18, "train":19, "tvmonitor":20}

    for category in tqdm(categories):
        category_id = category_list[category];
        query_data_dirs = ("pascal2007/train_data_img", "pascal2007/train_targeted.json")
        final_query_data_dirs = ("query_data_img/"+category, "query_data_txt_anno/"+category)

        with open("pascal2007/train_targeted.json") as f:
            data = json.load(f);
            query_im = data['images']
            annotations = data['annotations']
            # print(annotations[:5])
            images = [];
            for annotation in annotations:
                if(annotation['category_id'] == category_id):
                    images.append(annotation['image_id'])
                
            image_names = [];
            for image in  query_im:
                if image['id'] in images:
                    image_names.append(image['file_name'])
            print(image_names[:1])
            
            image_names = [i.split("/")[-1] for i in image_names]
            image_names = [names.split("/")[-1].replace(".jpg", "") for names in image_names]
            names = list(set(image_names));
            if len(image_names) > 5 :
                names = image_names[:5];

            for index in range(len(names)):
                # Source path
                source_img = os.path.join(query_data_dirs[0], "{}.jpg".format(names[index]))
                # Destination path
                # source_txt = os.path.join(query_data_dirs[1], "{}.txt".format(names[index]))

                destination_img = os.path.join(final_query_data_dirs[0], "{}.jpg".format(names[index]))
                # Destination path
                # destination_txt = os.path.join(final_query_data_dirs[1], "{}.txt".format(names[index]))
                
                if not os.path.exists(final_query_data_dirs[0]):
                    os.mkdir(final_query_data_dirs[0])
                
                try:
                    shutil.copy(source_img, destination_img)
                    # print("File copied successfully.")
                except shutil.SameFileError:
                    print("Source and destination represents the same file.")
                
                # If there is any permission issue
                except PermissionError:
                    print("Permission denied.")
                
                # For other errors
                except:
                    print("Error occurred while copying file.")

# create_query_dataset([
#             "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
#             "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
#             "pottedplant", "sheep", "sofa", "train", "tvmonitor"])