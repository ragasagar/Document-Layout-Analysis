# import some common libraries
from __future__ import annotations
from tkinter import image_names
from tokenize import String
from unicodedata import category
import numpy as np
import os, json, cv2, random, shutil
import torch
import random
from os.path import exists
import pandas as pd
from detectron2.structures import Boxes, pairwise_iou
import json

from tqdm import tqdm

def arrange_data(samples_path,img_path,txt_anno_path):
    try:
        os.mkdir(img_path)
        os.mkdir(txt_anno_path)
    except:
        pass
    
    for f in os.listdir(samples_path):
        source_path = os.path.join(samples_path, f)
        if f.endswith("_ori.jpg"):
            dest_path =  os.path.join(img_path, f)
            shutil.copy(source_path, dest_path)
        elif f.endswith("txt"):
            dest_path =  os.path.join(txt_anno_path, f)
            shutil.copy(source_path, dest_path)

    print("Number of image files: ",len(os.listdir(img_path)))
    print("Number of txt files: ",len(os.listdir(txt_anno_path)))

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



# def split_data(source_dirs, train_dirs, query_dirs, lake_dirs, test_dirs, val_dirs, split_cfg):
#     source_img_dir, source_txt_dir = source_dirs
#     names = [f.replace("_ori.jpg","") for f in os.listdir(source_img_dir)]
#     # print(len(names))
#     np.random.seed(42)

    
#     list_index = list(range(0, len(names)))
#     # random.shuffle(list_index)   
#     # train_indices = list_index[:int(split_cfg['train_ratio']*len(names))]
#     # query_indices = list_index[:int(split_cfg['train_ratio']*len(names))]
#     # val_indices = list_index[int(split_cfg['train_ratio']*len(names)): int(split_cfg['train_ratio']*len(names)+split_cfg['val_ratio']*len(names))]
#     # # print(val_indices)
#     # # rem_indices = Diff(Diff(list_index, val_indices), train_indices)
#     # test_indices = list_index[int(split_cfg['train_ratio']*len(names)+split_cfg['val_ratio']*len(names)): int(split_cfg['train_ratio']*len(names)+split_cfg['val_ratio']*len(names)+split_cfg['test_ratio']*len(names))]
#     # lake_indices = list_index[int(split_cfg['train_ratio']*len(names)+split_cfg['val_ratio']*len(names)+split_cfg['test_ratio']*len(names)):]

#     # # test_indices = rem_indices[0:int(split_cfg['test_ratio']*len(names))]
#     # # lake_indices =  Diff(rem_indices, test_indices)

#     train_indices = list_index[:100]
#     query_indices = list_index[101:200]
#     val_indices = list_index[201:300]
#     lake_indices = list_index[301:1300]
#     test_indices = list_index[1301:1400]
    
#     train_img_dir, train_txt_dir = train_dirs
#     query_img_dir, query_txt_dir = query_dirs
#     lake_img_dir, lake_txt_dir = lake_dirs
#     val_img_dir, val_txt_dir = val_dirs
#     test_img_dir, test_txt_dir = test_dirs

#     remove_dir(train_img_dir)
#     remove_dir(train_txt_dir)
#     remove_dir(lake_img_dir)
#     remove_dir(lake_txt_dir)
#     remove_dir(val_img_dir)
#     remove_dir(val_txt_dir)
#     remove_dir(test_img_dir)
#     remove_dir(test_txt_dir)
#     remove_dir(query_img_dir)
#     remove_dir(query_txt_dir)

#     create_dir(train_img_dir)
#     create_dir(train_txt_dir)
#     create_dir(lake_img_dir)
#     create_dir(lake_txt_dir)
#     create_dir(val_img_dir)
#     create_dir(val_txt_dir)
#     create_dir(test_img_dir)
#     create_dir(test_txt_dir)
#     create_dir(query_img_dir)
#     create_dir(query_txt_dir)
      

#     for index in train_indices:
#         store_file(source_img_dir,source_txt_dir, names, index, train_img_dir, train_txt_dir)
        
#     for index in query_indices:
#         store_file(source_img_dir,source_txt_dir, names, index, query_img_dir, query_txt_dir)
#     for index in lake_indices:
#         store_file(source_img_dir,source_txt_dir, names, index, lake_img_dir, lake_txt_dir)
#     for index in val_indices:
#         store_file(source_img_dir,source_txt_dir, names, index, val_img_dir, val_txt_dir)
#     for index in test_indices:
#         store_file(source_img_dir,source_txt_dir, names, index, test_img_dir, test_txt_dir)



def split_data(source_dirs, train_dirs, lake_dirs, test_dirs, val_dirs, split_cfg, data_size = 5000):
    source_img_dir, source_txt_dir = source_dirs
    names = [f.replace(".xml","") for f in os.listdir(source_txt_dir)]
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

    remove_dir(train_img_dir)
    remove_dir(train_txt_dir)
    remove_dir(lake_img_dir)
    remove_dir(lake_txt_dir)
    remove_dir(val_img_dir)
    remove_dir(val_txt_dir)
    remove_dir(test_img_dir)
    remove_dir(test_txt_dir)


    create_dir(train_img_dir)
    create_dir(train_txt_dir)
    create_dir(lake_img_dir)
    create_dir(lake_txt_dir)
    create_dir(val_img_dir)
    create_dir(val_txt_dir)
    create_dir(test_img_dir)
    create_dir(test_txt_dir)
    
    for index in train_indices:
        if exists(os.path.join(source_txt_dir, "{}.xml".format(names[index]))):
            store_file(source_img_dir,source_txt_dir, names, index, train_img_dir, train_txt_dir)
    for index_l in lake_indices:
        if exists(os.path.join(source_txt_dir, "{}.xml".format(names[index_l]))):
            store_file(source_img_dir,source_txt_dir, names, index_l, lake_img_dir, lake_txt_dir)
    for index_v in val_indices:
        if exists(os.path.join(source_txt_dir, "{}.xml".format(names[index_v]))):
            store_file(source_img_dir,source_txt_dir, names, index_v, val_img_dir, val_txt_dir)
    for index_t in test_indices:
        if exists(os.path.join(source_txt_dir, "{}.xml".format(names[index_t]))):
            store_file(source_img_dir,source_txt_dir, names, index_t, test_img_dir, test_txt_dir)

    split_dataset(train_img_dir, "PASCAL_VOC/PASCAL_VOC/pascal_train2007.json" ,"PASCAL_VOC/PASCAL_VOC/train_targeted.json")
    split_dataset(lake_img_dir, "PASCAL_VOC/PASCAL_VOC/pascal_train2007.json" ,"PASCAL_VOC/PASCAL_VOC/lake_targeted.json")
    split_dataset(val_img_dir, "PASCAL_VOC/PASCAL_VOC/pascal_train2007.json" ,"PASCAL_VOC/PASCAL_VOC/val_targeted.json")

def coco_bbox_to_coordinates(bbox):
    out = bbox.copy().astype(float)
    out[:, 2] = bbox[:, 0] + bbox[:, 2]
    out[:, 3] = bbox[:, 1] + bbox[:, 3]
    return out


def conf_matrix_calc(labels, detections, n_classes, conf_thresh, iou_thresh):
    confusion_matrix = np.zeros([n_classes + 1, n_classes + 1])
    l_classes = np.array(labels)[:, 0].astype(int)
    l_bboxs = coco_bbox_to_coordinates((np.array(labels)[:, 1:]))
    d_confs = np.array(detections)[:, 4]
    d_bboxs = (np.array(detections)[:, :4])
    d_classes = np.array(detections)[:, -1].astype(int)
    detections = detections[np.where(d_confs > conf_thresh)]
    labels_detected = np.zeros(len(labels))
    detections_matched = np.zeros(len(detections))
    for l_idx, (l_class, l_bbox) in enumerate(zip(l_classes, l_bboxs)):
        for d_idx, (d_bbox, d_class) in enumerate(zip(d_bboxs, d_classes)):
            iou = pairwise_iou(Boxes(torch.from_numpy(np.array([l_bbox]))), Boxes(torch.from_numpy(np.array([d_bbox]))))
            if iou >= iou_thresh:
                confusion_matrix[l_class, d_class] += 1
                labels_detected[l_idx] = 1
                detections_matched[d_idx] = 1
    for i in np.where(labels_detected == 0)[0]:
        confusion_matrix[l_classes[i], -1] += 1
    for i in np.where(detections_matched == 0)[0]:
        confusion_matrix[-1, d_classes[i]] += 1
    return confusion_matrix

def create_query_dataset(categories):
    category_list = {
            "aeroplane":1, "bicycle":2, "bird":3, "boat":4, "bottle":5, "bus":6, "car":7, "cat":8,
    "chair":9, "cow":10, "diningtable":11, "dog":12, "horse":13, "motorbike":14, "person":15,
    "pottedplant":16, "sheep":17, "sofa":18, "train":19, "tvmonitor":20}

    for category in tqdm(categories):
        category_id = category_list[category];
        query_data_dirs = ("voctest_06-nov-2007/VOCdevkit/VOC2007/JPEGImages", "pascal_voc/PASCAL_VOC/pascal_test2007.json")
        final_query_data_dirs = ("query_data_img/"+category, "query_data_txt_anno/"+category)

        with open("pascal_voc/PASCAL_VOC/pascal_test2007.json") as f:
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

def find_missclassified_object(result):
    ignore_class = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl']
    minAP = 100
    category=""
    for k, val in result.items():
        for key, value in val.items():
            if(key in ignore_class):
                continue
            else:
                if(minAP> value):
                    category = key;
                    minAP = value;
    category = category.lower().replace('ap-',"");
    return ('query_data_img/'+ category,[category]) 


def get_test_score(result):
    for k, val in result.items():
        for key, value in val.items():
            if(key == "AP"):
                return value;


def split_dataset(img_data_dir, img_anno_file, dest_anno_file):
    with open(img_anno_file, mode="r") as f:
        dataset = json.load(f)

    data_images = dataset['images']
    list_images = os.listdir(img_data_dir)
    random.shuffle(list_images)
    list_images = list_images[:10000];
    images = list(filter(lambda x: x['file_name'] in list_images, data_images))
    print("filter completed")
    annotations = dataset['annotations']
    categories = dataset["categories"]

    image_ids = [id['id'] for id in images]

    create_labels(image_ids, images, annotations, categories, dest_anno_file)
    # print("train_label_created")

def aug_train_subset(subset_result, train_data_json, lake_data_json, budget):
    with open(lake_data_json, mode="r") as f:
        lake_dataset = json.load(f)
    with open(train_data_json, mode="r") as f:
        train_dataset = json.load(f)

    categories = lake_dataset['categories']
    image_list = list(filter(lambda x: x['file_name'] in subset_result, lake_dataset['images']))
    image_id = [image['id'] for image in image_list]
    annotations_shift = list(filter(lambda x: x['image_id'] in image_id, lake_dataset['annotations']))

    train_annotations = train_dataset['annotations'];
    train_image_list = train_dataset['images'];

    # appending the images to train images
    train_image_list += image_list;
    train_annotations += annotations_shift;

    #removing the images lake dataset.
    final_lake_image_list = list(filter(lambda x: x['file_name'] not in subset_result, lake_dataset['images']))
    image_id = [image['id'] for image in image_list]
    final_lake_annotations = list(filter(lambda x: x['image_id'] in image_id, lake_dataset['annotations']))

    create_labels(train_image_list, train_annotations, categories, train_data_json)
    create_labels(final_lake_image_list, final_lake_annotations, categories, lake_data_json)


def create_labels(indices, images, annotations, categories, filename):
    labels = {}
    image_list = list(filter(lambda x: x['id'] in indices, images))
    annotation_list = list(filter(lambda x: x['image_id'] in indices, annotations))
    labels['images'] = image_list
    labels['annotations'] = annotation_list
    labels['categories'] = categories

    with open(filename, "w") as f:
        json.dump(labels, f)

# create_query_dataset([
#     "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
#     "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
#     "pottedplant", "sheep", "sofa", "train", "tvmonitor"
# ])

