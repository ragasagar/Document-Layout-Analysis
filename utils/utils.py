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
    source_img_path = os.path.join(source_img_dir, "{}_ori.jpg".format(names[index]))
    source_txt_path = os.path.join(source_txt_dir, "{}.txt".format(names[index]))
    dest_img_path = os.path.join(img_dir, "{}_ori.jpg".format(names[index]))
    dest_txt_path = os.path.join(txt_dir, "{}.txt".format(names[index]))
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



def split_data(source_dirs, train_dirs, query_dirs, lake_dirs, test_dirs, val_dirs, split_cfg):
    source_img_dir, source_txt_dir = source_dirs
    names = [f.replace(".txt","") for f in os.listdir(source_txt_dir)]
    print(len(names))
    np.random.seed(42)

    
    list_index = list(range(0, len(names)))
    random.shuffle(list_index)
    train_indices = list_index[:1000]
    # query_indices = list_index[:int(split_cfg['train_ratio']*len(names))]
    val_indices = list_index[1001:1500]

    # # print(val_indices)
    # # rem_indices = Diff(Diff(list_index, val_indices), train_indices)
    test_indices = list_index[1501:2000]
    lake_indices = list_index[2001:4000]

    # # test_indices = rem_indices[0:int(split_cfg['test_ratio']*len(names))]
    # # lake_indices =  Diff(rem_indices, test_indices)

    
    train_img_dir, train_txt_dir = train_dirs
    query_img_dir, query_txt_dir = query_dirs
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
    remove_dir(query_img_dir)
    remove_dir(query_txt_dir)

    create_dir(train_img_dir)
    create_dir(train_txt_dir)
    create_dir(lake_img_dir)
    create_dir(lake_txt_dir)
    create_dir(val_img_dir)
    create_dir(val_txt_dir)
    create_dir(test_img_dir)
    create_dir(test_txt_dir)
    create_dir(query_img_dir)
    create_dir(query_txt_dir)
    
    for index in train_indices:
        if exists(os.path.join(source_txt_dir, "{}.txt".format(names[index]))):
            store_file(source_img_dir,source_txt_dir, names, index, train_img_dir, train_txt_dir)
    for index_l in lake_indices:
        if exists(os.path.join(source_txt_dir, "{}.txt".format(names[index]))):
            store_file(source_img_dir,source_txt_dir, names, index_l, lake_img_dir, lake_txt_dir)
    for index_v in val_indices:
        if exists(os.path.join(source_txt_dir, "{}.txt".format(names[index]))):
            store_file(source_img_dir,source_txt_dir, names, index_v, val_img_dir, val_txt_dir)
    for index_t in test_indices:
        if exists(os.path.join(source_txt_dir, "{}.txt".format(names[index_t]))):
            store_file(source_img_dir,source_txt_dir, names, index_t, test_img_dir, test_txt_dir)

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
            'abstract': 0, 'author': 1, 'caption': 2, 'equation': 3, 'figure': 4, 'footer': 5, 
             'list': 6, 'paragraph': 7, 'reference': 8, 'section': 9, 'table': 10, 'title': 11, "date": 12}
    for category in tqdm(categories):
        category_id = category_list[category];
        query_data_dirs = ("DocBank_500K_ori_img", "DocBank_500K_txt")
        final_query_data_dirs = ("query_data_img/"+category, "query_data_txt_anno/"+category)

        with open("COCOQueryData.json") as f:
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
            image_names = [names.split("/")[-1].replace("_ori.jpg", "") for names in image_names]
            names = list(set(image_names));
            if len(image_names) > 5 :
                names = image_names[:5];

            for index in range(len(names)):
                # Source path
                source_img = os.path.join(query_data_dirs[0], "{}_ori.jpg".format(names[index]))
                print(source_img)
                # Destination path
                source_txt = os.path.join(query_data_dirs[1], "{}.txt".format(names[index]))

                destination_img = os.path.join(final_query_data_dirs[0], "{}_ori.jpg".format(names[index]))
                # Destination path
                destination_txt = os.path.join(final_query_data_dirs[1], "{}.txt".format(names[index]))
                
                if not os.path.exists(query_data_dirs[0]) or not os.path.exists(query_data_dirs[1]):
                    os.mkdir(query_data_dirs[0])
                    os.mkdir(query_data_dirs[1])
                
                try:
                    shutil.copy(source_img, destination_img)
                    shutil.copy(source_txt, destination_txt)
                    # print("File copied successfully.")
                except shutil.SameFileError:
                    print("Source and destination represents the same file.")
                
                # If there is any permission issue
                except PermissionError:
                    print("Permission denied.")
                
                # For other errors
                except:
                    print("Error occurred while copying file.")


                # # removing the data from the lake data
                # try:
                #     os.remove(os.path.join(query_data_dirs[0], "{}_ori.jpg".format(names[index])))
                #     os.remove(os.path.join(query_data_dirs[1], "{}.txt".format(names[index])))
                # except:
                #     pass
                    
                

            # query_im = [names.split("/")[-1].replace("_ori.jpg", "") for names in query_im]
            # print(query_im[0:5])
            # query_im = [i["file_name"].split("/")[-1] for i in query_im]
            # query_im = [names.split("/")[-1].replace("_ori.jpg", "") for names in query_im]
            # print(query_im[0:5])

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