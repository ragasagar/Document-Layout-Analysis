import os
import torch
from tqdm import tqdm
from detectron2.structures import Boxes, pairwise_iou


import numpy as np
import os, json, random, shutil
import torch
import random
import json

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


    
def aug_train_subset(subset_result, train_data_json, lake_data_json, budget, src_dir, dest_dir):
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
    final_lake_annotations = list(filter(lambda x: x['image_id'] not in image_id, lake_dataset['annotations']))

    #moving data from lake set to train set.
    change_dir(subset_result, src_dir, dest_dir)

    #changing the coco-file for annotations
    create_labels_update(train_image_list, train_annotations, categories, train_data_json)
    create_labels_update(final_lake_image_list, final_lake_annotations, categories, lake_data_json)

def create_labels_update(images, annotations, categories, filename):
    labels = {}
    labels['images'] = images
    labels['annotations'] = annotations
    labels['categories'] = categories

    with open(filename, "w") as f:
        json.dump(labels, f)

def change_dir(image_results, src_dir, dest_dir):
    names = [names.split("/")[-1].replace(".jpg", "") for names in image_results]
    for index in range(len(names)):
        # Source path
        source_img = os.path.join(src_dir[0], "{}.jpg".format(names[index]))
        # Destination path
        # source_txt = os.path.join(src_dir[1], "{}.xml".format(names[index]))

        destination_img = os.path.join(dest_dir[0], "{}.jpg".format(names[index]))
        # Destination path
        # destination_txt = os.path.join(dest_dir[1], "{}.xml".format(names[index]))
        
        if not os.path.exists(dest_dir[0]) or not os.path.exists(dest_dir[1]):
            os.mkdir(dest_dir[0])
            os.mkdir(dest_dir[1])
        
        try:
            shutil.copy(source_img, destination_img)
            # shutil.copy(source_txt, destination_txt)
            # print("File copied successfully.")
        except shutil.SameFileError:
            print("Source and destination represents the same file.")
        
        # If there is any permission issue
        except PermissionError:
            print("Permission denied.")
        
        # For other errors
        except Exception as e:
            print("Error occurred while copying file.", e)


        # removing the data from the lake data
        try:
            os.remove(os.path.join(src_dir[0], "{}.jpg".format(names[index])))
            # os.remove(os.path.join(src_dir[1], "{}.xml".format(names[index])))
        except:
            pass

def change_dir_sanskrit(image_results, src_dir, dest_dir):
    names = [names.split("/")[-1].replace(".jpeg", "") for names in image_results]
    for index in range(len(names)):
        # Source path
        source_img = os.path.join(src_dir[0], "{}.jpeg".format(names[index]))
        # Destination path
        # source_txt = os.path.join(src_dir[1], "{}.xml".format(names[index]))

        destination_img = os.path.join(dest_dir[0], "{}.jpeg".format(names[index]))
        # Destination path
        # destination_txt = os.path.join(dest_dir[1], "{}.xml".format(names[index]))
        
        if not os.path.exists(dest_dir[0]) or not os.path.exists(dest_dir[1]):
            os.mkdir(dest_dir[0])
            os.mkdir(dest_dir[1])
        
        try:
            shutil.copy(source_img, destination_img)
            # shutil.copy(source_txt, destination_txt)
            # print("File copied successfully.")
        except shutil.SameFileError:
            print("Source and destination represents the same file.")
        
        # If there is any permission issue
        except PermissionError:
            print("Permission denied.")
        
        # For other errors
        except Exception as e:
            print("Error occurred while copying file.", e)


        # removing the data from the lake data
        try:
            os.remove(os.path.join(src_dir[0], "{}.jpeg".format(names[index])))
            # os.remove(os.path.join(src_dir[1], "{}.xml".format(names[index])))
        except:
            pass

def create_labels(indices, images, annotations, categories, filename):
    labels = {}
    image_list = list(filter(lambda x: x['id'] in indices, images))
    annotation_list = list(filter(lambda x: x['image_id'] in indices, annotations))
    labels['images'] = image_list
    labels['annotations'] = annotation_list
    labels['categories'] = categories

    with open(filename, "w") as f:
        json.dump(labels, f)

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

def copy_trained_image(train_data_dirs:tuple):
     #removing the existing trained images and putting the original trained images.
    remove_dir(train_data_dirs[0])
    create_dir(train_data_dirs[0])
    remove_dir(train_data_dirs[1])
    try:
        shutil.copytree("../publaynet/intial_train_img/train_data_img", train_data_dirs[0])
        shutil.copy("../publaynet/intial_train_img/train_targeted.json", train_data_dirs[1])
    except Exception as e:
        logger.error ("Error while copying the original trained images:", e)

def create_query_dataset(categories, path):
    # category_list = {
    #     "text":1, "title":2, "list":3,"table":4, "figure":5
    # }
    category_list = {"bg":0, "Image":1, "Math":2, "Table":3, "Text":4}

    with open(path) as f:
        data = json.load(f);
        query_im = data['images']
        ids = [x['id'] for x in query_im if x['file_name'] in os.listdir("sanskrit_tada/train_data_img")];
        annotations = data['annotations']
        for category in tqdm(categories):
            category_id = category_list[category];
            query_data_dirs = ("sanskrit_tada/train_data_img", "sanskrit_tada/train_targeted.json")
            final_query_data_dirs = ("query_data_img/"+category, "query_data_txt_anno/"+category)
            # print(annotations[:5])
            images = set();
            for annotation in annotations:
                if(annotation['category_id'] == category_id and annotation['image_id'] in ids):
                    images.add(annotation['image_id']) 
                    if(len(images)==50):
                        break
                
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
                except Exception as e:
                    print("Error occurred while copying file.", e)

def store_file(source_img_dir,name,img_dir):
    source_img_path = os.path.join(source_img_dir, name)
    # source_txt_path = os.path.join(source_txt_dir, "{}.xml".format(names[index]))
    dest_img_path = os.path.join(img_dir, name)
    # dest_txt_path = os.path.join(txt_dir, "{}.xml".format(names[index]))
    try:
        # shutil.copy(source_txt_path, dest_txt_path)
        shutil.copy(source_img_path, dest_img_path)
    except:
        pass

def get_lakeset_images(dataset_dir, lake_data_dirs, size, ratio):
    category_list = {
        "text":1, "title":2, "list":3,"table":4, "figure":5
    }
    
    with open(dataset_dir[1]) as f:
        data = json.load(f)
    
    lake_images = set(os.listdir(dataset_dir[0]))
    json_images_data = set([x['file_name'] for x in data['images']])
    lake_images = lake_images.intersection(json_images_data)

    total_images = []
    total_annotations = []
    lake_imgs = []
    for key, value in category_list.items():
        size_value = int(size * ratio[key])
        lake_data_img, lake_img_annotatons, images = get_images_for_category(data, lake_images, value, size_value)
        total_images+=images
        total_annotations+=lake_img_annotatons
        lake_images = lake_images - set(lake_data_img)
        lake_imgs+=lake_data_img
    categories = data['categories']

    #copying files to lake_folder
    remove_dir(lake_data_dirs[0])
    create_dir(lake_data_dirs[0])

    create_labels_update(total_images, total_annotations, categories, lake_data_dirs[1])

    for images in lake_imgs:
        store_file(dataset_dir[0], images, lake_data_dirs[0])


def get_images_for_category(data, lake_images,category, size):
    image_ids = [x['id'] for x in data['images'] if x['file_name'] in lake_images]

    #fitlering the images on the basis of category id
    category_annotations = [ x for x in data['annotations'] if x['category_id']==category]
    category_image_ids = [x['image_id'] for x in category_annotations]
    category_image_ids = list(set(image_ids).intersection(set(category_image_ids)))[:size]

    #getting image_names for the respective images_id
    category_images = [x for x in data['images'] if x['id'] in category_image_ids]
    category_image_names = [x['file_name'] for x in category_images]
    
    #filtering the images exist in the current train folder
    images = lake_images.intersection(set(category_image_names))

    category_annotations = list(filter(lambda x: x['image_id'] in category_image_ids, category_annotations))
    
    return images, category_annotations, category_images

def create_new_query(train_data_dirs, query_data_dir, category):
    with open(train_data_dirs[1], "r") as f:
        data = json.load(f);
    category_list = {
        "text":1, "title":2, "list":3,"table":4, "figure":5
    }
    # category_list = {
    #         "aeroplane":1, "bicycle":2, "bird":3, "boat":4, "bottle":5, "bus":6, "car":7, "cat":8,
    # "chair":9, "cow":10, "diningtable":11, "dog":12, "horse":13, "motorbike":14, "person":15,
    # "pottedplant":16, "sheep":17, "sofa":18, "train":19, "tvmonitor":20}

    annotation = data['annotations']
    # final_lake_annotations = list(filter(lambda x: x['image_id'] in image_id, lake_dataset['annotations']))
    annotation = list(filter(lambda x: x['category_id'] == category_list[category[0]], annotation))
    images_id = list(set([x['image_id'] for x in annotation]))

    images = filter(lambda x: x['id'] in images_id, data['images'])
    images = [x['file_name'] for x in images];

    random.seed(42);
    random.shuffle(images);

    remove_dir(query_data_dir);
    
    create_dir(query_data_dir);

    for img in tqdm(images[:10]):
        source_img = os.path.join(train_data_dirs[0], "{}".format(img))
        # Destination path
        # source_txt = os.path.join(src_dir[1], "{}.xml".format(names[index]))

        destination_img = os.path.join(query_data_dir, "{}".format(img));
        try:
            shutil.copy(source_img, destination_img)
            # shutil.copy(source_txt, destination_txt)
            # print("File copied successfully.")
        except shutil.SameFileError:
            print("Source and destination represents the same file.")
        
        # If there is any permission issue
        except PermissionError:
            print("Permission denied.")
        
        # For other errors
        except:
            print("Error occurred while copying file.")

def get_original_images_path(subset_result:list):
    return ["_".join(x.split("/")[-1].split("_")[:2])+ ".jpg" for x in subset_result]

def get_original_images_path_sanskrit(subset_result:list):
    return ["_".join(x.split("/")[-1].split("_")[:2]) for x in subset_result]


def get_category_details(subset_result, train_data_dirs, category):
    with open(train_data_dirs[1], "r") as f:
        data = json.load(f);
    category_list = {
        "text":1, "title":2, "list":3,"table":4, "figure":5
    }
    # category_list = {"bg":0, "Image":1, "Math":2, "Table":3, "Text":4}
    images = [x['id'] for x in  data['images'] if x['file_name'] in subset_result]
    annotation = data['annotations']
    # final_lake_annotations = list(filter(lambda x: x['image_id'] in image_id, lake_dataset['annotations']))
    annotation = list(filter(lambda x: x['category_id'] == category_list[category[0]] and x['image_id'] in images, annotation))
    return len(annotation)

def conf_matrix_calc(labels, detections, n_classes, conf_thresh, iou_thresh):
    confusion_matrix = np.zeros([n_classes, n_classes])
    l_classes = np.array(labels)[:, 0].astype(int)
    l_bboxs = coco_bbox_to_coordinates((np.array(labels)[:, 1:]))
    d_confs = np.array(detections)[:, 4]
    d_bboxs = (np.array(detections)[:, :4])
    d_classes = np.array(detections)[:, -1].astype(int)
    detections = detections[np.where(d_confs > conf_thresh)]
    labels_detected = np.zeros(len(labels))
    detections_matched = np.zeros(len(labels))
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

def coco_bbox_to_coordinates(bbox):
    out = bbox.copy().astype(float)
    out[:, 2] = bbox[:, 0] + bbox[:, 2]
    out[:, 3] = bbox[:, 1] + bbox[:, 3]
    return out