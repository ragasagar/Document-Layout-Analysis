import json
import random
import os
from numpy import append
from torch import rand
import tqdm

def split_dataset():
    train_targeted = "train_targeted.json"
    lake_targeted = "lake_targeted.json"
    val_targeted = "val_targeted.json"
    test_targeted = "test_targeted.json"
    with open("/home/sysadminkcdh/subset_selection/publaynet/val.json", mode="r") as f:
        dataset = json.load(f)

    print(dataset["images"][0])
    print(dataset["annotations"][0])
    print(dataset["categories"][2])

    data_images = dataset['images']
    list_images = os.listdir("/home/sysadminkcdh/subset_selection/publaynet/val")
    print(len(list_images));
    random.shuffle(list_images)
    list_images = list_images[:10000];
    images = list(filter(lambda x: x['file_name'] in list_images, data_images))
    print("filter completed")
    annotations = dataset['annotations']
    categories = dataset["categories"]

    image_ids = [id['id'] for id in images]
    
    random.seed(42);
    random.shuffle(image_ids)

    train_indices = image_ids
    # lake_indices = image_ids[1001:5000]
    # val_indices = image_ids[5001:5500]
    # test_indices = image_ids[5501:6000]

    create_labels(train_indices, images, annotations, categories, train_targeted)
    print("train_label_created")
    # create_labels(lake_indices, images, annotations, categories, lake_targeted)
    # print("lake_label_created")
    # create_labels(test_indices, images, annotations, categories, test_targeted)
    # print("test_label_created")
    # create_labels(val_indices, images, annotations, categories, val_targeted)
    # print("val_label_created")



def create_labels(indices, images, annotations, categories, filename):
    labels = {}
    image_list = list(filter(lambda x: x['id'] in indices, images))
    annotation_list = list(filter(lambda x: x['image_id'] in indices, annotations))
    labels['images'] = image_list
    labels['annotations'] = annotation_list
    labels['categories'] = categories

    with open(filename, "w") as f:
        json.dump(labels, f)

# split_dataset();