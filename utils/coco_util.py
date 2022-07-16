import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import json
# from pprint import PrettyPrinter as pprint
# from IPython.display import JSON
import re
from tqdm import tqdm
import traceback
from PIL import Image

class COCOData:
    """
        COCOData class allows DocBank dataset to be converted to COCO Format.

        Functions Available:

        1. read_src_folder(src_path, dest_path): Reads all the DocBank JSON label files from the provided parent path and stores the destination path for later use to save the converted labels.
        2. create_dict_layout(): Creates a basic layout for COCO format with basic static information.
        3. set_image_properties(file_name, image_id): Sets Image Properties. Used in convert_to_coco() function.
        4. set_caption_properties(object_dict, doc_object): Set Caption Properties. Used in set_object_properties() function.
        5. set_object_properties(doc_object, doc_object_id, image_id): Set the Object properties. Used in convert_to_coco() function.
        6. convert_to_coco(): Convert the source dataset to COCO format and store the converted data in coco_dictionary.
        7. save_coco_dataset(): Saves the converted dataset into the destination folder (Destination Folder was provided in read_src_folder function).
    """
    

    
    def __init__(self,coco_file_path,examples):
        self.coco_file_path = coco_file_path
        self.coco_dictionary = []
        self.examples = examples

    def create_dict_layout(self):
        temp_dict = {}
        temp_dict["info"] = {
            "year": "",
            "version": "1",
            "description": "",
            "contributor": "",
            "url": "",
            "date_created": "",
        }
        temp_dict["licenses"] = []
        
        ['abstract',
         'author',
         'caption',
         'equation',
         'figure',
         'footer',
         'list',
         'paragraph',
         'reference',
         'section',
         'table',
         'title',
         'date']
        temp_dict["categories"] = [{"id": 0,"name": "Abstract","supercategory": ""},{"id": 1,"name": "Author","supercategory": ""},
            {"id": 2,"name": "Caption","supercategory": ""},{"id": 3,"name": "Equation","supercategory": ""},
            {"id": 4,"name": "Figure","supercategory": ""},{"id": 5,"name": "Footer","supercategory": ""},
            {"id": 6,"name": "List","supercategory": ""},{"id": 7,"name": "Paragraph","supercategory": ""},
            {"id": 8,"name": "Reference","supercategory": ""},{"id": 9,"name": "Section","supercategory": ""},
            {"id": 10,"name": "Table","supercategory": ""},{"id": 11,"name": "Title","supercategory": ""},
            {"id": 12,"name": "Date","supercategory": ""}]
        # temp_dict["categories"] = [{"id": 1,"name": "Abstract","supercategory": ""},{"id": 2,"name": "Author","supercategory": ""},
        #     {"id": 3,"name": "Caption","supercategory": ""},{"id": 4,"name": "Equation","supercategory": ""},
        #     {"id": 5,"name": "Figure","supercategory": ""},{"id": 6,"name": "Footer","supercategory": ""},
        #     {"id": 7,"name": "List","supercategory": ""},{"id": 8,"name": "Paragraph","supercategory": ""},
        #     {"id": 9,"name": "Reference","supercategory": ""},{"id": 10,"name": "Section","supercategory": ""},
        #     {"id": 11,"name": "Table","supercategory": ""},{"id": 12,"name": "Title","supercategory": ""},
        #     {"id": 13,"name": "Date","supercategory": ""}]
        temp_dict["images"] = []
        temp_dict["annotations"] = []
        
        return temp_dict
    
    # Image denotes the image of a page where a set of objects exist
    def set_image_properties(self, example, image_id):
        image_name = example.filepath
        width, height = example.pagesize

        image_dict = {
            "id": image_id,
            "license": "",
            "file_name": image_name,
            "height": height,
            "width": width,
            "date_captured": "",
        } 
        return image_dict
     
    # Object denotes either a Table or Figure
    def set_object_properties(self, doc_object, doc_object_id, image_id):
        x1,y1,x2,y2,category = doc_object.split("\t")
        object_dict = {}
        object_dict["id"] = doc_object_id
        object_dict["image_id"] = image_id
        object_dict["iscrowd"] = 0
        object_dict["segmentation"] = []
        
        # category_list = {
            # 'abstract': 1, 'author': 2, 'caption': 3, 'equation': 4, 'figure': 5, 'footer': 6, 
            #  'list': 7, 'paragraph': 8, 'reference': 9, 'section': 10, 'table': 11, 'title': 12, "date": 13}
        category_list = {
            'abstract': 0, 'author': 1, 'caption': 2, 'equation': 3, 'figure': 4, 'footer': 5, 
             'list': 6, 'paragraph': 7, 'reference': 8, 'section': 9, 'table': 10, 'title': 11, "date": 12}
            
        object_dict["category_id"] = category_list[category]
        object_width = int(x2) - int(x1)
        object_height = int(y2) - int(y1)
        
        object_dict["bbox"] = [
            int(x1),
            int(y1),
            int(object_width),
            int(object_height)
        ]
        object_dict["area"] = int(object_width * object_height) 
        
        return object_dict
    
    def convert_to_coco(self):
        try:
            # Init Image ID
            image_id = 0
            # Init Object ID
            doc_object_id = 0

            json_dict = self.create_dict_layout()

            # Fetch each JSON file present in the folders
            for example in tqdm(self.examples, desc="Converting annotoations to COCO JSON"):
                image_dict = self.set_image_properties(example, image_id)
                # Each Image present in the file is fetched and added to a cocoData object
                bbox = example.print_bbox()
                bbox_list = bbox.split("\n")
                bbox_list = [x for x in bbox_list if len(x)>0]

                for doc_object in bbox_list:
                    object_dict = self.set_object_properties(doc_object, doc_object_id, image_id)
                    # Add the object properties to the annotations key in COCO
                    json_dict["annotations"].append(object_dict)
                    # Increment the object ID for next annotated object in the file
                    doc_object_id += 1

                # Increment the Image ID for the next Image in the file
                image_id+=1
                # Extract Image width and height if annotations exist. There has to be atleast one annotation for an image to have the dimension attributes.
                json_dict["images"].append(image_dict)
            self.coco_dictionary.append(json_dict)
        except:
            traceback.print_exc()

    # Converts final dictionary in COCO format for storing into file.
    def save_coco_dataset(self):
        try:
            coco_file_dir = os.path.split(self.coco_file_path)[0]
            if not os.path.exists(coco_file_dir):
                # Creates the parent folder and all the subfolders for the file.
                #   Does not throw an error if parent or any subfolders already exists.
                Path(coco_file_dir).mkdir(parents=True, exist_ok=True)

            output_file = open(self.coco_file_path, mode="w")
            output_file.writelines(json.dumps(self.coco_dictionary[0], indent=4))
        except:
            traceback.print_exc()
        # finally:
        #     # output_file.close()
