#Function to get image features
import torch
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultPredictor

import cv2

import numpy as np
from tqdm import tqdm
import os

class Targeted_Feature_Selection:
    def __init__(self,cfg,ckpt_path, category_list):
        torch.cuda.empty_cache()
        self.model = build_model(cfg)
        self.ckpt_path = ckpt_path
        DetectionCheckpointer(self.model).load(self.ckpt_path)
        self.model.eval()#make sure its in eval mode
        self.classes = category_list

    def targeted_classes_image_embeddings(self, image_file_list, targeted_classes=None, sel_strategy='weight_avg'):
        """
            sel_strategy: How to compute final embeddings:
                avg: Normal average
                weight_avg: Weighted average based on score provided by the RPN, or can be provided by user per class.
                area: Take box with the maximum area.
                score: take box with maximum score
        """
        # tmp_path = "/content/tmp"
        # try:
        #     shutil.rmtree(tmp_path)
        # except:
        #     pass

        # try:
        #     os.mkdir(tmp_path)
        # except:
        #     pass

        # image_file_list = os.listdir(image_file_path)
        # image_file_list = [os.path.join(image_file_path, x) for x in image_file_list]
        if targeted_classes is None:
            targeted_classes = ['Equation','Table']
        elif type(targeted_classes)!= list:
            raise Exception("ERROR: targeted_classes has to be list")


        target_ids = []
        for target in targeted_classes:
            target_ids.append(self.classes.index(target)+1)

        inputs = []
        for f in image_file_list:
            image = cv2.imread(f)
            image = cv2.resize(image, (1000, 1000))
            height, width = image.shape[:2]
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs.append({"image": image, "height": height, "width": width})

        embeddings = []
        targeted_pages = []

        for k,inp in enumerate(inputs):

            with torch.no_grad():
                images = self.model.preprocess_image([inp])  # don't forget to preprocess
                features = self.model.backbone(images.tensor)  # set of cnn features
                proposals, _ = self.model.proposal_generator(images, features, None)  # RPN

                features_ = [features[f] for f in self.model.roi_heads.box_in_features]
                box_features = self.model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
                box_features = self.model.roi_heads.box_head(box_features)  # features of all 1k candidates
                predictions = self.model.roi_heads.box_predictor(box_features)
                pred_instances, pred_inds = self.model.roi_heads.box_predictor.inference(predictions, proposals)
                pred_instances = self.model.roi_heads.forward_with_given_boxes(features, pred_instances)

                # output boxes, masks, scores, etc
                pred_instances = self.model._postprocess(pred_instances, inputs, images.image_sizes)  # scale box to orig size
                # print(box_features.shape)
                # features of the proposed boxes
                feats = box_features[pred_inds]

            # print(pred_instances)
            for i in range(len(pred_instances)):
                fields = pred_instances[i]['instances'].get_fields()
                pred_boxes = fields["pred_boxes"]
                scores = fields["scores"]
                pred_classes = fields["pred_classes"]
                areas = pred_boxes.area()

                np_scores = scores.cpu().detach().numpy()
                np_feats = feats.cpu().detach().numpy()
                np_areas = areas.cpu().detach().numpy()
                np_pred_classes = pred_classes.cpu().detach().numpy()
                # list_pred_classes = list(np_pred_classes)
                targeted_indices = []
                for id in target_ids:
                    temp = np.where(np_pred_classes==id)[0]
                    targeted_indices.append(list(temp))
                targeted_indices = [item for sublist in targeted_indices for item in sublist]
                if len(targeted_indices)==0:
                    break
                # print(np_pred_classes[targeted_indices])
                targeted_scores = np_scores[targeted_indices]
                targeted_features = np_feats[targeted_indices]
                targeted_areas = np_areas[targeted_indices]

                # print(targeted_scores, targeted_features, targeted_areas)


                if sel_strategy=="avg":
                    final_image_embedding = np.mean(targeted_features, axis=0)
                elif sel_strategy=="weight_avg":
                    final_image_embedding = np.transpose(np.matmul(np.transpose(targeted_features), targeted_scores)/np.sum(targeted_scores))
                elif sel_strategy=="area":
                    max_area = np.argmax(targeted_areas)
                    # print(max_area)
                    final_image_embedding = targeted_features[max_area]
                elif sel_strategy=="score":
                    max_score = np.argmax(targeted_scores)
                    final_image_embedding = targeted_features[max_score]


                ## weighted average by area

                # print(final_image_embedding)

                # final_image_embedding = np.transpose(np.matmul(np.transpose(np_feats), np_scores)/np.sum(np_scores))
                embeddings.append(final_image_embedding)
                targeted_pages.append(image_file_list[k])

        return np.array(embeddings), targeted_pages

    
class ImageModel:

    def __init__(self, cfg) -> None:
        self.predictor = DefaultPredictor(cfg)


    def get_embeddings_image_list(self, image_dir, targeted_classes, is_query=False, ground_data = None, sel_strategy='area'):
        # OLD_MAPPING = {
        #     "1":'abstract', "2": 'author', "3": 'caption', "4": 'equation', "5": 'figure', "6": 'footer', 
        #       "7":'list', "8": 'paragraph', "9": 'reference', "10": 'section', "11": 'table', "12": 'title', "13": "date"}


        OLD_MAPPING = {
            "0":"aeroplane", "1":"bicycle", "2":"bird", "3":"boat", "4":"bottle", "5":"bus", "6":"car", "7":"cat",
            "8":"chair", "9":"cow", "10":"diningtable", "11":"dog", "12":"horse", "13":"motorbike", "14":"person", 
            "15":"pottedplant", "16":"sheep", "17":"sofa", "18":"train", "19":"tvmonitor"}

        CLASSES = [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
            "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        if is_query==True:
            final_im_list = None
        else:
            final_im_list = []

        embeddings = []

        if ground_data is None:
            with torch.no_grad():
                for im in tqdm(sorted(os.listdir(image_dir))):
                    image_path = os.path.join(image_dir, im)
                    image = cv2.imread(image_path)
                    height, width = image.shape[:2]
                    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                    inputs = [{"image": image, "height": height, "width": width}]
                    images = self.predictor.model.preprocess_image(inputs)
                
                    features = self.predictor.model.backbone(images.tensor)
                    proposals, _ = self.predictor.model.proposal_generator(images, features)
                    instances, _ = self.predictor.model.roi_heads(images, features, proposals)
                    pred_classes = instances[0].pred_classes.cpu().numpy().tolist()
                    pred_scores = instances[0].scores.cpu().numpy().tolist()

                    # new_pred_class = []
                    # for pred in pred_classes:
                    #     new_pred_class.append(pred)

                    # pred_classes = new_pred_class
                    # if 0 in pred_classes:
                    #     raise Exception("Lol")
                    pred_boxes = instances[0].pred_boxes.tensor.cpu().numpy()

                    box_features = [features[f] for f in self.predictor.model.roi_heads.in_features]
                    box_features_from_pooler = self.predictor.model.roi_heads.box_pooler(box_features, [x.pred_boxes for x in instances])
                    box_features_from_head = self.predictor.model.roi_heads.box_head(box_features_from_pooler)


                    best_bbox_area = {OLD_MAPPING[str(cl)]:0 for cl in pred_classes}
                    best_bbox = {OLD_MAPPING[str(cl)]:0 for cl in pred_classes}

                    if sel_strategy == 'area':
                        #Max area bounding box
                        for i,cl in enumerate(pred_classes):
                            cl = OLD_MAPPING[str(cl)]
                            if is_query or cl in targeted_classes:
                                # print("here!")
                                area = (pred_boxes[i,2]-pred_boxes[i,0])*(pred_boxes[i,3]-pred_boxes[i,1])
                                # print(area)
                                if area>best_bbox_area[cl]:
                                    best_bbox[cl] = box_features_from_head[i,:].cpu().numpy()
                                    best_bbox_area[cl] = area
                                    # print("Here")
                        for k in best_bbox:
                            if isinstance(best_bbox[k], np.ndarray):
                                best_bbox[k] = best_bbox[k].reshape((1,-1))
                                # print(best_bbox[k].shape)
                                # print("here")
                                if isinstance(embeddings, np.ndarray):
                                    # print("1here")
                                    embeddings = np.append(embeddings, best_bbox[k], axis=0)
                                else:
                                    # print("2here")
                                    embeddings = best_bbox[k]
                                if final_im_list is not None:
                                    final_im_list.append(im)

                    elif sel_strategy == 'weight_avg':
                        box_features_weights=[];
                        box_score=[]
                        for i,cl in enumerate(pred_classes):
                            cl = OLD_MAPPING[str(cl)]
                            if cl in targeted_classes:
                                box_features_weights.append(box_features_from_head[i,:].cpu().numpy())
                                box_score.append(pred_scores[i]);
                        
                        if(box_features_weights):
                            weitage_avg = (np.transpose(np.matmul(np.transpose(box_features_weights), box_score)/np.sum(box_score)))
                            if(isinstance(weitage_avg, np.ndarray)):
                                weitage_avg = weitage_avg.reshape((1,-1));
                                if isinstance(embeddings, np.ndarray):
                                    # print("1here")
                                    embeddings = np.append(embeddings, weitage_avg, axis=0)
                                else:
                                    # print("2here")
                                    embeddings = weitage_avg;
                                if final_im_list is not None:
                                    final_im_list.append(im)
                    elif sel_strategy == 'avg':
                        box_features_weights=[];
                        box_score=[]
                        for i,cl in enumerate(pred_classes):
                            cl = OLD_MAPPING[str(cl)]
                            if cl in targeted_classes:
                                box_features_weights.append(box_features_from_head[i,:].cpu().numpy())
                        
                        if(box_features_weights):
                            weitage_avg = np.mean(box_features_weights, axis=0);
                            if(isinstance(weitage_avg, np.ndarray)):
                                weitage_avg = weitage_avg.reshape((1,-1));
                                if isinstance(embeddings, np.ndarray):
                                    # print("1here")
                                    embeddings = np.append(embeddings, weitage_avg, axis=0)
                                else:
                                    # print("2here")
                                    embeddings = weitage_avg;
                                if final_im_list is not None:
                                    final_im_list.append(im)

                    del features, proposals, instances, pred_classes, box_features, box_features_from_pooler, box_features_from_head
                    # break

        else:
            raise NotImplementedError

        return final_im_list, embeddings
