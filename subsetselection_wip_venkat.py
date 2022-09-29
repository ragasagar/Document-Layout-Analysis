from calendar import c
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from training_layerwise import train_layerwise
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
import os, json, cv2, random, shutil
from detectron2.modeling import build_model
from detectron2.engine import DefaultPredictor
from PIL import Image
from tqdm import tqdm

#given an initial_set and test_set dataset
register_coco_instances("initial_set",{}, "../publaynet/train_targeted.json", "../publaynet/train_data_img")
register_coco_instances("test_set",{}, "../publaynet/test_targeted.json", "../publaynet/test_data_img")


training_name='firstattempt'

#step 1 
#train a faster_rcnn model on the initial_set
''' output_dir=training_name+"/initial_training"
config_file_path = '../publaynet/configs/publaynet/faster_rcnn_R_101_FPN_3x.yaml'
prediction_score_threshold = 0.7
cfg = get_cfg()
cfg.merge_from_file(config_file_path)
cfg.DATASETS.TRAIN = ("initial_set",)
cfg.DATASETS.TEST = ('test_set',)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.TEST.EVAL_PERIOD = 500
cfg.MODEL.BACKBONE.FREEZE_AT = 5
cfg.SOLVER.MAX_ITER = 15000
cfg.MODEL.RPN.NMS_THRESH = 0.5
cfg.TEST.EVAL_PERIOD = 2000
cfg.OUTPUT_DIR = output_dir
cfg.TRAINING_NAME=training_name
logger = setup_logger(cfg.TRAINING_NAME)

logger.info("Starting Initial_set Training")
train_layerwise(cfg)
logger.info("Initial_set training complete")

 '''



def crop_object(image, box):
    """Crops an object in an image

  Inputs:
    image: PIL image
    box: one box from Detectron2 pred_boxes
    """
    x_top_left = box[0]
    y_top_left = box[1]
    x_bottom_right = box[2]
    y_bottom_right = box[3]
    x_center = (x_top_left + x_bottom_right) / 2
    y_center = (y_top_left + y_bottom_right) / 2

    crop_img = image.crop((int(x_top_left), int(y_top_left), int(x_bottom_right), int(y_bottom_right)))
    return crop_img

def crop_image_classwise():
    #evaluate the inital model and get worst performing class
    output_dir=training_name+"/test_after_initial_training"
    if not os.path.exists(output_dir+'/obj_images'):
        os.makedirs(output_dir+'/obj_images')
    obj_im_dir=output_dir+'/obj_images'
    dataset_dicts_validation = DatasetCatalog.get("test_set")

    cfg = get_cfg()
    config_file_path = '../publaynet/configs/publaynet/faster_rcnn_R_101_FPN_3x.yaml'
    cfg.merge_from_file(config_file_path)
    cfg.DATASETS.TRAIN = ("initial_set",)
    cfg.DATASETS.TEST = ('test_set',)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.TEST.EVAL_PERIOD = 500
    cfg.MODEL.BACKBONE.FREEZE_AT = 5
    cfg.SOLVER.MAX_ITER = 15000
    cfg.MODEL.RPN.NMS_THRESH = 0.5
    cfg.TEST.EVAL_PERIOD = 2000
    cfg.OUTPUT_DIR = output_dir
    cfg.TRAINING_NAME = training_name
    
    logger = setup_logger(cfg.TRAINING_NAME)


    #get the weights trained in the first step
    weights_path=training_name+"/initial_training/model_final.pth"
    #weights_path="/home/venkat/targeted_learning_docbank/publaynet/intial_train_img/model_final.pth"
    cfg.MODEL.WEIGHTS=weights_path
    predictor=DefaultPredictor(cfg)
    MAPPING = {"0":"text", "1":"title","2":"list","3":"table","4":"figure"}


    no_of_objects=0
    for d in tqdm(dataset_dicts_validation):
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)
        print(outputs["instances"].pred_classes.cpu().numpy().tolist())
        boxes = outputs["instances"].pred_boxes
        classes=outputs["instances"].pred_classes.cpu().numpy().tolist()

        for singleclass in classes:
            if not os.path.exists(os.path.join(output_dir,'obj_images',MAPPING[str(singleclass)])):
                os.makedirs(os.path.join(output_dir,'obj_images',MAPPING[str(singleclass)]))

        img = Image.open(d["file_name"])
        for idx,box in enumerate(list(boxes)):
            no_of_objects+=1
            box = box.detach().cpu().numpy()
        
            crop_img = crop_object(img, box)      
            crop_img.save(os.path.join(obj_im_dir,MAPPING[str(classes[idx])],os.path.split(d["file_name"])[1].replace(".jpg","")+"_"+str(idx)+".jpg"))

    print("Number of objects: "+str(no_of_objects))


#step 2
crop_image_classwise()

#step 3
#get embeddings for initial and lakeset from RESNET50

#step 4 
#select query images from initial set based on worst performing class

#step 5
#subset selection based on query and lakeset embeddings





