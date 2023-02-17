import cProfile
import io
import os
import pstats
from pstats import SortKey

import pandas as pd
import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import default_argument_parser
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo

from CBIR.src.DB import Database
from CBIR.src.evaluate import infer_subset
from CBIR.src.resnet import ResNetFeat
from utils.detectron_util import (create_model, crop_images_classwise,
                                  crop_images_classwise_ground_truth,
                                  do_evaluate, remove_dataset)
from utils.submodlib_util import Random_wrapper
from utils.util import (aug_train_subset, create_dir, create_new_query,
                        get_category_details, get_lakeset_images,
                        get_original_images_path_sanskrit, remove_dir)

parser = default_argument_parser()

parser.add_argument("--output_path", default="sanskrit_gcmi", help="Output_path")
parser.add_argument("--strategy",
                    default="fl2mi",
                    help="subset selection strategy")
parser.add_argument("--total_budget",
                    default="500",
                    type=int,
                    help="Total AL budget")
parser.add_argument("--budget",
                    default="10",
                    type=int,
                    help="selection budget")
parser.add_argument("--lake_size",
                    default="100",
                    type=int,
                    help="selection budget")
parser.add_argument("--train_size",
                    default="100",
                    type=int,
                    help="selection budget")
parser.add_argument("--category",
                    default="Table",
                    type=str,
                    help="Targeted class")
parser.add_argument("--device", default="1", type=int, help="GPU device")
parser.add_argument("--proposal_budget",
                    default="30",
                    type=int,
                    help="Proposal Budget for each image objects")
parser.add_argument("--iterations",
                    default="10",
                    type=int,
                    help="Active learning iteration")
arg = parser.parse_args()
print(arg)
query_path = 'query_data_img/' + arg.category
category = [arg.category]
torch.cuda.set_device(arg.device)
dataset_path = "sanskrit_tada1" if arg.device is 0 else "sanskrit_tadad"
dataset_dir = ("../dataset/sanskrit-layout-gt-manual-old/train/images",
               "../dataset/sanskrit-layout-gt-manual-old/train/train.json")
train_data_dirs = (dataset_path+"/train_data_img",
                   dataset_path+"/train_targeted.json")
lake_data_dirs = (dataset_path+"/lake_data_img",
                  dataset_path+"/lake_targeted.json")
test_data_dirs = ("../dataset/sanskrit_tada/test_data_img",
                  "../dataset/sanskrit_tada/test_targeted.json")
val_data_dirs = ("../dataset/sanskrit_tada/val_data_img",
                 "../dataset/sanskrit_tada/val_targeted.json")

train_path = 'model_result/sanskrit'
training_name = arg.output_path
model_path = os.path.join(train_path, training_name)
if (not os.path.exists(model_path)):
    create_dir(model_path)

# train a faster_rcnn model on the initial_set
output_dir = os.path.join(model_path, "initial_training")
config_file_path = '../publaynet/configs/coco/faster_rcnn_R_101_FPN_3x.yaml'
selection_strag = arg.strategy
selection_budget = arg.budget
budget = arg.total_budget
proposal_budget = arg.proposal_budget

cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("initial_set", )
cfg.DATASETS.TEST = ("val_set", "test_set")
cfg.DATALOADER.NUM_WORKERS = 6
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.INPUT.MAX_SIZE_TEST = 1247
cfg.INPUT.MAX_SIZE_TRAIN = 1247
cfg.INPUT.MIN_SIZE_TEST = 743
cfg.INPUT.MIN_SIZE_TRAIN = (743, )
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001  # pick a good LR
cfg.SOLVER.STEPS = []  # do not decay learning rate
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[1.0, 3.0, 10.0]]
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 2000  # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.7
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
cfg.MODEL.BACKBONE.FREEZE_AT = 5
cfg.SOLVER.MAX_ITER = 3000
cfg.MODEL.RPN.NMS_THRESH = 0.5
# cfg.TEST.EVAL_PERIOD = 2000
cfg.OUTPUT_DIR = output_dir
cfg.TRAINING_NAME = training_name

logger = setup_logger(os.path.join(output_dir, cfg.TRAINING_NAME))

# category ratio based on the publaynet paper.
# intial lake set images from intial dataset path
# category_ratio = {
#     "bg": .71,
#     "Image": .19,
#     "Math": .024,
#     "Table": .031,
#     "Text": 0.05
# }

# Registering dataset intial_set for initial training, test_set and val_set for test and validation respectively.
register_coco_instances("initial_set", {}, train_data_dirs[1],
                        train_data_dirs[0])
register_coco_instances("test_set", {}, test_data_dirs[1], test_data_dirs[0])
register_coco_instances("val_set", {}, val_data_dirs[1], val_data_dirs[0])

# step 1
# train a faster_rcnn model on the initial_set.
logger.info("Starting Initial_set Training")
model = create_model(cfg)
torch.cuda.empty_cache()
model.train()
logger.info("Initial_set training complete")

iteration = arg.iterations
result_val = []
result_test = []

del model
torch.cuda.empty_cache()
# step 2
# evaluate the inital model and get worst performing classcfg.MODEL.WEIGHTS = cfg.OUTPUT_DIR + "/model_final.pth
cfg.MODEL.WEIGHTS = cfg.OUTPUT_DIR + "/model_final.pth"
model = create_model(cfg, "test")
result = do_evaluate(cfg, model, output_dir)
result_val.append(result['val_set'])
result_test.append(result['test_set'])
category_selection = []
i = 0
try:
    while (i < iteration and budget > 0):
        # step 3
        # get embeddings for initial and lakeset from RESNET101

        # creating different flow for the different smi strategies
        # query_path, category = find_missclassified_object(
        #     result['test_set'])

        # dynamic query images after each iteraion
        # create_new_query(train_data_dirs, query_path, category)

        if (selection_strag != "random"):

            # creating new query set for under performing class for each iteration
            remove_dir(os.path.join(model_path, "query_images"))
            try:
                os.remove(os.path.join(model_path, "data_query.csv"))
            except:
                pass

            # Cropping object based on ground truth for the query set.
            # The set is part of train set, so no need of using object detection model to find the bounding box.
            if (i <= iteration):
                crop_images_classwise_ground_truth(
                    train_data_dirs[1], query_path,
                    os.path.join(model_path, "query_images"), category)
            else:
                crop_images_classwise(model, query_path,
                                      os.path.join(model_path, "query_images"))
            db2 = Database(dir=os.path.join(model_path, "query_images"),
                           csv=os.path.join(model_path, "data_query.csv"))

            # Getting feature map for query set using pre-trained classification model. ResNet.
            f_model = ResNetFeat()
            query_set_embeddings = f_model.make_samples(db2,
                                                        cache_path="query-" +
                                                        str(i))

            # removing the previous cropped images
            remove_dir(os.path.join(model_path, "lake_images"))
            try:
                os.remove(os.path.join(model_path, "data.csv"))
            except:
                pass

            # Cropping object based on the bouding box detection using trained model
            # We use object detection model to obtain bouding box,as we are unaware of objects in lake set.
            crop_images_classwise(model,
                                  lake_data_dirs[0],
                                  os.path.join(model_path, "lake_images"),
                                  proposal_budget=proposal_budget)
            db = Database(dir=os.path.join(model_path, "lake_images"),
                          csv=os.path.join(model_path, "data.csv"))

            # Obtaining the new lake embedding using new trained model
            lake_set_embeddings = f_model.make_samples(db,
                                                       cache_path="lake-" +
                                                       str(i),
                                                       RES_model="resnet101",
                                                       pick_layer="avg")

            # step 4
            # subset selection based on query and lakeset embeddings
            '''
            The below code uses the default libary for kernel computation and  subset selection.
            '''
            pr = cProfile.Profile()
            pr.enable()
            AP, subset_result = infer_subset(query_set_embeddings,
                                             lake_set_embeddings,
                                             budget=selection_budget,
                                             strategy=selection_strag,
                                             clazz=category)
            pr.disable()
            s = io.StringIO()
            sortby = SortKey.CUMULATIVE
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()

            print(s.getvalue())
            '''
            The below method first compute the query kernel matrix and make use of subset selection library to find subset.
            '''
            # query_embedding = []

            # for idx, query in enumerate(query_set_embeddings, start=1):
            #     q_img, q_cls, q_hist = query['img'], query['cls'], query['hist']
            #     if q_cls in category:
            #         query_embedding.append(q_hist)

            # lake_embedding = []
            # lake_image_list = []
            # for idx, sample in enumerate(lake_set_embeddings, start=1):
            #     s_img, s_cls, s_hist = sample['img'], sample['cls'], sample['hist']
            #     lake_embedding.append(s_hist)
            #     lake_image_list.append(s_img)
            # query_sij = compute_queryimage_kernel(np.stack(query_embedding), np.stack(lake_embedding));
            # subset_result = subset([], query_sij, 1,lake_image_list, budget=selection_budget, metric='cosine',
            #             stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False, strategry=selection_strag, kernel=True);

            # subset_result = list(set(subset_result))
            # print(subset_result)
            subset_result = list(set(get_original_images_path_sanskrit(subset_result)))
            print(subset_result)
        else:
            lake_image_list = os.listdir(lake_data_dirs[0])
            subset_result = Random_wrapper(lake_image_list, selection_budget)

        # reducing the selection budget
        budget -= len(subset_result)
        if (budget > 0):

            # transferring images from lake set to train set
            aug_train_subset(subset_result, train_data_dirs[1],
                             lake_data_dirs[1], budget, lake_data_dirs,
                             train_data_dirs)
            image_list = get_category_details(subset_result, train_data_dirs,
                                              category)
            category_selection.append([category[0], image_list])
            print(category_selection)
        # removing the old training images from the detectron configuration and adding new one
        remove_dataset("initial_set")
        register_coco_instances("initial_set", {}, train_data_dirs[1],
                                train_data_dirs[0])

        del model
        torch.cuda.empty_cache()
        # before starting the model active learning loop, calculating the embedding of the lake datset
        cfg.MODEL.WEIGHTS = cfg.OUTPUT_DIR + "/model_final.pth"
        cfg.SOLVER.MAX_ITER = 3000
        model = create_model(cfg, "train")
        model.train()

        # reevaluating the model train once again
        del model
        torch.cuda.empty_cache()
        # before starting the model active learning loop, calculating the embedding of the lake datset
        cfg.MODEL.WEIGHTS = cfg.OUTPUT_DIR + "/model_final.pth"
        model = create_model(cfg, "test")
        result = do_evaluate(cfg, model, output_dir)
        result_val.append(result['val_set'])
        result_test.append(result['test_set'])

        # increasing the iteration number
        # publishing each iteration result to csv
        i += 1
        print("remaining_budget", budget)
        final_data = []
        temp = []
        for it in result_val:
            print(it)
            for k, val in it.items():
                temp = list(val.keys())
                final_data.append(list(val.values()))
        csv = pd.DataFrame(final_data, columns=temp)
        csv.to_csv(
            os.path.join(output_dir,
                         '{}'.format("val_scores" + selection_strag + ".csv")))
        final_data = []
        for it in result_test:
            print(it)
            for k, val in it.items():
                temp = list(val.keys())
                final_data.append(list(val.values()))
        csv = pd.DataFrame(final_data, columns=temp)
        csv.to_csv(
            os.path.join(output_dir, '{}'.format("test_scores" +
                                                 selection_strag + ".csv")))
except Exception as e:
    logger.error("Error while training:", e)

finally:
    final_data = []
    temp = []
    for i in result_val:
        print(i)
        for k, val in i.items():
            temp = list(val.keys())
            final_data.append(list(val.values()))
    csv = pd.DataFrame(final_data, columns=temp)
    csv.to_csv(
        os.path.join(output_dir,
                     '{}'.format("val_scores" + selection_strag + ".csv")))
    final_data = []
    for i in result_test:
        print(i)
        for k, val in i.items():
            temp = list(val.keys())
            final_data.append(list(val.values()))
    csv = pd.DataFrame(final_data, columns=temp)
    csv.to_csv(
        os.path.join(output_dir,
                     '{}'.format("test_scores" + selection_strag + ".csv")))
