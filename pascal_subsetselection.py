import os
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.engine import default_argument_parser
from CBIR.src.DB import Database
from CBIR.src.evaluate import infer_subset
from CBIR.src.resnet import ResNetFeat
from utils.detectron_util import create_model, crop_images_classwise, crop_images_classwise_ground_truth, do_evaluate, remove_dataset
from utils.pascal_util import create_pascal_db
from utils.submodlib_util import Margin_Sampling, Random_wrapper
import pandas as pd
from detectron2 import model_zoo
from utils.util import aug_train_subset, create_dir, create_new_query, find_missclassified_object, get_category_details, get_lakeset_images, get_original_images_path, remove_dir

import torch

torch.cuda.set_device(1)
parser = default_argument_parser()

parser.add_argument("--output_path", default="fl1mi_test", help="Output_path")
parser.add_argument("--strategy",
                    default="fl1mi",
                    help="subset selection strategy")
parser.add_argument("--total_budget",
                    default="200",
                    type=int,
                    help="Total AL budget")
parser.add_argument("--budget",
                    default="20",
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
                    default="list",
                    type=str,
                    help="Targeted class")
parser.add_argument("--private_category",
                    default="text",
                    type=str,
                    help="Private Targeted class")

arg = parser.parse_args()
print(arg)
query_path = 'query_data_img/' + arg.category
private_query_path = 'query_data_img/' + arg.private_category
category = [arg.category]
private_category = [arg.private_category]

train_data_dirs = ("pascal2007/train_data_img",
                   "pascal2007/train_targeted.json")
lake_data_dirs = ("pascal2007/lake_data_img", "pascal2007/lake_targeted.json")
test_data_dirs = ("pascal2007/test_data_img", "pascal2007/test_targeted.json")
val_data_dirs = ("pascal2007/val_data_img", "pascal2007/val_targeted.json")

train_path = 'model_result'
training_name = arg.output_path
model_path = os.path.join(train_path, training_name)
if (not os.path.exists(model_path)):
    create_dir(model_path)

# step 1
# train a faster_rcnn model on the initial_set
output_dir = os.path.join(model_path, "initial_training")
config_file_path = '../publaynet/configs/publaynet/faster_rcnn_R_101_FPN_3x.yaml'
prediction_score_threshold = 0.7
selection_strag = arg.strategy
selection_budget = arg.budget
budget = arg.total_budget

cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("initial_set", )
cfg.DATASETS.TEST = ("val_set", "test_set")
cfg.DATALOADER.NUM_WORKERS = 6
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
)  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 2500  #adjust up if val mAP is still rising, adjust down if overfit
cfg.SOLVER.STEPS = (1000, 1300)
cfg.SOLVER.GAMMA = 0.05
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20  #your number of classes + 1
# cfg.TEST.EVAL_PERIOD = 500
cfg.MODEL.RPN.NMS_THRESH = 0.8

cfg.OUTPUT_DIR = output_dir
cfg.TRAINING_NAME = training_name

logger = setup_logger(os.path.join(output_dir, cfg.TRAINING_NAME))

create_pascal_db()

# given an initial_set and test_set dataset
register_coco_instances("initial_set", {}, train_data_dirs[1],
                        train_data_dirs[0])
register_coco_instances("test_set", {}, test_data_dirs[1], test_data_dirs[0])
register_coco_instances("val_set", {}, val_data_dirs[1], val_data_dirs[0])

logger.info("Starting Initial_set Training")
cfg.MODEL_WEIGHTS = 'pascal2007/model_final_68b088.pkl'
model = create_model(cfg)
model.train()
logger.info("Initial_set training complete")

iteration = 100
result_val = []
result_test = []
# step 2
# evaluate the inital model and get worst performing class
cfg.MODEL.WEIGHTS = cfg.OUTPUT_DIR + "/model_final.pth"
model = create_model(cfg, "test")
result = do_evaluate(cfg, model, output_dir)
result_val.append(result['val_set'])
result_test.append(result['test_set'])
category_selection = []
i = 0
try:
    while (i < iteration and budget > 0):
        # creating different flow for the different smi strategies
        query_path, category = find_missclassified_object(result['test_set'])

        # dynamic query images after each iteraion
        create_new_query(train_data_dirs, query_path, category)

        if (selection_strag != "margin"):

            if (selection_strag == "margin"):
                subset_result = Margin_Sampling(lake_data_dirs[0], query_path,
                                                model, budget)
            elif (selection_strag != "random"):
                # cropping and calculating the resnet embedding for the lake dataset images

                # creating new query set for under performing class for each iteration
                remove_dir(os.path.join(model_path, "query_images"))
                try:
                    os.remove(os.path.join(model_path, "data_query.csv"))
                except:
                    pass

                if (i <= 40):
                    crop_images_classwise_ground_truth(
                        train_data_dirs[1], query_path,
                        os.path.join(model_path, "query_images"), category)
                    if selection_strag == 'cmi':
                        crop_images_classwise_ground_truth(
                            train_data_dirs[1], private_query_path,
                            os.path.join(model_path, "query_images"),
                            private_category)
                else:
                    crop_images_classwise(
                        model, query_path,
                        os.path.join(model_path, "query_images"))
                db2 = Database(dir=os.path.join(model_path, "query_images"),
                               csv=os.path.join(model_path, "data_query.csv"))

                # # removing the old images calculation
                f_model = ResNetFeat()
                query_set_embeddings = f_model.make_samples(
                    db2, cache_path="query-" + str(i))

                # removing the previous crop images
                remove_dir(os.path.join(model_path, "lake_images"))
                try:
                    os.remove(os.path.join(model_path, "data.csv"))
                except:
                    pass
                # creating the new crop images
                crop_images_classwise(model, lake_data_dirs[0],
                                      os.path.join(model_path, "lake_images"))
                db = Database(dir=os.path.join(model_path, "lake_images"),
                              csv=os.path.join(model_path, "data.csv"))

                # Obtaining the new lake embedding using new trained model
                lake_set_embeddings = f_model.make_samples(
                    db,
                    cache_path="lake-" + str(i),
                    RES_model="resnet152",
                    pick_layer="avg")

                AP, subset_result = infer_subset(query_set_embeddings,
                                                 lake_set_embeddings,
                                                 budget=selection_budget,
                                                 strategy=selection_strag,
                                                 clazz=category,
                                                 private=None)

                subset_result = list(
                    set(get_original_images_path(subset_result)))
                print(subset_result)
            else:
                lake_image_list = os.listdir(lake_data_dirs[0])
                subset_result = Random_wrapper(lake_image_list,
                                               selection_budget)

        # reducing the selection budget
        budget -= len(subset_result)
        if (budget > 0):

            # transferring images from lake set to train set
            aug_train_subset(subset_result, train_data_dirs[1],
                             lake_data_dirs[1], budget, lake_data_dirs,
                             train_data_dirs)
            # image_list = get_category_details(
            #         subset_result, train_data_dirs, category)
            # category_selection.append([category[0], image_list])
            # print(category_selection)
        # removing the old training images from the detectron configuration and adding new one
        remove_dataset("initial_set")
        register_coco_instances("initial_set", {}, train_data_dirs[1],
                                train_data_dirs[0])

        del model
        torch.cuda.empty_cache()
        # before starting the model active learning loop, calculating the embedding of the lake datset
        cfg.MODEL.WEIGHTS = cfg.OUTPUT_DIR + "/model_final.pth"
        cfg.SOLVER.MAX_ITER = 1500
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

    # step 3
    # get embeddings for initial and lakeset from RESNET50

    # step 4
    # select query images from initial set based on worst performing class

    # step 5
    # subset selection based on query and lakeset embeddings
