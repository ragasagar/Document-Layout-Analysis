import os
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.engine import default_argument_parser
from CBIR.src.DB import Database
from CBIR.src.evaluate import infer_subset
from CBIR.src.resnet import ResNetFeat
from utils.detectron_util import create_model, do_evaluate, get_lake_embedding, get_query_embedding, remove_dataset
from utils.kernel_util import  compute_queryimage_kernel
from utils.submodlib_util import Random_wrapper, subset
import pandas as pd
from utils.util import aug_train_subset, create_dir, create_new_query, find_missclassified_object, get_category_details, get_lakeset_images, get_original_images_path, remove_dir

import torch


parser = default_argument_parser()

parser.add_argument("--output_path",          default="talisman_test4", help="Output_path")
parser.add_argument("--strategy", default="fl2mi", help="subset selection strategy")
parser.add_argument("--total_budget",      default="500", type=int,  help="Total AL budget")
parser.add_argument("--budget",   default="10", type=int, help="selection budget")
parser.add_argument("--lake_size",   default="100", type=int, help="selection budget")
parser.add_argument("--train_size",   default="100", type=int, help="selection budget")
parser.add_argument("--category",   default="list", type=str, help="Targeted class")
parser.add_argument("--private_category",   default="text", type=str, help="Private Targeted class")
parser.add_argument("--device",   default="0", type=int, help="GPU device")
parser.add_argument("--proposal_budget",   default="40", type=int, help="Proposal Budget for each image objects")

arg = parser.parse_args()
print(arg)
torch.cuda.set_device(arg.device)
query_path = 'query_data_img/'+ arg.category;
private_query_path = 'query_data_img/'+arg.private_category
category = [arg.category];
private_category = [arg.private_category]
proposal_budget = arg.proposal_budget

dataset_dir = ("../publaynet/publaynet/train5",
               "../publaynet/publaynet/train.json")
init_train_dataset_dir = ("../publaynet/intial_train_img/train_data_img",
                          "../publaynet/intial_train_img/train_targeted.json")
train_data_dirs = ("publaynet/train_data_img",
                   "publaynet/train_targeted.json")
# val_data_dirs = ("publaynet/train_data_img",
#                    "publaynet/train_targeted.json")
# test_data_dirs = ("publaynet/train_data_img",
#                    "publaynet/train_targeted.json")
lake_data_dirs = ("publaynet/lake_data_img",
                   "publaynet/lake_targeted.json")
test_data_dirs = ("../publaynet/test_data_img",
                  "../publaynet/test_targeted.json")
val_data_dirs = ("../publaynet/val_data_img",
                 "../publaynet/val_targeted.json")

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
cfg.merge_from_file(config_file_path)
cfg.DATASETS.TRAIN = ("initial_set",)
cfg.DATASETS.TEST = ('test_set', 'val_set')
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 6000
cfg.SOLVER.IMS_PER_BATCH = 6
cfg.MODEL.RPN.NMS_THRESH = 0.8
cfg.MODEL.RPN.POST_NMS_TOPK_TEST: 2000
cfg.TEST.EVAL_PERIOD = 1000
cfg.OUTPUT_DIR = output_dir
cfg.TRAINING_NAME = training_name

logger = setup_logger(os.path.join(output_dir, cfg.TRAINING_NAME))

# intial lake set images from intial dataset path
category_ratio = {
    "text": .71,
    "title": .19,
    "list": .024,
    "table": .031,
    "figure": 0.05
}
 
# # getting the new lake set and train set
# get_lakeset_images(dataset_dir, lake_data_dirs, int(arg.lake_size), category_ratio)
# get_lakeset_images(init_train_dataset_dir,
#                    train_data_dirs, int(arg.train_size), category_ratio)

# given an initial_set and test_set dataset
register_coco_instances(
    "initial_set", {}, train_data_dirs[1], train_data_dirs[0])
register_coco_instances("test_set", {}, test_data_dirs[1], test_data_dirs[0])
register_coco_instances("val_set", {}, val_data_dirs[1], val_data_dirs[0])

logger.info("Starting Initial_set Training")
cfg.MODEL_WEIGHTS = '../publaynet/Initial_model_weight/model_final.pkl'
model = create_model(cfg)
torch.cuda.empty_cache()
model.train()
logger.info("Initial_set training complete")

iteration = 100
result_val = []
result_test = []
# before starting the model active learning loop, calculating the embedding of the lake datset
cfg.MODEL.WEIGHTS = cfg.OUTPUT_DIR + "/model_final.pth"
del model
torch.cuda.empty_cache()

model = create_model(cfg, "test")
result = do_evaluate(cfg, model, output_dir)
result_val.append(result['val_set'])
result_test.append(result['test_set'])
category_selection = []
i = 0
try:
    while (i < iteration and budget > 0):
        # creating different flow for the different smi strategies
        # query_path, category = find_missclassified_object(
        #     result['test_set'])
        
        # dynamic query images after each iteraion
        create_new_query(train_data_dirs, query_path, category)

        if (selection_strag != "margin"):

            
            if (selection_strag != "random"):
                # cropping and calculating the resnet embedding for the lake dataset images                
                _, query_set_embeddings = get_query_embedding(model, query_path, category[0], train_json_path=train_data_dirs[1], device=arg.device)

                # Obtaining the new lake embedding using new trained model
                image_list, lake_set_embeddings = get_lake_embedding(model, lake_data_dirs[0], proposal_budget=proposal_budget)

                query_sij = compute_queryimage_kernel(query_set_embeddings, lake_set_embeddings);

                subset_result = subset([], query_sij, 1,image_list, budget=selection_budget, metric='cosine',
                           stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False, strategry=selection_strag, kernel=True);

                subset_result = list(set(subset_result))
                print(subset_result)
            else:
                lake_image_list = os.listdir(lake_data_dirs[0])
                subset_result = Random_wrapper(
                    lake_image_list, selection_budget)

        # reducing the selection budget
        budget -= len(subset_result)
        if (budget > 0):

            # transferring images from lake set to train set
            aug_train_subset(
                subset_result, train_data_dirs[1], lake_data_dirs[1], budget, lake_data_dirs, train_data_dirs)
            image_list = get_category_details(
                    subset_result, train_data_dirs, category)
            category_selection.append([category[0], image_list])
            print(category_selection)
        # removing the old training images from the detectron configuration and adding new one
        remove_dataset("initial_set")
        register_coco_instances(
            "initial_set", {}, train_data_dirs[1], train_data_dirs[0])

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
        csv.to_csv(os.path.join(output_dir, '{}'.format(
            "val_scores"+selection_strag+".csv")))
        final_data = []
        for it in result_test:
            print(it)
            for k, val in it.items():
                temp = list(val.keys())
                final_data.append(list(val.values()))
        csv = pd.DataFrame(final_data, columns=temp)
        csv.to_csv(os.path.join(output_dir, '{}'.format(
            "test_scores"+selection_strag+".csv")))
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
    csv.to_csv(os.path.join(output_dir, '{}'.format(
        "val_scores"+selection_strag+".csv")))
    final_data = []
    for i in result_test:
        print(i)
        for k, val in i.items():
            temp = list(val.keys())
            final_data.append(list(val.values()))
    csv = pd.DataFrame(final_data, columns=temp)
    csv.to_csv(os.path.join(output_dir, '{}'.format(
        "test_scores"+selection_strag+".csv")))
