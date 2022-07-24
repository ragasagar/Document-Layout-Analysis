#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel


from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import log_every_n_seconds, setup_logger
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetMapper,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
        COCOEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
import numpy as np
logger = logging.getLogger("detectron2")

def do_evaluate(cfg, model):
    print("Evaluating")
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = COCOEvaluator(dataset_name, cfg, True, "inference")
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        print(results)
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results

def do_loss(cfg, model):
  losses = []
  data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0],DatasetMapper(cfg,True))
  for idx, inputs in enumerate(data_loader):
    metrics_dict = model(inputs)
    metrics_dict = {
        k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
        for k, v in metrics_dict.items()
    }
    total_losses_reduced = sum(loss for loss in metrics_dict.values())
    losses.append(total_losses_reduced)
  mean_loss = np.mean(losses)
  return mean_loss



def do_train(cfg, model, trainiteration,resume=False):
    check_loss=2
    loss_floor=0.2
    loss_ceiling=0.4
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    data_loader = build_detection_train_loader(cfg)
    loss_list = []
    eval_list = []
    total_loss_list = []
    val_loss_list = []

 
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            loss_list.append(loss_dict_reduced)
            total_loss_list.append(losses_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                results = do_evaluate(cfg, model)
                eval_list.append(results["bbox"])
              #checkpointer.save("model_"+str(train_round)+"_"+str(iteration))
                val_loss = do_loss(cfg, model)
                val_loss_list.append(val_loss)


                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()

            periodic_checkpointer.step(iteration)
           
            if len(val_loss_list)>=check_loss:
              stop_flag = all(l >= loss_floor and l < loss_ceiling for l in val_loss_list[-int(check_loss):]) 
              if stop_flag: 
                #checkpointer.save("model_final_"+str(train_round))
                print("Stop Condition for training has been met.")
                break 

    
    checkpointer.save("model_final_"+str(trainiteration))
            


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg



#config_file = "/content/config/DLA_mask_rcnn_X_101_32x8d_FPN_3x.yaml"
  #model_file = "/content/models/model_final_trimmed.pth"
#cfg._open_cfg(config_file)

def publaynet(cfg):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    do_train(cfg, model, 1, resume=False) #never resume. Weights are taken from cfg yml

    model_prev = "output/model_final_1.pth"
    cfg.MODEL.WEIGHTS = model_prev

    print("Training Iteration 2 with Freeze layers at 3 CNN")
    cfg.MODEL.BACKBONE.FREEZE_AT = 3
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    do_train(cfg, model, 2, resume=False) #never resume. Weights are taken from cfg yml

    print("Training Iteration 3 with no freeze layers and very small LR")

    model_prev = "output/model_final_2.pth"
    cfg.MODEL.WEIGHTS = model_prev
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.SOLVER.BASE_LR = 0.00001    
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    do_train(cfg, model, 3, resume=False) #never resume. Weights are taken from cfg yml
