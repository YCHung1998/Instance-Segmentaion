# now
import os
import sys
# sys.path.append(os.path.join('detectron2-windows'))

from datetime import datetime
from detectron2 import model_zoo
from detectron2.config import get_cfg

'''
# initial config setting
# reference : https://github.com/facebookresearch/detectron2
#                    /blob/main/detectron2/config/defaults.py
# reference : https://detectron2.readthedocs.io/en/latest/modules/config.html
The most important is load the .yaml file and merge into the config.

1. model and model weight

Then we started adjusting our dataset, model weights
and save_path ... and so on.

2-1. output_loc #611 = [OUTPUT_DIR]
2-2. dataset #91  = [TRAIN, TEST]
2-3. dataloader #111 = [NUM_WORKERS]
===============================
============ Train ============
===============================
3. model [
    anchor #169
    ANCHOR_GENERATOR.SIZE,
    roi #248
    ROI_HEADS.NUM_CLASSES,
    ROI_MASK_HEAD.POOLER_RESOLUTION
    ]

warm up with based lr and set the learning scheduler
4. Solver [
    IMS_PER_BATCH
    LR_SCHEDULER_NAME
    MAX_ITER
    BASE_LR = 0.01
    WARMUP_FACTOR
    WARMUP_ITERS
    GAMMA
    STEPS
    CHECKPOINT_PERIOD
    ]

In the testing dataset or inference way
5. Test [ maybe it is more suitible to write in the inference ]
    [ DETECTIONS_PER_IMAGE
    EVAL_PERIOD
    MIN_SIZE_TEST
    MAX_SIZE_TEST
    MIN_SIZES
    ]
'''


def init_config(mode='training'):
    # Instance Segmentation save name
    output_name = datetime.today().strftime('In_Seg-%m-%d-%H%M')

    # Model Select
    # yml_path = 'COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml'
    yml_path = 'COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml'

    cfg = get_cfg()

    # Initail Setting
    cfg.merge_from_file(model_zoo.get_config_file(yml_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yml_path)  # 37

    # saving path and assign dataset
    cfg.OUTPUT_DIR = os.path.join('outputs', output_name)  # save the record
    cfg.DATASETS.TRAIN = ("Nuclei_train",)  # notice
    cfg.DATASETS.TEST = ()                  # none
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.DEVICE = 'cuda:2'

    # cfg.MODEL.BACKBONE.FREEZE_AT = 2
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 28

    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.MAX_ITER = 100 * 24  # 100 epochs
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
    cfg.SOLVER.BASE_LR = 1e-2
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 10000
    cfg.SOLVER.WARMUP_ITERS = 3 * 24  # 3 epochs
    cfg.SOLVER.GAMMA = 0.1
    # The iteration number to decrease learning rate by GAMMA.
    cfg.SOLVER.STEPS = (20*24, 50*24, 80*24, 90*24)
    # 480, 1200, 1920, 2160, 4800, 6000
    cfg.SOLVER.CHECKPOINT_PERIOD = 200
    # Save a checkpoint in the OUTPUT_DIR
    # after every this number of iterations

    # For inference

    # cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 12000
    # cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 2000
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

    cfg.INPUT.MIN_SIZE_TEST = 1000
    cfg.INPUT.MAX_SIZE_TEST = 1333
    # Enlarge the picture to catch the key point

    cfg.TEST.EVAL_PERIOD = 0

    cfg.TEST.DETECTIONS_PER_IMAGE = 500
    # the maximum number of the mask
    # According to this project
    # the

    cfg.TEST.AUG["ENABLED"] = True
    cfg.TEST.AUG.MIN_SIZES = (1500, 1600, 1700)
    # resize to check the objection

    cfg.SOLVER.CHECKPOINT_PERIOD = 200
    # Save a checkpoint in the OUTPUT_DIR
    # after every this number of iterations

    return cfg
