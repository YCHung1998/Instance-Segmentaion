import os
import sys
import yaml
# sys.path.append(os.path.join('detectron2-windows'))

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.build import build_detection_train_loader
from detectron2 import model_zoo

from source.trainer import Trainer
from Init_config import init_config
from Data_collation import get_nuclei_dicts


# Step : 
'''
1. take out data json file
2. register the name 
3. training 
4. save the config.yaml "Important"
'''
# Reference : how to registration the dataset
# https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html

if __name__=='__main__':
    mode='training'
    train_data = os.path.join('dataset', 'train.json') 
    DatasetCatalog.register("Nuclei_train", lambda : get_nuclei_dicts(train_data))
    cfg = init_config(mode='training')
    if mode == 'training':
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(cfg.OUTPUT_DIR, 'config.yaml'), 'w') as fp:
            yaml.dump(cfg, fp, default_flow_style=False)
            
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
