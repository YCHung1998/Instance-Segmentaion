from detectron2.engine.defaults import DefaultTrainer
from detectron2.data.build import build_detection_train_loader
# source/transform
from source.transform import Mapper

# https://detectron2.readthedocs.io/en/latest/tutorials/write-models.html

class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, Mapper)
