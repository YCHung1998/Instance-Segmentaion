import copy
import torch
import numpy as np
import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils

'''
Implement method:
    RandomCrop
    ResizeShortestEdge
    RandomFlip
'''

def Mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    auginput = T.StandardAugInput(image)
    transforms = auginput.apply_augmentations([
             T.RandomCrop("relative", (0.5, 0.5)),
             # size//32 \in int
             T.ResizeShortestEdge(
                 short_edge_length=608,
                 max_size=800,
                 sample_style='choice'),
             T.RandomFlip(prob=0.5, horizontal=True, vertical=False)
         ])

    image = auginput.image
    image_shape = image.shape[:2]
    image = np.ascontiguousarray(image.transpose(2, 0, 1))
    dataset_dict["image"] = torch.as_tensor(image)

    annos = [
        utils.transform_instance_annotations(
            annotation, transforms, image_shape)
        for annotation in dataset_dict.pop("annotations")
    ]

    instances = utils.annotations_to_instances(
        annos, image_shape, mask_format="bitmask"
    )

    instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
    dataset_dict["instances"] = utils.filter_empty_instances(instances)

    return dataset_dict