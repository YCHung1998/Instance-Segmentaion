# If you are the linux user, you can ignore the following comment.
import os
'''
# If you are the windows user, please excuted in detectron2-windows
# That is to say uncomment this region.
import os
import sys
sys.path.append(os.path.join('detectron2-windows'))
'''
import cv2
import json
import numpy as np
from PIL import Image

from pycocotools import mask
from detectron2.structures import BoxMode

def generate_json_format(dataset_loc=os.path.join('dataset', 'train'),
                         save_loc=os.path.join('dataset', 'train.json')):
    format_list = ['file_name', 'image_id', 'height', 'width']
    object_list = ['category_id', 'bbox_mode', 'bbox'] 
    # Notice : segmentation written bellow

    Namelist = os.listdir(dataset_loc)
    dataset_dict = []

    for idx, image_name in enumerate(Namelist):
        record={}
        filename = os.path.join(dataset_loc,
                                image_name, 'images', image_name+'.png')
        img = Image.open(filename)
        height = img.size[0]
        width = img.size[1]
        image_id = idx
        record = dict(zip(format_list, [filename, image_id, height, width]))

        # ==============
        # Annotations 
        # ==============
        MaskList = os.listdir(os.path.join(dataset_loc, image_name,'masks'))
        annotations = [] 
        for msk_name in MaskList:
            if not msk_name.endswith('.png'):
                continue

            Object = {}
            # read the mask and convert to binary ndarray
            msk = cv2.imread(os.path.join(dataset_loc,
                                          image_name,
                                          'masks',
                                          msk_name))
            msk = np.sum(msk, axis=2) > 0
            msk = np.asfortranarray(msk)
            # ==============
            # Setting : category_id, segmentation, bbox_mode, bbox
            # category_id is 0 because of only one category
            # need be predict in this project.
            # ==============
            category_id = 0
            segmentation = mask.encode(msk)
            segmentation['counts'] = segmentation['counts'].decode() 
            bbox_mode = BoxMode.XYWH_ABS 
            bbox = mask.toBbox(segmentation).tolist()
            
            # build the dictionary
            Object = dict(zip(object_list, [category_id, bbox_mode, bbox]))
            Object['segmentation'] = segmentation
            annotations.append(Object)
        record['annotations'] = annotations
        dataset_dict.append(record)
        print(f"Create id:{image_id}, name:{image_name} !")

    # write in json
    JSON = json.dumps(dataset_dict, indent = 4)
    with open(save_loc, 'w') as f:
        f.write(JSON)
    return None
                
def get_nuclei_dicts(json_loc=os.path.join('dataset', 'train.json')):
    '''
    This functoion is for read the json file and plug into DatasetCatalog to register it
    '''
    with open(json_loc, 'r') as f:
        dataset_dicts = json.load(f)

    # set each annotation["bbox_mode"] to BoxMode.XYWH_ABS
    for data in dataset_dicts:
        for anno in data["annotations"]:
            if anno["bbox_mode"]!=BoxMode.XYWH_ABS:
                anno["bbox_mode"] = BoxMode.XYWH_ABS

    return dataset_dicts

if __name__=='__main__':
    # Create the train json file
    generate_json_format()
