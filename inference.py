import os
import json
import argparse
import numpy as np
from detectron2.config import get_cfg
from pycocotools import mask 
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import detection_utils

# def RLE_encoder(Object):
#     # Translate the mask image into the string of codes
#     rle_encode = mask.encode(Object)
#     rle_encode["counts"] = rle_encode["counts"].decode()
#     return rle_encode


def Inference(cfg, args):

    # load test image json
    with open(args.dataset, 'r') as fp:
        test_img_ids = json.load(fp)

    # Set the model
    model = DefaultPredictor(cfg)

    submission_list = []
    # Evaluation
    for idx, test_image in enumerate(test_img_ids):
        # set the image format
        image_path = os.path.join('dataset', 'test', test_image['file_name'])
        image = detection_utils.read_image(image_path, format='BGR')  # cv2 format

        # Model predict
        pred = model(image)['instances'].to('cpu').get_fields()
        test_image_id = test_image['id']
        boxes = pred['pred_boxes'].tensor.numpy()
        scores = pred['scores'].numpy()
        masks = pred['pred_masks'].numpy()

        # Collect the information which we needed
        for box, score, msk in zip(boxes, scores, masks):
            box = box.tolist()
            rle_encode = mask.encode(np.asfortranarray(msk))
            rle_encode["counts"] = rle_encode["counts"].decode()
            seg = {
                'image_id': test_image_id,
                'box': box,
                'score': float(score),
                'category_id': 1,
                'segmentation': rle_encode
            }
            submission_list.append(seg)

        # Show the current proccess in evaluation 
        print(f'Done : {idx+1}//{len(test_img_ids)}.')

    save_root = os.path.join(
        'outputs', args.outputs,
        f'submission_{args.weight[:-4]}')

    save_path = os.path.join(save_root, 'answer.json')
    os.makedirs(save_root, exist_ok=True)

    with open(save_path, 'w') as fp:
        json.dump(submission_list, fp, indent=4)

    print(f'The inference step is finished!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs', type=str,
                        default='In_Seg-12-14-2243', help='path of output')
    parser.add_argument('--weight', type=str,
                        default='model_2199.pth', help='name of weight')
    parser.add_argument('--dataset', type=str,
                        default='dataset/test_img_ids.json',
                        help='path of dataset')

    # set the model caught from designated path
    args = parser.parse_args()
    cfg = get_cfg()
    cfg.merge_from_file(f'outputs/{args.outputs}/config.yaml')
    cfg.MODEL.WEIGHTS = os.path.join('outputs',
                                     args.outputs, args.weight)
    # cfg.TEST.DETECTIONS_PER_IMAGE = 500 
    # 700 , 800 is same as 500 in this project 
    # maybe 500 is much suitible (after observe testing image)
    cfg.MODEL.DEVICE = 'cuda'

    # load the testing data
    Inference(cfg, args)

# Example :  python inference.py --weight model_0001199.pth --outputs In_Seg-12-14-1944