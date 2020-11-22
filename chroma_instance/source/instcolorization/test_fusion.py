# import os
from os.path import join
# import time
import machine
from instcolorization.options.train_options import TrainOptions, TestOptions
from instcolorization.models import create_model
# from .util.visualizer import Visualizer

from os.path import join, isfile, isdir
from os import listdir
# import os
from argparse import ArgumentParser

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

import torch

import torch
# import torchvision
# import torchvision.transforms as transforms
from tqdm import trange, tqdm

from instcolorization.fusion_dataset import Fusion_Testing_Dataset
from instcolorization.util import util
import os
if machine.is_colab:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# import numpy as np
import multiprocessing
multiprocessing.set_start_method('spawn', True)

torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    # Bbox
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    input_dir = "../../dataset/test"
    image_list = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
    output_npz_dir = "{0}_bbox".format(input_dir)
    if os.path.isdir(output_npz_dir) is False:
        print('Create path: {0}'.format(output_npz_dir))
        os.makedirs(output_npz_dir)

    for image_path in image_list:
        img = cv2.imread(join(input_dir, image_path))
        lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        l_stack = np.stack([l_channel, l_channel, l_channel], axis=2)
        outputs = predictor(l_stack)
        save_path = join(output_npz_dir, image_path.split('.')[0])
        pred_bbox = outputs["instances"].pred_boxes.to(torch.device('cpu')).tensor.numpy()
        pred_scores = outputs["instances"].scores.cpu().data.numpy()
        np.savez(save_path, bbox=pred_bbox, scores=pred_scores)



    # Prediction
    opt = TestOptions().parse()
    save_img_path = opt.results_img_dir
    if os.path.isdir(save_img_path) is False:
        print('Create path: {0}'.format(save_img_path))
        os.makedirs(save_img_path)
    opt.batch_size = 1
    dataset = Fusion_Testing_Dataset(opt)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2)

    dataset_size = len(dataset)
    print('#Testing images = %d' % dataset_size)

    model = create_model(opt)
    # model.setup_to_test('coco_finetuned_mask_256')
    model.setup_to_test('coco_finetuned_mask_256_ffs')

    count_empty = 0
    for data_raw in tqdm(dataset_loader, dynamic_ncols=True):
        # if os.path.isfile(join(save_img_path, data_raw['file_id'][0] + '.png')) is True:
        #     continue
        data_raw['full_img'][0] = data_raw['full_img'][0].cuda()
        if data_raw['empty_box'][0] == 0:
            data_raw['cropped_img'][0] = data_raw['cropped_img'][0].cuda()
            box_info = data_raw['box_info'][0]
            box_info_2x = data_raw['box_info_2x'][0]
            box_info_4x = data_raw['box_info_4x'][0]
            box_info_8x = data_raw['box_info_8x'][0]
            cropped_data = util.get_colorization_data(data_raw['cropped_img'], opt, ab_thresh=0, p=opt.sample_p)
            full_img_data = util.get_colorization_data(data_raw['full_img'], opt, ab_thresh=0, p=opt.sample_p)
            model.set_input(cropped_data)
            model.set_fusion_input(full_img_data, [box_info, box_info_2x, box_info_4x, box_info_8x])
            model.forward()
        else:
            count_empty += 1
            full_img_data = util.get_colorization_data(data_raw['full_img'], opt, ab_thresh=0, p=opt.sample_p)
            model.set_forward_without_box(full_img_data)
        model.save_current_imgs(join(save_img_path, data_raw['file_id'][0] + '.png'))
    print('{0} images without bounding boxes'.format(count_empty))
