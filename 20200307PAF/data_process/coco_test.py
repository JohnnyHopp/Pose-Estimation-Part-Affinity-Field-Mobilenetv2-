import os
import json
import cv2
import torch.utils.data as data
from pycocotools.coco import COCO
import numpy as np

from .coco_test_process_utils import clean_annot
from .process_utils import resize_hm_paf, normalize


class CocoTestDataSet(data.Dataset):
    def __init__(self, data_path, opt, split='test'):
        imageid = {"images":[]}
        for file in sorted(os.listdir(os.path.join(data_path, split))):
            idnum = str(file)
            file_location = '/'.join((data_path, split,file))
            imageid['images'].append({'id':idnum, 'location':file_location})
              
        with open(os.path.join(data_path, 'annotations/person_keypoints_{}.json'.format(split)),'w') as outfile:
            json.dump(imageid, outfile)
        self.coco = COCO(
            os.path.join(data_path, 'annotations/person_keypoints_{}.json'.format(split)))
        self.split = split
        self.data_path = data_path

        # load annotations that meet specific standards
        self.indices = clean_annot(self.coco, data_path, split)
        self.img_dir = os.path.join(data_path, split)
        self.opt = opt
        print('Loaded {} images for {}'.format(len(self.indices), split))

    def get_item_raw(self, index, to_resize = True):
        index = self.indices[index]
        image = self.coco.loadImgs([index])[0]
        img_path = image['location']
        img = self.load_image(img_path)
        
        return img

    def __getitem__(self, index):
        img, heat_map, paf, ignore_mask, _ = self.get_item_raw(index)
        img = normalize(img)
        heat_map, paf, ignore_mask = resize_hm_paf(heat_map, paf, ignore_mask, self.opt.hmSize)
        return img, heat_map, paf, ignore_mask, index

    def load_image(self, img_path):
        img = cv2.imread(img_path)
        img = img.astype('float32') / 255.
        return img

    def get_imgs_multiscale(self, index, scales, flip = False):
        img = self.get_item_raw(index, False)
        imgs = []
        for scale in scales:
            width, height = img.shape[1], img.shape[0]
            new_width, new_height = int(scale* width), int(scale*height)
            scaled_img = cv2.resize(img.copy(), (new_width, new_height))
            flip_img = cv2.flip(scaled_img, 1)
            scaled_img = normalize(scaled_img)
            imgs.append(scaled_img)
            if flip:
                imgs.append(normalize(flip_img))
        return imgs

    def __len__(self):
        return len(self.indices)
