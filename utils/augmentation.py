import json
import os
from copy import deepcopy

import cv2
import imutils
import numpy as np

from utils import xml_utils, resize
from utils.augment import _rotate_save, _downscale_with_padding

from utils.augment import *


class Downscale(object):

    path_suffix = "_down_aug"

    def __init__(self, target_size):
        self.target_size = target_size


class PathUtil(object):

    def __init__(self):

        self.paths = {}

    IMGS_PATH = "imgs_path"
    LABELS_PATH = "labels"

    OUT_PATH = "out_path"
    OUT_LABELS = "labels_out"

    def __getitem__(self, name):
        return self.paths[name]

    def set_path(self, name, value):
        if name not in self.paths:
            self.paths[name] = {}
        # "folder" : "path"
        self.paths[name][value[0]] = value[1]

    def get_path(self, name, folder):
        return self.paths[name][folder]

    def __setitem__(self, name, value):

        if name not in self.paths:
            self.paths[name] = {}
        # "folder" : "path"
        self.paths[name][value[0]] = value[1]

    def __str__(self):
        return json.dumps(self, default=lambda x: x.__dict__)


class ImageInfo(object):

    img_ext = ".jpg"
    label_ext = ".xml"

    def __init__(self, xml_tree, label, path_utils: PathUtil, folder, new_size):
        self.xml_tree = deepcopy(xml_tree)

        self.xml_root = self.xml_tree.getroot()

        self.img_path = path_utils[path_utils.IMGS_PATH][folder] + os.path.splitext(os.path.basename(label))[0] + self.img_ext
        # print(img_path)
        size = self.xml_root.find('size')

        self.w_orig = int(size.find('width').text)
        self.h_orig = int(size.find('height').text)

        self.w = new_size[0]
        self.h = new_size[1]

        size.find('width').text = str(self.w)
        size.find('height').text = str(self.h)

        # output image path
        self.out_img_base = path_utils[path_utils.OUT_PATH][folder] + os.path.splitext(os.path.basename(label))[0]
        self.out_img_orig = self.out_img_base + self.img_ext

        self.out_label_base = path_utils[path_utils.OUT_LABELS][folder] + os.path.splitext(os.path.basename(label))[0]
        self.out_label_orig = self.out_label_base + self.label_ext

        self.original_bboxes = self.get_original_bboxes()

        self.patched_bboxes = []

    def get_original_bboxes(self):

        original_bboxes = []
        for obj in self.xml_root.iter('object'):
            xmlbox = obj.find('bndbox')

            b = [float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text)]
            original_bboxes.append(b)
        return original_bboxes

    def get_aug_angle_path(self, angle):

        img_path = self.out_img_base + ("_%d_aug" % angle) + self.img_ext
        label_path = self.out_label_base + ("_%d_aug" % angle) + self.label_ext
        return img_path, label_path

    def get_aug_downscale_path(self, angle=0):

        if angle != 0:
            img_path = self.out_img_base + ("_%d_down_aug" % angle) + self.img_ext
            label_path = self.out_label_base + ("_%d_down_aug" % angle) + self.label_ext
        else:
            img_path = self.out_img_base + "_down_aug" + self.img_ext
            label_path = self.out_label_base + "_down_aug" + self.label_ext
        return img_path, label_path

    def get_aug_color_path(self, op, val, angle=0):

        if angle != 0:
            img_path = self.out_img_base + ("_%d_%s_%s_color_aug" % (angle, op, val)) + self.img_ext
            label_path = self.out_label_base + ("_%d_%s_%s_color_aug" % (angle, op, val)) + self.label_ext
        else:
            img_path = self.out_img_base + ("_%s_%s_color_aug" % (op, val)) + self.img_ext
            label_path = self.out_label_base + ("_%s_%s_color_aug" % (op, val)) + self.label_ext
        return img_path, label_path

    def resize_save(self):
        img = cv2.imread(self.img_path)
        resized_image = cv2.resize(img, (self.w, self.h))
        cv2.imwrite(self.out_img_orig, resized_image)

    def rotate_save(self, angle):

        if angle != 0:
            out_img_angle, _ = self.get_aug_angle_path(angle)
            rot_h, rot_w = _rotate_save(self.out_img_orig, out_img_angle, angle, (self.w, self.h))
            return rot_h, rot_w
        else:
            return self.h, self.w

    def patch_bboxes(self, angle, rot_h, rot_w):

        self.patched_bboxes = []
        for bbox in deepcopy(self.original_bboxes):
            b = resize.resize_bbox(bbox, [self.w_orig, self.h_orig], [self.w, self.h])
            xmin = b[0]
            xmax = b[1]
            ymin = b[2]
            ymax = b[3]

            if angle != 0:
                corners = np.asarray([[xmin, ymin, xmax, ymin, xmin, ymax, xmax, ymax]])
                calculated = rotate_box(corners, -angle, int(self.w / 2), int(self.h / 2), self.w, self.h)

                new_bboxes = get_enclosing_box(calculated, self.w, self.h)
                scale_factor_x = rot_w / self.w
                scale_factor_y = rot_h / self.h

                new_bboxes /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]

            else:
                new_bboxes = [b[0], b[2], b[1], b[3]]

            self.patched_bboxes.append(new_bboxes)

        for i, obj in enumerate(self.xml_root.iter('object')):
            new_bboxes = self.patched_bboxes[i]
            xmlbox = obj.find('bndbox')
            xmlbox.find('xmin').text = str(int(new_bboxes[0]))
            xmlbox.find('ymin').text = str(int(new_bboxes[1]))
            xmlbox.find('xmax').text = str(int(new_bboxes[2]))
            xmlbox.find('ymax').text = str(int(new_bboxes[3]))

        if angle != 0:
            _, out_label_angle = self.get_aug_angle_path(angle)
            self.xml_tree.write(out_label_angle)
        else:
            # save original image labels (just resized)
            self.xml_tree.write(self.out_label_orig)

    def downscale_with_padding(self, down_size, angle):

        xml_tree = deepcopy(self.xml_tree)
        out_img_angle_path = self.out_img_orig

        if angle != 0:
            out_img_angle_path, out_label_angle = self.get_aug_angle_path(angle)

        out_img_downscale_path, out_label_downscale_path = self.get_aug_downscale_path(angle)

        # save downscaled-padded image
        offset_x, offset_y = _downscale_with_padding(out_img_angle_path, out_img_downscale_path, down_size[0], down_size[1])

        for i, obj in enumerate(xml_tree.getroot().iter('object')):
            xmlbox = obj.find('bndbox')
            box = deepcopy(self.patched_bboxes[i])
            box[1], box[2] = box[2], box[1]

            # box = resize.resize_bbox(self.original_bboxes[i], (self.w_orig, self.h_orig), (self.w, self.h))
            b = resize.resize_bbox(box, (self.w, self.h), (down_size[0], down_size[1]))

            b[0] = b[0] + offset_x
            b[1] = b[1] + offset_x

            b[2] = b[2] + offset_y
            b[3] = b[3] + offset_y

            xmlbox.find('xmin').text = str(int(b[0]))
            xmlbox.find('ymin').text = str(int(b[2]))
            xmlbox.find('xmax').text = str(int(b[1]))
            xmlbox.find('ymax').text = str(int(b[3]))

        xml_tree.write(out_label_downscale_path)

    def color_aug(self, op, a):
        out_img_angle_path = self.out_img_orig

        if a != 0:
            out_img_angle_path, _ = self.get_aug_angle_path(a)

        imgg = cv2.imread(out_img_angle_path)
        if "apply" in op["op"] and int(op["val"]) != 0:
            # we DO NOT care about security here :D
            out = eval(op["op"])(imgg, int(op["val"]))

            img_out_color, label_out_color = self.get_aug_color_path(op["op"].split("_")[1], op["val"], a)

            cv2.imwrite(img_out_color, out)

            self.xml_tree.write(label_out_color)


class Augmentation(object):

    def __init__(self, dataset_path, angles, ops, new_size):
        self.dataset_path = dataset_path
        self.angles = angles
        self.ops = ops
        self.new_size = new_size

        self.angles.append(0)
        self.downscale = self._setup_downscale()

        folders = ["train", "test"]
        self._setup_folders(folders, clean_out_folders=True)

        # print(self.path_util)

        for folder in folders:
            self.do_aug(folder)

    def _setup_downscale(self):

        for op in self.ops:
            if op["op"] == "downscale":
                # downscale with, height
                wh = list( map( lambda x: int(x), op["val"].split(",")))
                return Downscale(wh)

        return None

    def _setup_folders(self, subdirs, clean_out_folders=False):

        self.path_util = PathUtil()

        for folder in subdirs:

            base_folder = self.dataset_path + '/' + folder

            self.path_util[self.path_util.LABELS_PATH] = [folder, base_folder + "/labels/"]
            self.path_util[self.path_util.IMGS_PATH] = [folder, base_folder + "/imgs/"]

            self.path_util[self.path_util.OUT_PATH] = [folder, base_folder + "/out_imgs/"]
            self.path_util[self.path_util.OUT_LABELS] = [folder, base_folder + "/out_labels/"]

            if not os.path.exists(self.path_util[self.path_util.OUT_PATH][folder]):
                os.makedirs(self.path_util[self.path_util.OUT_PATH][folder])

            if not os.path.exists(self.path_util[self.path_util.OUT_LABELS][folder]):
                os.makedirs(self.path_util[self.path_util.OUT_LABELS][folder])

            if clean_out_folders:
                xml_utils.clean_aug(self.path_util[self.path_util.IMGS_PATH][folder])
                xml_utils.clean_aug(self.path_util[self.path_util.LABELS_PATH][folder])

                xml_utils.clean_aug(self.path_util[self.path_util.OUT_PATH][folder])
                xml_utils.clean_aug(self.path_util[self.path_util.OUT_LABELS][folder])

    def do_aug(self, folder):

        labels = xml_utils.get_files_in_dir(self.path_util[self.path_util.LABELS_PATH][folder])

        for label in labels:
            xml_tree = xml_utils.parse_xml(label)

            img = ImageInfo(xml_tree, label, self.path_util, folder, self.new_size)

            # resize image and save
            img.resize_save()

            # rotate image and save patched label
            for a in self.angles:
                rot_h, rot_w = img.rotate_save(a)
                img.patch_bboxes(a, rot_h, rot_w)

                # downscale
                if self.downscale is not None:
                    img.downscale_with_padding(self.downscale.target_size, a)

                # color augmentation
                for op in self.ops:
                    img.color_aug(op, a)

