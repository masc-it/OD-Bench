# Author: mascIT

import base64
import glob
import os
import cv2
import jinja2
import xml.etree.ElementTree as ET


def get_images(dir_path):
    image_list = []
    for filename in glob.glob(dir_path + '/*.jpg'):
        image_list.append(filename)

    return image_list


def get_image_base64(path, index):
    images = get_images(path)

    if index >= len(images):
        return {"img": ""}

    img = cv2.imread(images[index])
    retval, buffer = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buffer).decode("utf-8")
    return {"img": jpg_as_text, "w": img.shape[1], "h": img.shape[0], "name": images[index]}


def get_labels(dir_path):
    image_list = []
    for filename in glob.glob(dir_path + '/*.xml'):
        image_list.append(filename)

    return image_list


def load_xml(xml_path, img_name):

    tree = ET.parse(os.path.join(xml_path, img_name + ".xml"))
    root = tree.getroot()

    bboxes = []

    for member in root.findall('object'):

        bbox = {"xmin": int(member[4][0].text),
                "ymin": int(member[4][1].text),
                "xmax": int(member[4][2].text),
                "ymax": int(member[4][3].text),
                "class": member[0].text}

        bboxes.append(bbox)

    return bboxes


def save_xml(w, h, boxes, name, path):

    # w = 200
    # h = 300
    #
    # boxes = [{"x_min": 100, "y_min": 100, "x_max": 120, "y_max": 130},
    #          {"x_min": 120, "y_min": 100, "x_max": 120, "y_max": 130}]

    templateLoader = jinja2.FileSystemLoader(searchpath="./")
    templateEnv = jinja2.Environment(loader=templateLoader)
    TEMPLATE_FILE = "templ.xml"
    template = templateEnv.get_template(TEMPLATE_FILE)
    out = template.render(width=w, height=h, boxes=boxes, name=name)
    with open("%s.xml" % path, "w") as fh:
        fh.write(out)
        fh.close()
