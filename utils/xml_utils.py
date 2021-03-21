import glob
import os
import xml.etree.ElementTree as ET
import pandas as pd
import utils.resize as resize


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):

            value = (root.find('filename').text,  # filename
                     int(root.find('size')[0].text),  # width
                     int(root.find('size')[1].text),  # height
                     member[0].text,  # class
                     int(member[4][0].text),  # xmin
                     int(member[4][1].text),  # ymin
                     int(member[4][2].text),  # xmax
                     int(member[4][3].text)  # ymax
                     )

            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def save_csv(csv_file, out_path):
    csv_file.to_csv(out_path, index=None)


def get_files_in_dir(dir_path):
    image_list = []
    for filename in glob.glob(dir_path + '/*.xml'):
        image_list.append(filename)

    return image_list


def clean_aug(dir_path):

    for filename in glob.glob(dir_path + '*'):
        if "_aug" in filename:
            os.remove(filename)
            continue


def clean(dir_path):

    for filename in glob.glob(dir_path + '*'):
        os.remove(filename)


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return x, y, w, h


def parse_xml(path):
    in_file = open(path)
    return ET.parse(in_file)


# old
def convert_annotation(dir_path, output_path, image_path, classes, resize_size=None):
    basename = os.path.basename(image_path)
    basename_no_ext = os.path.splitext(basename)[0]

    in_file = open(dir_path + '/' + basename_no_ext + '.xml')
    out_file = open(output_path + basename_no_ext + '.txt', 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    target_w = int(size.find('width').text)
    target_h = int(size.find('height').text)

    if resize_size is not None:
        target_w, target_h = resize_size[0], resize_size[1]

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = [float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text)]

        if resize_size is not None:
            b = resize.resize_bbox(b, [w, h], resize_size)

        bb = convert((target_w, target_h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def convert_annotation2(dir_path, output_path, image_path, classes):
    basename = os.path.basename(image_path)
    basename_no_ext = os.path.splitext(basename)[0]

    in_file = open(dir_path + '/' + basename_no_ext + '.xml')
    out_file = open(output_path + basename_no_ext + '.txt', 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')

    target_w = int(size.find('width').text)
    target_h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = [float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text)]

        bb = convert((target_w, target_h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def main(dataset_path, classes):

    cwd = dataset_path

    yolo_dataset = cwd + '/yolo/'

    for folder in ['train', 'test']:
        full_dir_path = cwd + '/' + folder + "/out_labels/"
        image_paths = get_files_in_dir(full_dir_path)
        output_path = yolo_dataset + folder + "/labels/"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        clean(output_path)
        for image_path in image_paths:
            convert_annotation2(full_dir_path, output_path, image_path, classes)

