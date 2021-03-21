import cv2
import numpy as np
import utils.xml_utils as xml_utils
import utils.resize as resize
import os
from copy import deepcopy
import base64


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)


def rotate_box(corners, angle, cx, cy, h, w):
    """Rotate the bounding box.


    Parameters
    ----------

    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    angle : float
        angle by which the image is to be rotated

    cx : int
        x coordinate of the center of image (about which the box will be rotated)

    cy : int
        y coordinate of the center of image (about which the box will be rotated)

    h : int
        height of the image

    w : int
        width of the image

    Returns
    -------

    numpy.ndarray
        Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    """

    corners = corners.reshape(-1, 2)
    corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    # cos = np.abs(M[0, 0])
    # sin = np.abs(M[0, 1])
    #
    # nW = int((h * sin) + (w * cos))
    # nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (w / 2) - cx
    M[1, 2] += (h / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M, corners.T).T

    calculated = calculated.reshape(-1, 8)

    return calculated


def get_enclosing_box(corners, width, height):
    """Get an enclosing box for ratated corners of a bounding box

    Parameters
    ----------

    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    Returns
    -------

    numpy.ndarray
        Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    """

    x_ = corners[:, [0, 2, 4, 6]]
    y_ = corners[:, [1, 3, 5, 7]]

    xmin = np.min(x_, 1).reshape(-1, 1)
    ymin = np.min(y_, 1).reshape(-1, 1)
    xmax = np.max(x_, 1).reshape(-1, 1)
    ymax = np.max(y_, 1).reshape(-1, 1)

    final = np.hstack((xmin, ymin, xmax, ymax, corners[:, 8:])).reshape(-1)

    for i in range(0, 4):
        if (i == 0 or i == 2) and final[i] >= width:
            final[i] = width - 1
            continue
        if (i == 0 or i == 2) and final[i] <= 0:
            final[i] = 1
            continue
        if (i == 1 or i == 3) and final[i] >= height:
            final[i] = height - 1
            continue

        if (i == 1 or i == 3) and final[i] <= 0:
            final[i] = 1
            continue

    return final


def _rotate_save(img_path, out_path, angle):

    img = cv2.imread(img_path)
    img_rot = rotate_image(img, angle)
    cv2.imwrite(out_path, img_rot)


def _resize_save(img_path, out_path, width, height):
    img = cv2.imread(img_path)
    resized_image = cv2.resize(img, (width, height))
    cv2.imwrite(out_path, resized_image)


def run_aug(dataset_path, angles, ops, new_size=None):
    cwd = dataset_path

    angles.append(0)

    for folder in ['train', 'test']:
        labels_path = cwd + '/' + folder + "/labels/"
        labels = xml_utils.get_files_in_dir(labels_path)
        imgs_path = cwd + '/' + folder + "/imgs/"

        out_path = cwd + '/' + folder + "/out_imgs/"
        labels_out = cwd + '/' + folder + "/out_labels/"

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        if not os.path.exists(labels_out):
            os.makedirs(labels_out)

        xml_utils.clean_aug(imgs_path)
        xml_utils.clean_aug(labels_path)

        xml_utils.clean_aug(out_path)
        xml_utils.clean_aug(labels_out)

        for label in labels:
            xml_tree = xml_utils.parse_xml(label)

            xml_root = xml_tree.getroot()

            img_path = imgs_path + os.path.splitext(os.path.basename(label))[0] + ".jpg"
            print(img_path)
            size = xml_root.find('size')
            xml_root_copy = deepcopy(xml_root)

            w_orig = int(size.find('width').text)
            h_orig = int(size.find('height').text)

            if new_size is not None:
                size.find('width').text = str(new_size[0])
                size.find('height').text = str(new_size[1])
                _resize_save(img_path, out_path + os.path.splitext(os.path.basename(label))[0] + ".jpg", new_size[0], new_size[1])
            else:
                _resize_save(img_path, out_path + os.path.splitext(os.path.basename(label))[0] + ".jpg", w_orig, h_orig)
                # copy labels to out

            # xml_tree.write(labels_out + os.path.splitext(os.path.basename(label))[0] + ".xml")

            w = int(size.find('width').text)
            h = int(size.find('height').text)

            xml_tree_0_deg = ""

            for a in angles:

                # apply image rotation and save
                if a != 0:
                    _rotate_save(out_path + os.path.splitext(os.path.basename(label))[0] + ".jpg", out_path + os.path.splitext(os.path.basename(label))[0] + "_%d_aug.jpg" % a, -int(a))
                # apply bbox rotation

                boxes = []
                for obj in xml_root_copy.iter('object'):
                    xmlbox = obj.find('bndbox')

                    b = [float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                         float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text)]
                    if new_size is not None:
                        b = resize.resize_bbox(b, [w_orig, h_orig], new_size)

                    xmin = b[0]
                    xmax = b[1]
                    ymin = b[2]
                    ymax = b[3]

                    if a != 0:
                        corners = np.asarray([[xmin, ymin, xmax, ymin, xmin, ymax, xmax, ymax]])
                        calculated = rotate_box(corners, -int(a), int(w / 2), int(h / 2), w, h)

                        new_bboxes = get_enclosing_box(calculated, w, h)
                        boxes.append(new_bboxes)
                    else:
                        boxes.append([b[0], b[2], b[1], b[3]])

                for i, obj in enumerate(xml_root.iter('object')):
                    new_bboxes = boxes[i]
                    xmlbox = obj.find('bndbox')
                    xmlbox.find('xmin').text = str(int(new_bboxes[0]))
                    xmlbox.find('ymin').text = str(int(new_bboxes[1]))
                    xmlbox.find('xmax').text = str(int(new_bboxes[2]))
                    xmlbox.find('ymax').text = str(int(new_bboxes[3]))

                    # save changes to new .xml
                if a != 0:
                    xml_tree.write(labels_out + os.path.splitext(os.path.basename(label))[0] + "_%d_aug.xml" % a)
                else:

                    xml_tree.write(labels_out + os.path.splitext(os.path.basename(label))[0] + ".xml")

                    img = cv2.imread(out_path + os.path.splitext(os.path.basename(label))[0] + ".jpg")
                    for op in ops:
                        if int(op["val"]) != 0:
                            # laziness mode on, we DO NOT care about security here
                            out = eval(op["op"])(img, int(op["val"]))
                            cv2.imwrite(out_path + os.path.splitext(os.path.basename(label))[0] + "_%s_%s_aug.jpg" % (op["op"].split("_")[1], op["val"]), out)
                        xml_tree.write(labels_out + os.path.splitext(os.path.basename(label))[0] + "_%s_%s_aug.xml" % (op["op"].split("_")[1], op["val"]))


# https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
def apply_contrast(input_img, contrast=0):

    buf = input_img.copy()
    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def apply_brightness(input_img, brightness=0):
    buf = input_img.copy()
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(buf, alpha_b, buf, 0, gamma_b)

    return buf


def apply_saturation(input_img, saturation):
    imghsv = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV).astype("float32")
    (h, s, v) = cv2.split(imghsv)
    s = s + saturation

    s = np.clip(s, 0, 255)
    imghsv = cv2.merge([h, s, v])
    return cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    # hsv = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
    # greenMask = cv2.inRange(hsv, (26, 10, 30), (97, 100, 255))
    #
    # hsv[:, :, 1] = greenMask
    #
    # return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def apply_hue(input_img, hue):
    imghsv = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV).astype("float32")
    (h, s, v) = cv2.split(imghsv)
    h = h + hue

    h = np.clip(h, 0, 255)
    imghsv = cv2.merge([h, s, v])
    return cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2BGR)


def apply_red(input_img, red):

    img = input_img.copy()
    img[:, :, 0] = red
    return img


def apply_blue(input_img, blue):

    img = input_img.copy()
    img[:, :, 2] = blue
    return img


def apply_green(input_img, green):

    img = input_img.copy()
    img[:, :, 1] = green
    return img


# https://stackoverflow.com/questions/40928205/python-opencv-image-to-byte-string-for-json-transfer
def preview_img(img_path, ops):
    img = cv2.imread(img_path)

    results = []
    for op in ops:
        if int(op["val"]) != 0:
            # laziness mode on, we DO NOT care about security here
            out = eval(op["op"])(img, int(op["val"]))

            retval, buffer = cv2.imencode('.jpg', out)
            jpg_as_text = base64.b64encode(buffer)
            res = dict()
            res["name"] = op["op"].split("_")[1] + "_" + op["val"]
            res["val"] = jpg_as_text.decode('utf-8') # get as string
            results.append(res)

    return results
