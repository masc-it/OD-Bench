# Author: mascIT

import cv2
import numpy as np
import utils.xml_utils as xml_utils
import utils.resize as resize
import os
from copy import deepcopy
import base64
import imutils


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)


def rotate_image2(image, angle):
    return imutils.rotate_bound(image, angle)


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

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    #
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
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

    return final


def _rotate_save(img_path, out_path, angle, size):
    img = cv2.imread(img_path)
    img_rot = rotate_image2(img, angle)
    rot_h, rot_w = img_rot.shape[:2]
    img_rot = cv2.resize(img_rot, size)
    cv2.imwrite(out_path, img_rot)
    return rot_h, rot_w


def _get_rotated_img(img, angle, size):

    img_rot = rotate_image2(img, angle)
    img_rot = cv2.resize(img_rot, size)

    return img_rot


def _resize_save(img_path, out_path, width, height):
    img = cv2.imread(img_path)
    resized_image = cv2.resize(img, (width, height))
    cv2.imwrite(out_path, resized_image)


def _downscale_with_padding(img_path, out_path, down_width, down_height):
    img = cv2.imread(img_path)
    ht, wd, cc = img.shape

    ww = down_width
    hh = down_height
    averageR = img.mean(axis=0).mean(axis=0)

    color = (averageR[0], averageR[1], averageR[2])
    result = np.full((ht, wd, cc), color, dtype=np.uint8)

    # compute center offset
    offset_x = abs(wd - ww) // 2
    offset_y = abs(ht - hh) // 2

    # copy img image into center of result image
    result[offset_y:ht - offset_y, offset_x:wd - offset_x] = cv2.resize(img, (hh, ww))

    cv2.imwrite(out_path, result)
    return offset_x, offset_y


def _get_downscale_with_padding(img, down_width, down_height, resize):
    # img = cv2.imread(img_path)
    ht, wd, cc = img.shape
    # minv = min([ht, wd])
    img = cv2.resize(img, resize)
    ht, wd, cc = img.shape
    ww = down_width
    hh = down_height
    averageR = img.mean(axis=0).mean(axis=0)

    color = (averageR[0], averageR[1], averageR[2])
    result = np.full((ht, wd, cc), color, dtype=np.uint8)

    # compute center offset
    offset_x = abs(wd - ww) // 2
    offset_y = abs(ht - hh) // 2

    # copy img image into center of result image
    result[offset_y:ht - offset_y, offset_x:wd - offset_x] = cv2.resize(img, (hh, ww))

    return result


def downscale_with_padding2(xml_tree, img_path, out_path, labels_out, boxes, new_size, down_size):
    xml_tree = deepcopy(xml_tree)
    xml_root = xml_tree.getroot()
    offset_x, offset_y = _downscale_with_padding(img_path, out_path, down_size[0], down_size[1])

    for i, obj in enumerate(xml_root.iter('object')):
        xmlbox = obj.find('bndbox')
        box = deepcopy(boxes[i])
        box[1], box[2] = box[2], box[1]

        b = resize.resize_bbox(box, new_size, (down_size[0], down_size[1]))
        b[0] = b[0] + offset_x
        b[1] = b[1] + offset_x

        b[2] = b[2] + offset_y
        b[3] = b[3] + offset_y

        xmlbox.find('xmin').text = str(int(b[0]))
        xmlbox.find('ymin').text = str(int(b[2]))
        xmlbox.find('xmax').text = str(int(b[1]))
        xmlbox.find('ymax').text = str(int(b[3]))

    xml_tree.write(labels_out)


def _build_path(name, ext):
    return name + ext


def run_aug(dataset_path, angles, ops, new_size=None):
    cwd = dataset_path

    angles.append(0)

    do_downscale = False
    wh = 0

    for op in ops:
        if op["op"] == "downscale":
            do_downscale = True
            # downscale with, height
            wh = op["val"].split(",")
            break

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

            img_path = _build_path(imgs_path + os.path.splitext(os.path.basename(label))[0], ".jpg")
            print(img_path)
            size = xml_root.find('size')
            xml_root_copy = deepcopy(xml_root)

            w_orig = int(size.find('width').text)
            h_orig = int(size.find('height').text)

            # output image path
            out_img_orig = _build_path(out_path + os.path.splitext(os.path.basename(label))[0], ".jpg")
            out_label_orig = _build_path(labels_out + os.path.splitext(os.path.basename(label))[0], ".xml")

            if new_size is not None:
                size.find('width').text = str(new_size[0])
                size.find('height').text = str(new_size[1])
                _resize_save(img_path, out_img_orig, new_size[0],
                             new_size[1])
            else:
                _resize_save(img_path, out_img_orig, w_orig, h_orig)

            w = int(size.find('width').text)
            h = int(size.find('height').text)

            for a in angles:

                out_img_angle = _build_path(out_path + os.path.splitext(os.path.basename(label))[0] + "_%d_aug" % a,
                                            ".jpg")
                out_label_angle = _build_path(out_path + os.path.splitext(os.path.basename(label))[0] + "_%d_aug" % a,
                                              ".xml")

                rot_h, rot_w = 0, 0
                # apply image rotation and save
                if a != 0:
                    rot_h, rot_w = _rotate_save(out_img_orig, out_img_angle, int(a), (w, h))

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
                        scale_factor_x = rot_w / w
                        scale_factor_y = rot_h / h

                        new_bboxes /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]

                    else:
                        new_bboxes = [b[0], b[2], b[1], b[3]]

                    boxes.append(new_bboxes)

                if do_downscale:
                    out_img_angle_path = out_img_angle

                    if a == 0:
                        out_img_angle_path = out_img_orig

                    downscale_with_padding2(xml_tree,
                                            out_img_angle_path,
                                            out_path + os.path.splitext(os.path.basename(label))[
                                                0] + "_%d_down_aug.jpg" % int(a),
                                            labels_out + os.path.splitext(os.path.basename(label))[
                                                0] + "_%d_down_aug.xml" % int(a),
                                            boxes,
                                            (w, h),
                                            (int(wh[0]), int(wh[1]))
                                            )

                for i, obj in enumerate(xml_root.iter('object')):
                    new_bboxes = boxes[i]
                    xmlbox = obj.find('bndbox')
                    xmlbox.find('xmin').text = str(int(new_bboxes[0]))
                    xmlbox.find('ymin').text = str(int(new_bboxes[1]))
                    xmlbox.find('xmax').text = str(int(new_bboxes[2]))
                    xmlbox.find('ymax').text = str(int(new_bboxes[3]))

                    # save changes to new .xml
                if a != 0:
                    xml_tree.write(out_label_angle)
                else:
                    # save original image labels (just resized)
                    xml_tree.write(out_label_orig)

                    img = cv2.imread(out_img_orig)

                    # color augmentation, just for orig image (not rotated)
                    for op in ops:
                        if "apply" in op["op"] and int(op["val"]) != 0:
                            # laziness mode on, we DO NOT care about security here :D
                            out = eval(op["op"])(img, int(op["val"]))

                            cv2.imwrite(out_path + os.path.splitext(os.path.basename(label))[0] + "_%s_%s_aug.jpg" % (
                                op["op"].split("_")[1], op["val"]), out)

                            xml_tree.write(
                                labels_out + os.path.splitext(os.path.basename(label))[0] + "_%s_%s_aug.xml" % (
                                    op["op"].split("_")[1], op["val"]))


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


def apply_hist(input_img, dummy):
    ycrcb_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2YCrCb)

    # equalize the histogram of the Y channel
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

    # convert back to RGB color-space from YCrCb
    return cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)


def apply_gamma(image, gamma):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


# https://stackoverflow.com/questions/40928205/python-opencv-image-to-byte-string-for-json-transfer
def preview_img(img_path, ops, resize, angles):
    img = cv2.imread(img_path)
    results = []
    for op in ops:

        res = dict()
        if op["op"] != "downscale":
            # we DO NOT care about security here :D
            if "," not in op["val"]:
                out = eval(op["op"])(img, float(op["val"]))

                retval, buffer = cv2.imencode('.jpg', out)
                jpg_as_text = base64.b64encode(buffer)

                res["name"] = op["op"].split("_")[1] + "_" + op["val"]
                res["val"] = jpg_as_text.decode('utf-8')  # get as string
                results.append(res)
            else:
                values = op["val"].split(",")

                for val in values:
                    res = dict()
                    out = eval(op["op"])(img, float(val))

                    retval, buffer = cv2.imencode('.jpg', out)
                    jpg_as_text = base64.b64encode(buffer)

                    res["name"] = op["op"].split("_")[1] + "_" + val
                    res["val"] = jpg_as_text.decode('utf-8')  # get as string

                    results.append(res)

        if op["op"] == "downscale":
            wh = op["val"].split(",")
            out = _get_downscale_with_padding(img, int(wh[0]), int(wh[1]), (resize[0], resize[1]))
            retval, buffer = cv2.imencode('.jpg', out)
            jpg_as_text = base64.b64encode(buffer)

            res["name"] = "downscale_" + op["val"]
            res["val"] = jpg_as_text.decode('utf-8')  # get as string

            results.append(res)

    for a in angles:
        res = dict()
        out = _get_rotated_img(img, a, (resize[0], resize[1]))
        retval, buffer = cv2.imencode('.jpg', out)
        jpg_as_text = base64.b64encode(buffer)

        res["name"] = "rotate_%d" % a
        res["val"] = jpg_as_text.decode('utf-8')  # get as string
        results.append(res)
    return results
