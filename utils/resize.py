from copy import deepcopy


# bbox = (xmin, xmax, ymin, ymax)
# in_size = [w, h]
def resize_bbox(bbox, in_size, out_size):
    bboxx = deepcopy(bbox)
    x_scale = float(out_size[0]) / in_size[0]
    y_scale = float(out_size[1]) / in_size[1]

    # xmin, ymin
    bboxx[0] = x_scale * bboxx[0]
    bboxx[2] = y_scale * bbox[2]

    # xmax, ymax
    bboxx[1] = x_scale * bboxx[1]
    bboxx[3] = y_scale * bboxx[3]

    return bboxx


def convert_yolo_coordinates_to_voc(x_c_n, y_c_n, width_n, height_n, img_width, img_height):
    # remove normalization given the size of the image
    x_c = float(x_c_n) * img_width
    y_c = float(y_c_n) * img_height
    width = float(width_n) * img_width
    height = float(height_n) * img_height
    # compute half width and half height
    half_width = width / 2
    half_height = height / 2
    # compute left, top, right, bottom
    # in the official VOC challenge the top-left pixel in the image has coordinates (1;1)
    left = int(x_c - half_width)
    top = int(y_c - half_height)
    right = int(x_c + half_width)
    bottom = int(y_c + half_height)
    return left, top, right, bottom
