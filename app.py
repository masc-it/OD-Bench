# Author: mascIT
import os

from flask import Flask, render_template, request

import utils.xml_utils as xml_to_yolo
import utils.augment as augment
import utils.annotate_utils as annotate_utils

import json

from utils.augmentation import Augmentation

app = Flask(__name__, static_url_path='/static')


@app.route('/')
def hello_world():
    return app.send_static_file('augment.html')


@app.route('/datasetinfo', methods=['POST'])
def get_dataset_info():
    d_path = request.form['path']
    classes = request.form['classes']

    xml_to_yolo.main(d_path, classes.split(","))

    return "ok"


@app.route('/augmentation')
def augmentation():
    return app.send_static_file('augment.html')


@app.route('/annotate')
def annotate():
    return app.send_static_file('annotate.html')


@app.route('/load_img', methods=['POST'])
def load_img():
    # blob = request.get_json(force=True)

    d_path = request.form['path']
    labels_path = request.form['labels_path']
    index = request.form['index']
    img_width = request.form['w']
    img_height = request.form['h']
    name = request.form['name']
    boxes = request.form['boxes']
    boxes = json.loads(boxes)

    # take care of the image just processed
    if len(boxes) > 0:
        just_name = os.path.splitext(os.path.basename(name))[0]
        annotate_utils.save_xml(int(img_width), int(img_height), boxes, name, os.path.join(labels_path, just_name))

    next_img_data = annotate_utils.get_image_base64(d_path, int(index))
    next_img_name = next_img_data["name"]

    next_img_name = os.path.splitext(os.path.basename(next_img_name))[0]

    next_img_data["bboxes"] = annotate_utils.load_xml(labels_path, next_img_name)

    return json.dumps(next_img_data)


@app.route('/apply', methods=['POST'])
def apply():
    img_width = request.form['w']
    img_height = request.form['h']
    name = request.form['name']
    boxes = request.form['boxes']
    boxes = json.loads(boxes)
    annotate_utils.save_xml(int(img_width), int(img_height), boxes, name)

    return "ok"


@app.route('/do_augmentation', methods=['POST'])
def do_augmentation():

    _pipeline = request.form['payload']
    pipeline = json.loads(_pipeline)

    d_path = pipeline["path"]
    angles = pipeline["angles"]
    resize = pipeline["resize"]

    operations = pipeline["ops"]

    new_size = None
    if resize != "":
        new_size = list(map(lambda x: int(x), resize.split(",")))

    Augmentation(d_path, list(map(lambda x: int(x), angles.split(","))), operations, new_size)

    return "ok"


@app.route('/preview_augmentation', methods=['POST'])
def preview_augmentation():
    _pipeline = request.form['payload']
    pipeline = json.loads(_pipeline)

    operations = pipeline["ops"]
    resize = pipeline["resize"]
    img = request.form['img']
    angles = pipeline["angles"]

    new_size = list(map(lambda x: int(x), resize.split(",")))
    angles = list(map(lambda x: int(x), angles.split(",")))

    imgs = augment.preview_img(img, operations, new_size, angles)

    return json.dumps(imgs)


@app.route('/save_pipeline', methods=['POST'])
def save_pipeline():
    pipe_path = request.form['path']
    _pipeline = request.form['pipeline']

    with(open(pipe_path, "w")) as f:
        f.write(_pipeline)

    f.close()

    return "ok"


@app.route('/load_pipeline', methods=['POST'])
def load_pipeline():
    pipe_path = request.form['path']

    with(open(pipe_path, "r")) as f:
        res = f.readline()

    f.close()

    return res


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
