from flask import Flask, render_template, request

import utils.xml_utils as xml_to_yolo
import utils.augment as augment
import json

app = Flask(__name__, static_url_path='/static')


@app.route('/')
def hello_world():
    return app.send_static_file('index.html')


@app.route('/datasetinfo', methods=['POST'])
def get_dataset_info():
    d_path = request.form['path']
    classes = request.form['classes']

    xml_to_yolo.main(d_path, classes.split(","))

    return "ok"


@app.route('/augmentation')
def augmentation():
    return app.send_static_file('augment.html')


@app.route('/do_augmentation', methods=['POST'])
def do_augmentation():
    d_path = request.form['path']
    angles = request.form['angles']
    resize = request.form['resize']
    ops = request.form['ops']

    operations = json.loads(ops)["ops"]

    new_size = None
    if resize != "":
        new_size = list(map(lambda x: int(x), resize.split(",")))

    augment.run_aug(d_path, list(map(lambda x: int(x), angles.split(","))), operations, new_size)
    return "ok"


@app.route('/preview_augmentation', methods=['POST'])
def preview_augmentation():
    ops = request.form['ops']

    img = request.form['img']

    operations = json.loads(ops)["ops"]

    imgs = augment.preview_img(img, operations)

    return json.dumps(imgs)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
