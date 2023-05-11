model_root = '/opt/ml/baseline/mmdetection/configs/_base_/models/'

_base_ = [
    './cascade_rcnn_r50_fpn.py',
    './dataset.py',
    './schedule_1x.py',
    './default_runtime.py'
]

# models = [
# "cascade_rcnn_r50_fpn.py",
# "fast_rcnn_r50_fpn.py",
# "faster_rcnn_r50_caffe_c4.py",
# "faster_rcnn_r50_caffe_dc5.py",
# "faster_rcnn_r50_fpn.py",
# "retinanet_r50_fpn.py",
# "ssd300.py",
# ]