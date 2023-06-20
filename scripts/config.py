KPT_SHAPE = (21, 2)     # (21, 2)
ORDER = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 17, 18, 19, 20],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [5, 9, 13, 17],
]

ONNX_YOLO_MODEL = "../models/yolov8n-pose-relu-s.onnx"
ONNX_MLP_MODEL = "../models/handpose_cls.onnx"

OK_FORMAT = '\033[92m'

handpose_label_map = {
    'no_gesture': 0,
    'call': 1,
    'dislike': 2,
    'fist': 3,
    'four': 4,
    'like': 5,
    'mute': 6,
    'ok': 7,
    'one': 8,
    'palm': 9,
    'peace': 10,
    'peace_inverted': 11,
    'rock': 12,
    'stop': 13,
    'stop_inverted': 14,
    'three': 15,
    'three2': 16,
    'two_up': 17,
    'two_up_inverted': 18
}

handpose_label_map_inv = dict((v, k) for k, v in handpose_label_map.items())
