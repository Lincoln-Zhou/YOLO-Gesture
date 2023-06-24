from typing import List
import time
from contextlib import contextmanager

import cv2
import numpy as np
import torch
# import torchvision
import onnx
import onnxruntime

import yolo_src_util as ops
from config import KPT_SHAPE, handpose_label_map_inv, ORDER


@contextmanager
def Timer(msg):
    print(msg)
    start = time.perf_counter()
    try:
        yield
    finally:
        print("%.4f ms" % ((time.perf_counter() - start) * 1000))


class ONNXPoseResult:
    def __init__(self, bbox: np.ndarray, kpt: np.ndarray) -> None:
        self.bbox = bbox
        self.kpt = kpt


def preprocess_img(img: str):
    img = cv2.imread(img) if isinstance(img, str) else img

    resized = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    resized = np.stack(ops.LetterBox((640, 640), auto=False, stride=32)(image=resized)).astype(np.float32)

    resized = resized.transpose((2, 0, 1))  # (3, 640, 640)
    resized = np.expand_dims(resized, axis=0)  # Add batch dimension
    resized /= 255

    return img, resized


def postprocess(model_output: torch.Tensor, img, orig_img):
    pred = ops.non_max_suppression(model_output, nc=1)[0]
    shape = orig_img.shape

    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
    pred_kpts = pred[:, 6:].view(len(pred), *KPT_SHAPE) if len(pred) else pred[:, 6:]
    pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, shape)

    return ONNXPoseResult(pred[:, :6].numpy(), pred_kpts.numpy())


def single_img_inference(session, img):
    ori_img, resized = preprocess_img(img)

    if isinstance(session, onnxruntime.InferenceSession):
        input_name = session.get_inputs()[0].name
        output_names = [x.name for x in session.get_outputs()]
        
        output = session.run(output_names, {input_name: resized})[0]
    else:   # RKNN session
        output = session.inference(inputs=[resized], data_format='nchw')[0]

    model_output = torch.Tensor(output)

    predicted = postprocess(model_output, torch.Tensor(resized), ori_img)
    
    return predicted, ori_img


def keypoint_cls(session, pose_result: ONNXPoseResult):
    res = []
    
    for index in range(pose_result.bbox.shape[0]):
        x1, y1, x2, y2, _, _ = pose_result.bbox[index]
        kp = pose_result.kpt[index].reshape(KPT_SHAPE)
        
        width, height = x2 - x1, y2 - y1
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        
        kp[:, 0] = (kp[:, 0] - cx) / width
        kp[:, 1] = (kp[:, 1] - cy) / height
        
        kp = kp.reshape((1, -1))
        
        if isinstance(session, onnxruntime.InferenceSession):
            input_name = session.get_inputs()[0].name
            output_names = [x.name for x in session.get_outputs()]
            
            output = session.run(output_names, {input_name: kp})[0]
        else:   # RKNN session
            output = session.inference(inputs=[kp])[0]
            
        predicted_cls = np.argmax(output)
        res.append(predicted_cls)
    
    return res
        
        
def visualize_pose_res(pred: ONNXPoseResult, cls_res, img: np.ndarray):
    res = img

    assert len(pred.bbox) == len(pred.kpt) == len(cls_res), 'Data mismatch'

    for index in range(pred.bbox.shape[0]):
        x1, y1, x2, y2, conf, _ = pred.bbox[index]

        kpt = pred.kpt[index]

        res = cv2.rectangle(res, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        res = cv2.putText(res, f"{conf: .2f}", (int(x2), int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 0), 2)
        res = cv2.putText(res, handpose_label_map_inv[cls_res[index]], (int(x2), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 2)

        for pt in kpt.reshape((21, 2)):
            res = cv2.circle(res, pt.astype(int).tolist(), 0, (0, 255, 0), -1)

        if ORDER:
            for route in ORDER:
                for i in range(0, len(route) - 1, 1):
                    res = cv2.line(res, kpt[route[i]].astype(int).tolist(), kpt[route[i + 1]].astype(int).tolist(), (0, 255, 0), 2)
    
    return res
