import cv2
import onnxruntime

import copy
import argparse
import os

from onnx_utils import single_img_inference, keypoint_cls, visualize_pose_res
from config import OK_FORMAT, ONNX_YOLO_MODEL, ONNX_MLP_MODEL


def main(src: int | str, save, backend, kpt_model, cls_model, target):
    if backend == 'onnx':
        kp_session = onnxruntime.InferenceSession(kpt_model, providers=['CPUExecutionProvider'])
        gesture_session = onnxruntime.InferenceSession(cls_model, providers=['CPUExecutionProvider'])
    else:
        try:
            from rknn.api import RKNN
        except ImportError:
            raise ImportError('RKNN dependency not found!')

        kp_session = RKNN(verbose=False)
        gesture_session = RKNN(verbose=False)

        kp_session.load_rknn(kpt_model)
        gesture_session.load_rknn(cls_model)

        kp_session.init_runtime(target=target)
        gesture_session.init_runtime(target=target)

    cap = cv2.VideoCapture(src)

    if save:
        out = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (1280, 720))

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            res, vis = single_img_inference(session=kp_session, img=frame)

            cls_res = keypoint_cls(session=gesture_session, pose_result=copy.deepcopy(res))

            vis = visualize_pose_res(res, cls_res, vis)

            # Display the annotated frame
            cv2.imshow("Inference", vis)

            if save:
                out.write(vis)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f'{OK_FORMAT}Pipeline completed\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--source', type=str, default='0')
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--backend', type=str, default='onnx')
    parser.add_argument('--kptmodel', type=str, default=ONNX_YOLO_MODEL)
    parser.add_argument('--clsmodel', type=str, default=ONNX_MLP_MODEL)
    parser.add_argument('--target', type=str, default=None)

    args = parser.parse_args()

    if args.source == '0':
        source = 0
    elif os.path.isfile(args.source):
        source = args.source
    else:
        raise ValueError('Invalid source parameter!')

    if args.backend not in ['onnx', 'rknn']:
        raise ValueError('Invalid inferencing backend!')
    elif args.backend == 'onnx' and args.target:
        print('Warning: Target device is specified but ONNX backend is used, this argument would be ignored!')
    elif args.backend == 'rknn' and not args.target:
        raise ValueError('Target device not specified when using RKNN backend!')

    if args.kptmodel != ONNX_YOLO_MODEL and not os.path.isfile(args.kptmodel):
        raise ValueError('Invalid model path!')

    if args.clsmodel != ONNX_MLP_MODEL and not os.path.isfile(args.clsmodel):
        raise ValueError('Invalid model path!')

    if args.target and args.target not in ['rk3566', 'rk3568', 'rk3588', 'rv1103', 'rv1106', 'rk3562']:
        raise ValueError(f'Unsupported target device: {args.target}')

    main(source, args.save, args.backend, args.kptmodel, args.clsmodel, args.target)
