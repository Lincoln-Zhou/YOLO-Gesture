import cv2
import onnxruntime

import copy
import argparse
import os

from onnx_utils import single_img_inference, keypoint_cls, visualize_pose_res, Timer
from config import OK_FORMAT, ONNX_YOLO_MODEL, ONNX_MLP_MODEL


kp_session = onnxruntime.InferenceSession(ONNX_YOLO_MODEL, providers=['CPUExecutionProvider'])
gesture_session = onnxruntime.InferenceSession(ONNX_MLP_MODEL, providers=['CPUExecutionProvider'])


def main(src: int | str, save):
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

    args = parser.parse_args()

    if args.source == '0':
        source = 0
    elif os.path.isfile(args.source):
        source = args.source
    else:
        raise ValueError('Invalid source parameter!')

    main(source, args.save)
