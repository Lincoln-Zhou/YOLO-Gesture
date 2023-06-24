## YOLO-Gesture: Gesture Recognition with YOLOv8-Pose+MLP

*This repository contains code and models for my undergraduate final project in Tongji University.*

*No guarantee is made on the reliability, future maintenance or technical support of this repo.*

**We propose YOLO-Gesture, a gesture recognition method which integrates a hand detector and a gesture classifier. The method achieves good recognition result on RGB and IR (infrared) imagery. This project also keeps execution efficiency in mind, achieving ~30 FPS when running on a laptop's CPU.**

---

### Requirements
Python package requirements can be found in `requirements.txt`.

This program is CPU-only, a CUDA-enabled environment or GPU is not necessary.

---

### Usage
We provide a demo script for inferencing on video:

```commandline
cd scripts/
python demo.py --source 0 --save True --backend 'onnx'
```
**Available arguments:**

`--source` argument can be 0 (webcam) or path to a video file;

`--save` argument controls whether to save visualized recognition result in a `.avi` video;

`--backend` argument specifies which inference backend to use. Default as `onnx`, available options: `onnx` (use Open Neural Network Exchange library), `rknn` (use Rockchip RKNN framework, for Rockchip hardware only);

`--kptmodel` path to hand landmark detection model. Ignore this arg to use default path specified in config file;

`--clsmodel` path to gesture classification model. Ignore this arg to use default path specified in config file;

`--target` Target Rockchip device. Will only be used if `backend` is `rknn`. Please refer to official docs for the list of supported devices.

---

### RKNN Support
The program could leverage the Rockchip RKNN API for efficient inference on Rockchip NPU platforms. Please follow the steps below:
* Install RKNN-Toolkit2 SDK, official docs can be found [here](https://github.com/rockchip-linux/rknn-toolkit2/tree/master/doc). Note that we didn't implement support for platforms using RKNN-Toolkit SDK
* Run `demo.py` with `backend` arg set to `rknn`, also specify the paths to corresponding `.rknn` format models. Two sample models can be found inside `./models/`

Please note that our current support for RKNN framework is extremely limited. See the following table:

| Platform  | FP16 Inference | INT8 Inference |
|:---------:|:--------------:|:--------------:|
|  RK3566   |       ✅        |       ❗        |
|  RV1106   |       ❌        |       ❌        |
| Simulator |       ✅        |       ❗        |
✅: Fully functional

❗: It would run, but with high precision loss

❌: Not supported. Errors such as overflow would happen.

**The table might be updated if we locate and fix the issues.**

---

### Explanation
The method mainly consists of two parts: hand landmarks detection and gesture classification.

* For hand detection, a YOLOv8n-Pose model is used to detect hands and the corresponding hand landmarks in a given frame
* For gesture classification, a multi-layer perceptron is used to classify normalized hand landmark coordinates into 19 pre-defined gesture classes.

---

### Acknowledgements
We appreciate [Ultralytics](https://github.com/ultralytics/ultralytics) team for releasing the incredible YOLOv8-Pose model.

The training of our landmark detection and gesture classification models partly uses [the HaGRID dataset](https://github.com/hukenovs/hagrid).

I also express my sincere gratitude to all my friends for their companion and help during this research project.