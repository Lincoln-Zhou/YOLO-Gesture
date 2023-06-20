## YOLO-Gesture: Gesture Recognition with YOLOv8+MLP

*This repository contains code and models for my undergraduate final project in Tongji University.*

*No guarantee is made on the reliability, future maintenance or technical support of this repo.*

**We propose YOLO-Gesture, a gesture recognition method which integrates a hand detector and a gesture classifier. The method achieves good recognition result on RGB and IR (infrared) imagery. This project also keeps execution efficiency in mind, achieving ~30 FPS when running on a laptop's CPU.**

### Requirements
Python package requirements can be found in `requirements.txt`.

This program is CPU-only, a CUDA-enabled environment or GPU is not necessary.

### Usage
We provide a demo script for inferencing on video:

```commandline
python scripts/demo.py --source 0 --save True
```
`--source` argument can be 0 (webcam) or path to a video file;

`--save` argument controls whether to save visualized recognition result in a `.avi` video.

### Explanation
The method mainly consists of two parts: hand landmarks detection and gesture classification.

* For hand detection, a YOLOv8n-Pose model is used to detect hands and the corresponding hand landmarks in a given frame
* For gesture classification, a multi-layer perceptron is used to classify normalized hand landmark coordinates into 19 pre-defined gesture classes.

### Acknowledgements
We appreciate [Ultralytics](https://github.com/ultralytics/ultralytics) team for releasing the incredible YOLOv8-Pose model.

The training of our landmark detection and gesture classification models partly uses [the HaGRID dataset](https://github.com/hukenovs/hagrid).

I also express my sincere gratitude to all my friends for their companion and help during this research project.