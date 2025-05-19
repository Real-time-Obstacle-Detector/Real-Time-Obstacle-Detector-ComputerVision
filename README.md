
## Real-time Obstacle Detector Project: ComputerVision

### Overview

This repository implements the core computer vision pipeline of the Real-time Obstacle Detector Project, including data preprocessing, model training, and inference scripts. for more information about the application, please check this reposiroty: [Real-Time-Obstacle-Detector-Android](https://github.com/Abtinz/Real-Time-Obstacle-Detector-Android).

### Features
<img src="https://avatars.githubusercontent.com/u/26833451?s=200&v=4" alt="ultralytics" width="25" height="25" /> <img src="https://avatars.githubusercontent.com/u/53104118?s=200&v=4" alt="roboflow" width="25" height="25" />
<img src="https://avatars.githubusercontent.com/u/15658638?s=200&v=4" alt="tesorflow" width="25" height="25" /> <img src="https://avatars.githubusercontent.com/u/21003710?s=200&v=4" alt="PyTorch" width="25" height="25" /> <img src="https://avatars.githubusercontent.com/u/34455048?s=200&v=4" alt="Keras" width="25" height="25" /> <img src="https://avatars.githubusercontent.com/u/31675368?s=200&v=4" alt="ONNX" width="25" height="25" /> <img src="https://irangpu.com/wp-content/uploads/2019/05/IG_Logo-e1713278154978.png" alt="IranGPU" width="25" height="25" /> <img src="https://avatars.githubusercontent.com/u/33467679?s=200&v=4" alt="Google Colab" width="25" height="25" />

## Description

**Real-time Obstacle Detector (Computer Vision)** is a Python-based module that implements an on-device object detection pipeline for pedestrian and environmental hazards. Leveraging the ultralytics YOLOv8 framework, this repository provides reproducible scripts for training, evaluating, and exporting a custom object detector model suited for real-time applications on resource-constrained devices.

The core of the solution retrains YOLOv8-nano on a Roboflow-hosted dataset of obstacle categories (e.g., road edges, vehicles, pedestrians, dogs) with tailored augmentation strategies to improve robustness under varied lighting and occlusion conditions. A modular training script allows seamless integration of new data and augmentations, while preserving full compatibility with YOLOv8’s hyperparameter and callback systems.

Post-training, the pipeline includes tools to generate detailed evaluation artifacts: confusion matrices, per-class precision/recall/F1 metrics, and mAP curves. These statistics guide model refinement, ensuring that critical obstacle classes achieve high detection performance before deployment. Finally, the repository includes utilities to export the trained PyTorch model into TensorFlow Lite format (with FP16 and INT8 quantizations), preparing the detector for integration into Android or other edge platforms. Comprehensive examples demonstrate inference on single images, video streams, and whole folders of test data.

## Statistics and Vlidation

### Precision, Recall, F1-Score

| Class      | Precision | Recall | mAP50 |
| ---------- | --------- | ------ | -------- |
| Overall-YOLOv8n        | 0.888    |   0.809   |    0.884     |
| Bike      | 0.942      | 0.901   | 0.972     |
| Building    | 0.862      | 0.926   | 0.898    |
| Car | 0.92      | 0.733   | 0.868     |
| Person |   0.869   |  0.7  |   0.841   |
| Stairs |    0.959  |    0.951   |   0.986  |
| Traffic sign |   0.899    |  0.865    |   0.88   |
| Electrical Pole |   0.874  |    0.833   |   0.917    |
| Road |  0.824   |   0.506   |   0.767    |
| Motorcycle |   0.899  |   0.721  |    0.867   |
| Dustbin  | 0.92    |    0.5   |   0.625     |
| Dog  | 0.888    |      1   |   0.987     |
| Manhole  | 0.958  |    0.924  |    0.957     |
| Tree  | 0.863   |   0.843  |    0.881     |
| Dustbin  | 0.92    |    0.5   |   0.625     |
| Guard rail  | 0.753   |   0.898   |   0.919    |
| Pedestrian crosswalk  | 0.881   |    0.772  |    0.836    |
| Truck  | 0.85    |  0.636   |   0.807    |
| Bus  | 0.867   |   0.848    |   0.91    |
| Bench  | 0.959    |      1    |  0.995    |


## Requirements

* **Python 3.8+**
* **PyTorch**
* **Ultralytics**, **Onxx** 
* **Roboflow**
* **OpenCV**
* **NumPy**, **Pandas**, **Matplotlib**
* **TensorFlow** & **TensorFlow Lite**

## Getting Started

1. **Clone** this repository:

   ```bash
   git clone https://github.com/YourUsername/Real-time-Obstacle-Detector-ComputerVision.git
   cd Real-time-Obstacle-Detector-ComputerVision
   ```
2. **Install** dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## License

MIT License

Copyright (c) 2024 Abtin Zandi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
authors OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
