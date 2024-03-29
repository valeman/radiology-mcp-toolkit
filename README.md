# Radiology Mondrian Conformal Prediction Toolkit
This repository contains a toolkit for implementing Mondrian conformal prediction (MCP) in radiological deep learning settings. While it is designed for use in medical image analysis, the creators encourage curious minds from other disciplines to apply MCP in their workflow to further investigate its potential. In general, this repository is created for the specific task of multilabel intracranial hemorrhage detection using a YOLOv8 object detection model trained on non-contrast head CT slices. As such, the code used to demonstrate each step of our method is intended for our specific use case. We do not expect that our code will satisfy your use case, especially if it is not related to multilabel object detection on 2D input images. We have done our best to format this repository as a tutorial, and each step has accompanying textual and video instructions for deploying our methods in other use cases. This toolkit is released in conjunction with the paper "Toward Clinically Trustworthy Deep Learning: Applying Conformal Prediction to Intracranial Hemorrhage Detection" from the [Mayo Clinic AI Lab](https://mayo-radiology-informatics-lab.github.io/MIDeL/index.html).

## Features
- Sample inference script for multilabel hemorrhage detection on head CT slices using MCP 
- Weights for state-of-the-art ICH detection YOLOv8 model (download [here](https://cq500-mcp.s3.amazonaws.com/yolo-v8-final-weights.pt))
- Detailed descriptions of methods (via comments) for each major step of the MCP process
    - At the top of each file, we highlight *Prerequisites* and *Return Values* to clarify what is needed before running the file and what you will obtain from running it
- Accompanying figures to demonstrate MCP and videos to further explain the implementation
- Instructions for applying our MCP methodology to other use cases

## Usage
#### Setup and Installation
1. It is strongly encouraged that you have access to a GPU to accelerate your model training and inference time. If you have already trained a model or are using our weights, inference is possible on a CPU but it will be markedly slower.
2. Clone this repository by running the following command: ```git clone https://github.com/c-gamble/radiology-mcp-toolkit.git```
3. Create an environment with Python version ```3.8.16``` using your desired Python dependency manager.
4. Activate your new environment and ensure it is stable.
5. Install all required dependencies by running ```pip install -r requirements.txt```
#### Getting Ready for Conformal Prediction
1. Decide whether you are performing multilabel classification and localization (i.e. object detection) or another task. If the former is true, you should follow our instructions as closely as possible. If the latter is true, you will likely need to significantly modify our pipeline or develop your own, but the general process of calibration and Mondrian conformal prediction will remain the same.
2. Before you begin training your model, designate a subset of your in-domain data as a held-out calibration dataset. If you have already trained your model, select small subset of your test dataset to serve as your calibration dataset.
3. Train and validate (tune) your model until you have achieved satisfactory performance on the validation (tuning) dataset.
#### Performing Conformal Prediction
1.  Define a heuristic notion of uncertainty (calibration score) for your task. If you are using a model from the YOLO family, this should be the ```conf``` value of your model's predictions.
2. Follow the instructions in ```Step 1: Calibration/``` to obtain and format calibration scores for your calibration dataset.
#### Interpreting Results

## Bugs and Feature Requests

## Citation
If you find this work useful, please use the appropriate citation format by clicking "Cite this repository" in the About section.
