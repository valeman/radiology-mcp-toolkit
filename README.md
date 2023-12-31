# Radiology Mondrian Conformal Prediction Toolkit
This repository contains a toolkit for implementing Mondrian conformal prediction (MCP) in radiological deep learning settings. While it is designed for use in medical image analysis, the creators encourage researchers and curious minds from other disciplines to apply MCP to their workflows to further investigate its potential. This toolkit is released in conjunction with the paper "Toward Clinically Trustworthy Deep Learning: Applying Conformal Prediction to Intracranial Hemorrhage Detection" from the [Mayo Clinic AI Lab](https://mayo-radiology-informatics-lab.github.io/MIDeL/index.html).

## Features
- Sample inference script for multilabel hemorrhage detection on head CT slices using MCP 
- Weights for state-of-the-art ICH detection YOLOv8 model (download [here](https://cq500-mcp.s3.amazonaws.com/yolo-v8-final-weights.pt))
- Detailed descriptions of methods (via comments) for each major step of the MCP process
- Accompanying figures to demonstrate MCP and videos to further explain the implementation
- Instructions for applying our MCP methodology to other use cases
## Usage
1. Decide whether you are performing multilabel classification and localization (i.e. object detection) or another task. If the former is true, you should follow our instructions as closely as possible. If the latter is true, however, you will likely need to significantly modify our pipeline or develop your own, but the general process of calibration and Mondrian conformal prediction will still apply.
2. Before you begin training your model, designate a subset of your in-domain data as a held-out calibration dataset. If you have already trained your model, select small subset of your test dataset to serve as your calibration dataset.
3. Define a heuristic notion of uncertainty (calibration score) for your task. If you are using a model from the YOLO family, this should be the ```conf``` value of your model's predictions.
4. Follow the instructions in ```calibrate.py``` to obtain and format calibration scores for your calibration dataset.

## Citation
If you find this work useful, please use the appropriate citation format by clicking "Cite this repository" in the About section.
