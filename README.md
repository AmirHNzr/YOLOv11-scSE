# YOLOv11 with scSE Attention and CARAFE Module

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**YOLOv11_scSE_CARAFE** is a modified version of the existing YOLOv11 model, re-implemented with **PyTorch**.
The motivation for this project was to enhance the model for improved small object detection.

## Highlights

- Added **scSE attention** to emphasize important parts of the feature map in both spatial and channel-wise dimensions.  
- Replaced the traditional upsampling method in **YOLOv11** with **CARAFE**, which adaptively upsamples based on the input feature map.
