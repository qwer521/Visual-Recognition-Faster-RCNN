# NYCU Computer Vision 2025 Spring HW2

Author: 黃皓君  
StudentID: 111550034

## Introduction

The task involves digit recognition using an adapted Faster R-CNN framework. The objective is to improve detection robustness and accuracy by introducing custom modifications, such as integrating focal loss into the classification branch and combining predictions from multiple networks (e.g., ResNet50 FPN v2, ResNet50 FPN, and MobileNet). While the ensemble strategy (combining predictions via Non-Maximum Suppression) improves the overall mAP for Task 1, the whole-number predictions (Task 2) from the combined output were suboptimal. Therefore, only the predictions from the ResNet50 FPN v2 model are used for Task 2.

## How to Run
> **Example**:
> 1. Run the training script:
>    ```bash
>    python train.py
>    ```
> 2. Run the inference script:
>    ```bash
>    python test.py
>    ```
> 3. (Optional) Run the script to combine multiple prediction results:
>    ```bash
>    python combine.py
>    ```

## Performance Snapshot

![snapshot](./image/snapshot.jpg)
