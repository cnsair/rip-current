---
language:
- en
pretty_name: SAWI (SWAD Anchored Weight Interpolation) for Vision-Based Rip-Current 
tags:
- semantic segmentation
- computer vision
- Weight-Space Stabilization
- rip current
- Hazard Monitoring 
size_categories:
- 100K<n<1M
author:
- NWACHUKWU CHISOM SAMSON
---

# NTIRE 2026 Rip Current Detection and Segmentation (RipDetSeg) Challenge @CVPR2026

Full readme, licensing details, starting kit, evaluation script, and other challenge information available at https://drive.google.com/drive/folders/1weQw7sucOTEv1Y_mqps0rilU4xfJsAyN
Challenge link: https://www.codabench.org/competitions/12730/

The dataset is split into the following ways:
- **train_images**: Training images
- **train_labels_segmentation**: Segmentation labels in polygon format
- **train_labels_detection**: Detection labels in yolo format
- **train_annotations.json**: Training annotations in COCO format for both bounding boxes and segmentation

# RipVIS v1.8.4
This Readme describes the RipVIS dataset, its contents, structure, known limitations, how to use it and what to expect in future updates. For more details, future challenges and other information, keep an eye on [RipVIS website](https://ripvis.ai) or write to andrei.dumitriu@uni-wuerzburg.de .

Current version: v1.1.6
Last Update: 1st of February, 10:00 P.M.

## Short description
RipVIS dataset was introduced with [RipVIS: Rip Currents Video Instance Segmentation Benchmark for Beach Monitoring and Safety](https://arxiv.org/abs/2504.01128) paper, accepted at [CVPR 2025](https://cvpr.thecvf.com/Conferences/2025). It is the result of a collaboration of a multi-disciplinary team between [University of Würzburg's](https://www.uni-wuerzburg.de/en/) [Computer Vision Laboratory](https://www.informatik.uni-wuerzburg.de/computervision/) and [University of Bucharest's](https://unibuc.ro/) [Faculty of Mathematics and Computer Science](https://fmi.unibuc.ro/) and [Faculty of Geography](https://fmi.unibuc.ro/).

The dataset consists of 184 videos, out of which 150 videos contain rip currents annotated for instance segmentation. It is authored by:
- Andrei Dumitriu (andrei.dumitriu@uni-wuerzbuerg.de)
- Conf. Dr. Florin Tatui
- Florin Miron
- Aakash Ralhan
- Prof. Dr. Radu Ionescu
- Prof. Dr. Radu Timofte

## Contributing

We welcome contributions! Please check out https://RipVIS.ai for more details. Feel free to contact us with any contribution, including suggestions for improving this readme.

### Contributing to the extension of RipVIS Dataset
We are actively increasing the RipVIS dataset. If you have a video with rip currents, you can send it to us and we will annotate it and include it in the dataset. The video is added under a license decided by you and the video source is credited 100% to you.


**[⬆ back to top](#table-of-contents)**

## Licensing
This dataset is released under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0), **with the following additional conditions**:

- Hosting or redistribution of the dataset in its entirety, without explicit written permission, is not permitted.  
- Redistribution of RipVIS as part of derivative datasets is only permitted if RipVIS constitutes no more than 20% of the resulting dataset. For larger inclusions, prior written permission from the main author is required.  

### In summary:
1. Non-commercial use only  
2. Attribution required  
3. Redistribution conditions (≤ 20% unless permission granted)  
4. No full-dataset hosting or mirroring without permission  

By downloading or using RipVIS, you agree to these terms.  

A subset of RipVIS that was collected and annotated entirely by the authors is available for commercial licensing. For inquiries regarding commercial use, please contact the main author (andrei.dumitriu@uni-wuerzburg.de).

**[⬆ back to top](#table-of-contents)**


## Workshops and Challenges
1. We organized the [AIM 2025 Rip Current Segmentation (RipSeg) Challenge](https://www.codabench.org/competitions/9109/) challenge at AIM workshop in conjuction with [ICCV2025](https://iccv.thecvf.com/). See the [AIM 2025 Rip Current Segmentation (RipSeg) Challenge Report](https://arxiv.org/abs/2508.13401) on arXiv, which will be published in the ICCVW2025 Proceedings.
1. Another challenge coming soon, stay tuned.

## Current Version
Current version of RipVIS is 1.8.4.

Last DATASET update: 25.09.2025 15:40
Last README update: 27.09.2025 13:00

**[⬆ back to top](#table-of-contents)**


