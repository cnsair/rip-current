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

# Welcome to the official SAWI Repository

SAWI is a weight-space stabilisation method implemented through a five-stage algorithm that leaves the deployed network unchanged. It is designed to address two coupled behaviours established experimentally in Section IV: continued fine-tuning can reduce cross-dataset sensitivity through source-domain specialisation, and the resulting sensitivity can vary across optimisation trajectories even when the nominal training configuration is unchanged. SAWI therefore replaces dependence on a single terminal checkpoint with a dense average of the fine-tuning trajectory, then interpolates this averaged solution toward the early-stopped task-adaptation checkpoint. BatchNorm statistics are re-estimated after averaging and interpolation, and the deployed operating points are selected using validation data alone. The final model remains a standard SegFormer-B2 with no additional modules, parameters, or inference-time computation. Algorithm 1 specifies the five-stage procedure, and Fig. 2 summarises the construction and validation-only selection of the deployed operating points.

# Datasets 

The datasets used in this study are available from their respective providers: 
- RipDetSeg at https://drive.google.com/drive/folders/1weQw7sucOTEv1Y_mqps0rilU4xfJsAyN
- RipVIS at https://huggingface.co/datasets/Irikos/RipVIS/
- RipAID version 1.0.0 at https://doi.org/10.5281/zenodo.15082427


## Contributing

We welcome contributions! Feel free to contact us with any contribution, including suggestions for improving this readme through nwachukwu_chisom@njit.edu.ng

### Contributing to the Improvment and Reliability of Vision-Based Rip-Current Hazard Monitoring through Weight-Space Stabilization
We are actively looking for ways to improve this domain. 

## Licensing
MiT 

### In summary:
1. Non-commercial use only  
2. Attribution required  
3. Redistribution conditions (≤ 20% unless permission granted)  
4. No full-dataset hosting or mirroring without permission  

## Current Version
Current version of RipVIS is 1.0.0.



