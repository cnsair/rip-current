---
language:
- en
pretty_name: Rip Current Detection training using RipDetSeg & RipVIS Datasets
tags:
- semantic & instance segmentation
- segmentation
- computer vision
- rip
- rip_current
size_categories:
- 100K<n<1M
---

# ================================================

# NTIRE 2026 Rip Current Detection and Segmentation (RipDetSeg) Challenge @CVPR2026

# This dataset is part of the aformentioned challenge and is provided for participating in the challenge. It is composed of publicly available (or soon to be available) datasets and it should not be used in this format for any other purpose other than the challenge.

Full readme, licensing details, starting kit, evaluation script, and other challenge information available at https://drive.google.com/drive/folders/1weQw7sucOTEv1Y_mqps0rilU4xfJsAyN
Challenge link: https://www.codabench.org/competitions/12730/

The dataset is split into the following ways:
- **train_images**: Training images
- **train_labels_segmentation**: Segmentation labels in polygon format
- **train_labels_detection**: Detection labels in yolo format
- **train_annotations.json**: Training annotations in COCO format for both bounding boxes and segmentation

# ============================================
# =============================================

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

## Table of Contents
1. [Short Description](#short-description)
1. [Links of Interest](#links-of-interest)
1. [Structure](#structure)
1. [Splits](#splits)
1. [Citations](#citations)
1. [Contributing](#contributing)
1. [Licensing](#licensing)
1. [Workshops and Challenges](#workshops-and-challenges)
1. [Known Limitations](#known-limitations)
1. [Future Updates](#future-updates)
1. [Current Version](#current-version)


**[⬆ back to top](#table-of-contents)**

## Links of interest
- Main contact: andrei.dumitriu@uni-wuerzburg.de
- Website: https://RipVIS.ai
- HuggingFace Link: https://huggingface.co/datasets/Irikos/RipVIS/
- Codabench Evaluation server: coming soon, see ripvis website for updates

**[⬆ back to top](#table-of-contents)**

## Structure

RipVIS folder structure is the following:
```
RipVISv1.8.4_instance_segmentation
    - train
        -- sampled_images.zip
            --- images -> folder containing the RipVIS paper sampled images for training split
            --- addittional_data -> folder containing the extra 2466 images from Dumitriu et al. (CVPRW 2023)
        -- coco_annotations 
            --- train.json -> train file with ONLY the data from RipVIS paper
            --- train_with_additional_data.json -> train file with the data from RipVIS paper and the extra 2466 annotated images from Dumitriu et al. (CVPRW 2023)
            --- additional_train_data.json -> train file with only the additional data from Dumitriu et al. (CVPRW 2023)
        -- yolo_annotations.zip -> labels (txt files with train annotations in yolo format)
        -- videos -> full videos from training split (both with rips and no-rips)

    - val
        -- sampled_images.zip
            --- images -> folder containing the RipVIS paper sampled images for validation split
        -- coco_annotations 
            --- val.json -> annotations for validation split in coco format
        -- yolo_annotations.zip -> labels (txt files with validation annotations in yolo format)
        -- videos -> full videos from validation split (both with rips and no-rips)

    - test
        -- coco_annotations 
            --- test_without_annotations.json -> test split in coco format, but with video and frame names from all test split videos, but without annotations. File is provided for structure information when submitting for automatic evaluation.
        -- videos -> full videos from test split (both with rips and no-rips)

    - compute_coco_ap.py -> script used to evaluate the results and compute AP50 and AP[50:95]
    - compute_pr_f1_f2.py -> script used to evaluate the results and comptue F1 and F2 scores
        
    - RipVISv1.8.4_dataset_info.pdf -> contains information on the dataset (FPS, Duration, Sampling Rate, Resolution, Annotator, Total Sampled Frames and Video Source). Please check "last_update" always when comparing information.
    - Readme.md (this Readme file)
```

As exemplifie in the folder structure, we provided the videos and the sampled images alongside their annotations. We also included the "additional data", both images and annotations, which is the training data from our previous paper, [Rip Current Segmentation: A Novel Benchmark and YOLOv8 Baseline Results](https://arxiv.org/abs/2504.02558), presented at CVPRW in 2023. The test data from that paper has been included in the RipVIS data.

For the results in the RipVIS paper, the additional data (2466 images with polygon annotations) has been used in training.

**[⬆ back to top](#table-of-contents)**


## Splits
Due to the amorphous nature of the rip currents, the different locations, video duration and other factors, two of our co-authors, which are rip currents experts, worked in doing a manual train-val-test split. This split structure allows for a reasonable distribution in size, both in video durations and number of annotated frames, ensuring no leakage of data between splits. 

Unfortunately, automatic split of videos (or even worse, of sampled frames) leads to results that do not accuratenly portrait real world results. This can occur due to data leakage (having sampled frames from the same video in multiple splits), uneven distribution of time (splits done on number or videos leading to equal number of videos but of different durations), uneven distribution of orientation or uneven distribution of rip current types. We did our best to minimze all these possible errors in the manual split.

Thus, our recommendation, for obtaining accurate results is to use this split structure. 

<b>The format of the videos and frames is the following:</b>
1. ```RipVIS-<video_number>``` for videos with rip currents
1. ```RipVIS-NR-<video_number>```for video without rip currents (NR = No Rips)
1. ```RipVIS-<video_number>_<frame_number>``` or ```RipVIS-NR-<video_number>_<frame_number>```for frames in video (results should be reported on frames)
1. video_number is zero-padded to 3 digits. E.g. 4th video is `RipVIS-004.mp4` not `RipVIS-4.mp4`
1. frame_number is zero-padded to 5 digits. E.g. 28th frames from 4th video is `RipVIS-004_00028.jpg` and NOT `RipVIS-004_28.jpg`. This difference is relevant for the evaluation script.


### Train
RipVIS-
```['003', '004', '005', '006', '010', '011', '016', '017', '018', '020', '021', '022', '028', '029', '030', '031', '032', '033', '034', '035', '036', '037', '040', '041', '042', '045', '049', '050', '051', '052', '054', '056', '057', '058', '060', '061', '062', '063', '064', '065', '068', '069', '070', '071', '075', '076', '077', '080', '082', '083', '084', '085', '086', '087', '088', '089', '091', '092', '093', '094', '095', '096', '097', '098', '100', '101', '103', '104', '105', '106', '107', '110', '111', '112', '115', '116', '117', '118', '122', '123', '124', '126', '127', '135', '136', '140', '141', '148', '149', '150']```

RipVIS-NR-
```['002', '003', '006', '007', '008', '009', '010', '011', '012', '015', '016', '017', '018', '025', '027', '028', '029', '030', '031', '032', '033', '034']```

### Validation
RipVIS-
```['001', '007', '012', '014', '015', '024', '026', '039', '044', '046', '053', '059', '066', '067', '072', '079', '090', '102', '108', '109', '121', '128', '129', '133', '134', '137', '143', '144', '146', '147']```

RipVIS-NR-
```['004', '019', '020', '022', '024', '026']```

### Test
RipVIS-
```['002', '008', '009', '013', '019', '023', '025', '027', '038', '043', '047', '048', '055', '073', '074', '078', '081', '099', '113', '114', '119', '120', '125', '130', '131', '132', '138', '139', '142', '145']```

RipVIS-NR-
```['001', '005', '013', '014', '021', '023']```

**[⬆ back to top](#table-of-contents)**


## Citations
Our work is comprised of 3 papers (and many more to come):

1. **RipVIS: Rip Currents Video Instance Segmentation Benchmark for Beach Monitoring and Safety**
<pre>
@inproceedings{dumitriu2025ripvis,
    author    = {Dumitriu, Andrei and Tatui, Florin and Miron, Florin and Ralhan, Aakash and Ionescu, Radu Tudor and Timofte, Radu},
    title     = {RipVIS: Rip Currents Video Instance Segmentation Benchmark for Beach Monitoring and Safety},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {3427--3437}
}
</pre>

2. **AIM 2025 Rip Current Segmentation (RipSeg) Challenge**
<pre>
@inproceedings{aim2025ripseg,
    title={{AIM} 2025 Rip Current Segmentation ({RipSeg})} Challenge Report,
    author={Andrei Dumitriu and Florin Miron and Florin Tatui and Radu Tudor Ionescu and Radu Timofte and Aakash Ralhan and Florin-Alexandru Vasluianu and others},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    year={2025}
}
</pre>

3. **Rip Current Segmentation: A Novel Benchmark and YOLOv8 Baseline Results**
<pre>
@inproceedings{dumitriu2023rip,
    title="{Rip Current Segmentation: A novel benchmark and YOLOv8 baseline results}",
    author={Dumitriu, Andrei and Tatui, Florin and Miron, Florin and Ionescu, Radu Tudor and Timofte, Radu},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    pages={1261--1271},
    year={2023}
}
</pre>

**[⬆ back to top](#table-of-contents)**


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

## Known Limitations
1. Some of the videos have been annotated with Roboflow. By default, Roboflow sorts files by "Newest". This, however, is rather random when uploading files in bulk. We realised this at some point and sorted the files by filename, leading to annotations in the frame order. However, for the videos annotated with "Newest", the annotations can be quite jittery when overlaied in a video. This does not affect the quantitative results in any observable way and is strictly a visual issue. 
1. Video sampling rate varies greatly, depending on video duration, movement and annotator disponibility at the specific time. Frames are usually numbered in such a way that they match the exact frame from the video (e.g. RipVIS-015_00005.jpg is the 6th frame from RipVIS-015 video). However, several annotations contain the frames in order, with frame number not matching the actual frame from the video. The actual frame can still be reasonably accurately found by calculating the sampling rate, the total number of frames and the total number of annotated frames. We will fix this in a future patch. 
Known videos: RipVIS-
```['001', '003', '004', '006', '007', '011', '012', '014', '016', '017', '018', '020', '021', '022', '024', '033', '034', '035', '036', '037', '039', '040']```.
1. Some sampled and annotated frames were removed due to invalid annotation format (an error most likely created in transitioning from video to frames to Roboflow and back). While we mitigated this, it is still the case that a a small number of annotated sampled frames might be missing in the published version. 

**[⬆ back to top](#table-of-contents)**


## Future Updates
1. Codabench website for automatic evaluation on the test split (ETA mid October 2025). 
1. Fixes for known limitation #2.
1. Manually re-add the #3 in known limitations.
1. Organizing a new challenge.

**[⬆ back to top](#table-of-contents)**

## Current Version
Current version of RipVIS is 1.8.4.

Last DATASET update: 25.09.2025 15:40
Last README update: 27.09.2025 13:00

**[⬆ back to top](#table-of-contents)**


