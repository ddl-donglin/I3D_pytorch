# I3D models trained on Vidor

## Overview

This repo is the trunk net 4 2nd task of VidVRD: Video Relation Prediction

The 1st stage project: [Video Object Detection](https://github.com/Daviddddl/FasterRCNN4VidVRDT1.git)

[The Grand Challenge MM2019](http://lms.comp.nus.edu.sg/research/dataset.html) 

This repository contains trained models reported in the paper "[Quo Vadis,
Action Recognition? A New Model and the Kinetics
Dataset](https://arxiv.org/abs/1705.07750)" by Joao Carreira and Andrew
Zisserman.

This code is based on Deepmind's [Kinetics-I3D](https://github.com/deepmind/kinetics-i3d). Including PyTorch versions of their models.

## Download
[Charades_v1_rgb](http://ai2-website.s3.amazonaws.com/data/Charades_v1_rgb.tar)

[Vidor](http://lms.comp.nus.edu.sg/research/dataset.html)

# Fine-tuning and Feature Extraction
We provide code to extract I3D features and fine-tune I3D for vidor.
Our fine-tuned models on Vidor are also available in the models director (in addition to Deepmind's trained models).
The Charades pre-trained models on Pytorch were saved to (flow_charades.pt and rgb_charades.pt).
The deepmind pre-trained models were converted to PyTorch and give identical results (flow_imagenet.pt and rgb_imagenet.pt).
These models were pretrained on imagenet and kinetics (see [Kinetics-I3D](https://github.com/deepmind/kinetics-i3d) for details). 

## Fine-tuning I3D
[train_i3d.py](train_i3d.py) 
contains the code to fine-tune I3D based on the details in the paper and obtained from the authors.
Specifically, this version follows the settings to fine-tune on the 
[Charades](allenai.org/plato/charades/) dataset based on the author's implementation 
that won the Charades 2017 challenge. 
The charades fine-tuned RGB and Flow I3D models are available in the model directory 
(rgb_charades.pt and flow_charades.pt).

This relied on having the optical flow and RGB frames extracted and saved as images on dist.
[vidor_dataset.py](vidor_dataset.py) script <b>VidorPytorchTrain Class</b>
contains the code to load charades video segments for training.

E.g.
```bash
python train_i3d.py -anno_rpath /storage/dldi/PyProjects/vidor/annotation -video_rpath /storage/dldi/PyProjects/vidor/train_vids
```

This relied on having the optical flow and RGB frames extracted and saved as images on dist.
[charades_dataset.py](charades_dataset.py) contains the code to load charades video segments for training.

## Feature Extraction
[extract_features.py](extract_features.py) 
contains the code to load a pre-trained I3D model and extract the features 
and save the features as numpy arrays.

E.g.
```bash
python extract_features.py -anno_rpath /storage/dldi/PyProjects/vidor/annotation -video_rpath /storage/dldi/PyProjects/vidor/train_vids
```
The [vidor_dataset.py](vidor_dataset.py) script <b>VidorPytorchExtract Class</b> 
loads an entire video to 
extract per-segment features.
The [charades_dataset_full.py](charades_dataset_full.py) script loads an entire video to 
extract per-segment features.

