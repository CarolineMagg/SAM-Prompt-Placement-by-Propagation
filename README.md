
# SAM Box Prompt Placement by Propagation


## Overview

This repository contains the code for the MIDL paper [Training-free Prompt Placement by Propagation for SAM Predictions in Bone CT Scans](https://openreview.net/forum?id=F6rhgpGkAy). We have presented four training-free strategies to apply [Segment Anything](https://github.com/facebookresearch/segment-anything) with box prompts to 3D bone CT scans, initialized with only two pixels annotated. Our methods significantly reduce the number of annotated pixels compared to using SAM in the traditional slice-by-slice manner or a fully supervised  egmentation method, while maintaining a certain level of segmentation performance and showing promising results.



## Installation

Our code is based on SAM for prediction and nnUNet for data structure and evaluation. 

1. Create a conda environment with python >= 3.8.
2. Install [PyTorch](https://pytorch.org/get-started/locally/) as described on their website (conda/pip).
3. Install [nnUNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md) as described on their website.
4. Install [SAM](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#installation) as described on their website.
5. Install additional dependencies:
```
pip install nibabel
pip install natsort
pip install scikit-image
```
6. Download pre-trained SAM weights for vit_b from [website](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints). Put the file in the main directory (together with the run_script.sh file)


## Usage

The data has to be structured based on the nnUNet structure, see [nnUNet dataset format](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md) for instructions, and contain images and reference labels.

For our propagation methods, the reference labels are used to select a random sample of the center of the object of interest and extract the initial box prompt. For the traditional SAM evaluation, the reference labels are used to extract one box prompt per slice.

Create **predictions by box prompt propagation**:
```
python -m run_scripts.prompt_propagation \
--dataset NNUNET_DATASET_ID \
--label_order ORDER_OF_LABELS_TO_BE_PROCESSED \
--output_folder OUTPUT_FOLDER \
--experiment EXPERIMENT
```
There are four options for experiment:
* baseline
* stochastic
* nested 
* combined


Create **predictions by prompt extraction from reference labels (reference)**:
```
python -m run_scripts.prompt_reference \
--dataset NNUNET_DATASET_ID \
--label_order ORDER_OF_LABELS_TO_BE_PROCESSED \
--output_folder OUTPUT_FOLDER
```

Perform **postprocessing** (highly recommended for prompt by propagation): <br>
First postprocessing step keeps the largest component associated with the prediction generated by the initial prompt
and removes all other disconnected components:
```
python -m run_scripts.prompt_propagation_postprocessing \
--dataset NNUNET_DATASET_ID \
--label_order ORDER_OF_LABELS_TO_BE_PROCESSED \
--input_folder FOLDER_WITH_PREDICTIONS \
--output_folder OUTPUT_FOLDER
```

Second postprocessing step removes predictions if the box is significantly smaller than the initial box:
```
python -m run_scripts.prompt_propagation_postprocessing2 \
--dataset NNUNET_DATASET_ID \
--label_order ORDER_OF_LABELS_TO_BE_PROCESSED  --input_folder FOLDER_WITH_PREDICTIONS \
--input_folder_prompt FOLDER_WITH_PROMPTS \
--output_folder OUTPUT_FOLDER
```

Note: Postprocessing only creates new nifti files with the processed predictions. The json file with the prompt information is neither altered nor copied. Thus, other scripts might require the folder with the predictions files and the prompt information separately.

Perform **evaluation**: <br>
-> with overlap measurements:
```
python -m run_scripts.prompt_propagation_evaluation \
--dataset NNUNET_DATASET_ID \
--label_order ORDER_OF_LABELS_TO_BE_PROCESSED \
--output_folder FOLDER_WITH_PREDICTIONS
```
-> with surface distances:
```
python -m run_scripts.prompt_propagation_evaluation2 \
--dataset NNUNET_DATASET_ID \
--label_order ORDER_OF_LABELS_TO_BE_PROCESSED \
--output_folder FOLDER_WITH_PREDICTIONS
```

Create **box visulalizations**:
```
python -m run_scripts.create_box_prompt_visualizations \
--dataset NNUNET_DATASET_ID \
--label_order ORDER_OF_LABELS_TO_BE_PROCESSED \ --output_folder FOLDER_WITH_PREDICTIONS \
--output_folder_prompt FOLDER_WITH_PROMPTS
```
Note: For postprocessed predictions, only the original prompt information is available. This will be used for creating the box overlays.

For an example, see the _run_script_example.sh_ file.

## Datasets

Three different datasets are used for our experiments, i.e., two internal datasets and one external dataset. The external dataset is extracted from a publicly available dataset (TotalSegmentator: https://doi.org/10.5281/zenodo.6802613, Wasserthal et al. 2023). Two different subsets are extracted from the provided test set: 33 CT scans containing the left and right humerus and scapula; 53 CT scans containing the left and right hip and femur.


## Acknowledgment
Thanks to the open-source of the following projects:
* [nnUNet](https://github.com/MIC-DKFZ/nnUNet)
* [Segmentation Anything](https://github.com/facebookresearch/segment-anything)
* [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)

## Reference
For more details see [paper](https://openreview.net/forum?id=F6rhgpGkAy).
