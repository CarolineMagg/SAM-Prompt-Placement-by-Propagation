import argparse
import json
import logging
import os
import random

import nibabel as nib
import numpy as np
from natsort import natsorted

from nnunetv2.paths import nnUNet_raw
from scr.SAMReferencePrompt import SAMPredictorVolumeBasedOnReference
from run_scripts.prompt_propagation import get_first_last_slice

logging.basicConfig(level=logging.INFO)

# Fix randomness in prompt selection
np.random.seed(1)
random.seed(1)


def propagate_prompt_per_class(sam_predictor, class_value):
    mask_binary = np.array(sam_predictor.mask_reference_full == class_value, dtype=np.uint8)
    if np.sum(mask_binary) == 0:
        logging.info(f"no SAM predicts volume for single class, since {class_value} has no pixels")
        return None, None
    idx_first, idx_last = get_first_last_slice(mask_binary)
    sam_predictor.select_slices(idx_first, idx_last, class_value)
    pred_mask, prompts = sam_predictor.predict_volume()
    pred_mask = np.moveaxis(pred_mask, 0, -1)
    pred_mask[pred_mask == 1] = class_value
    prompts["idx_init"] = idx_first
    prompts["idx_last"] = idx_last
    return {class_value: pred_mask}, {class_value: prompts}


if __name__ == '__main__':
    logging.info(" ")
    parser = argparse.ArgumentParser(description="SAM segmentor for medical images with reference labels")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--label_order", required=True, nargs='+')
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--remove_small_boxes", required=False, default=0, type=int)
    parser.add_argument("--nnunet_suffix", required=False, default="Tr")
    args = parser.parse_args()

    # Set up dataset
    dataset = "Dataset" + args.dataset
    init_path = os.path.join(nnUNet_raw, dataset)
    logging.info(f"dataset to analyze {dataset}")

    input_img_dir = os.path.join(init_path, f"images{args.nnunet_suffix}")
    input_seg_dir = os.path.join(init_path, f"labels{args.nnunet_suffix}")

    output_dir = args.output_folder
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    mask_list = natsorted(os.listdir(input_seg_dir))
    im_list = None  # if only  subset of files should be analyzed
    class_labels = [int(x) for x in args.label_order]
    num_proc = len(class_labels)
    logging.info(f"# of samples {len(mask_list)}")

    # Logging of dataset info
    logging.info(f"input image directory {input_img_dir}")
    logging.info(f"input segmentation directory {input_seg_dir}")
    logging.info(f"labels {class_labels}")
    logging.info(f"output directory {output_dir}")
    logging.info("run reference")

    # iterate through names
    for im_idx, im_name in enumerate(mask_list):
        # Skip non-selected images if specified
        if im_list is not None:
            if im_name not in im_list:
                continue

        # Logging
        logging.info(" ")
        logging.info(f"file {im_name}")

        predictor = SAMPredictorVolumeBasedOnReference(margin=0)
        predictor.read_volume(os.path.join(input_img_dir, im_name.replace(".nii", "_0000.nii")))
        predictor.read_reference(os.path.join(input_seg_dir, im_name))

        if not predictor.check_for_empty_reference(class_labels):
            logging.info(f"no labels {class_labels} present")
            continue

        # iterate through classes
        results_mask = []
        results_prompt = []
        for cls in class_labels:
            b1, b2 = propagate_prompt_per_class(predictor, cls)
            if b1 is not None and b2 is not None:
                results_mask.append(b1)
                results_prompt.append(b2)
        full_output_mask = np.zeros_like(np.moveaxis(predictor.mask_reference_full, 0, -1), dtype=np.uint8)

        # collect classes
        for cls in class_labels:
            for elem in results_mask:
                for c, mask in elem.items():
                    if c == cls:
                        full_output_mask[mask == cls] = cls
        full_prompts = {k: v for res in results_prompt for k, v in res.items()}

        # store result for sample
        nii = nib.Nifti1Image(full_output_mask.astype(np.uint8), affine=predictor.affine)
        nib.save(nii, os.path.join(output_dir, f'{im_name[:-7]}.nii.gz'))
        with open(os.path.join(output_dir, f'{im_name[:-7]}_prompts.json'), "w") as file:
            json.dump(full_prompts, file)

        break