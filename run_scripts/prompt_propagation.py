import argparse
import json
import logging
import os
import random

import nibabel as nib
import numpy as np
from natsort import natsorted

from nnunetv2.paths import nnUNet_raw
from scr.SAMPromptPropagation import SAMPredictorVolumeBaseline, \
    SAMPredictorVolumeStochastic, SAMPredictorVolumeNested, SAMPredictorVolumeNestedStochastic

logging.basicConfig(level=logging.INFO)

# Fix randomness in prompt selection
np.random.seed(1)
random.seed(1)


def get_random_init_slice(idx_first, idx_last, percent_middle):
    r = int((idx_last - idx_first + 1) * percent_middle)
    list_choices = list(range((idx_last - idx_first) // 2 - r // 2, (idx_last - idx_first) // 2 + r // 2 + 1))
    if len(list_choices) == 0:  # too small of a range
        choice = (idx_last - idx_first) // 2
    else:
        choice = random.choice(list_choices)
    return choice + idx_first


def get_first_last_slice(mask_binary):
    idx_first = next(idx for idx, x in enumerate(mask_binary) if np.sum(x) != 0)
    idx_last = len(mask_binary) - 1 - next(idx for idx, x in enumerate(reversed(mask_binary)) if np.sum(x) != 0)
    return idx_first, idx_last


def propagate_prompt_per_class(predictor, class_value, experiment, percent_middle, slice_select=None):
    mask_binary = np.array(predictor.mask_reference_full == class_value, dtype=np.uint8)
    if np.sum(mask_binary) == 0:
        logging.info(f"no SAM predicts volume for single class, since {class_value} has no pixels")
        return None, None
    idx_first, idx_last = get_first_last_slice(mask_binary)
    if experiment == "baseline":
        if slice_select == "random":
            init_slice = get_random_init_slice(idx_first, idx_last, percent_middle)
        elif slice_select == "first":
            init_slice = idx_first
        elif slice_select == "last":
            init_slice = idx_last
        else:
            logging.info(f"{args.slice_select} not valid")
            return None, None
    else:
        init_slice = get_random_init_slice(idx_first, idx_last, percent_middle)
    predictor.select_init_slice(init_slice, class_value)
    pred_mask, prompts = predictor.predict_volume()
    pred_mask = np.moveaxis(pred_mask, 0, -1)
    pred_mask[pred_mask == 1] = class_value
    prompts["idx_init"] = init_slice
    return {class_value: pred_mask}, {class_value: prompts}


if __name__ == '__main__':
    logging.info(" ")
    parser = argparse.ArgumentParser(description="SAM segmentor for medical images with automatically inferred prompts")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--label_order", required=True, nargs='+')
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--experiment",
                        required=True)  # reference, baseline, stochastic, nested, combined
    parser.add_argument("--margin", required=False, default=0, type=int)
    parser.add_argument("--remove_small_boxes", required=False, default=1, type=int)
    parser.add_argument("--percentage", required=False, default=0.5, type=float)
    parser.add_argument("--iterations", required=False, default=10, type=int)
    parser.add_argument("--max_margin", required=False, default=5, type=int)
    parser.add_argument("--majority_vote", required=False, default=0.55, type=float)
    parser.add_argument("--merge_results", required=False, default=0, type=int)
    parser.add_argument("--nnunet_suffix", required=False, default="Tr")
    parser.add_argument("--slice_select", required=False, default="random")  # can be "random", "first" or "last"
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
    logging.info(f"experiment type {args.experiment}")
    logging.info(f"remove small boxes {args.remove_small_boxes}")
    logging.info(f"percentage range of center {args.percentage}")
    if args.experiment == "baseline":
        logging.info(f"slice select {args.slice_select}")
    if args.experiment == "stochastic" or args.experiment == "constoch":
        logging.info(f"iterations {args.iterations}")
        logging.info(f"majority voting {args.majority_vote}")
        logging.info(f"maximal  margin {args.max_margin}")
    if args.experiment == "consistency" or args.experiment == "constoch":
        logging.info(f"merge results {args.merge_results}")

    # iterate through names
    for im_idx, im_name in enumerate(mask_list):
        # Skip non-selected images if specified
        if im_list is not None:
            if im_name not in im_list:
                continue

        # Logging
        logging.info(" ")
        logging.info(f"file {im_name}")

        # Run predictor for volume
        if args.experiment == "baseline":
            predictor = SAMPredictorVolumeBaseline(margin=args.margin, remove_small_boxes=args.remove_small_boxes)
        elif args.experiment == "stochastic":
            predictor = SAMPredictorVolumeStochastic(margin=args.margin, iterations=args.iterations,
                                                     max_margin=args.max_margin,
                                                     remove_small_boxes=args.remove_small_boxes,
                                                     majority_vote=args.majority_vote)
        elif args.experiment == "nested":
            predictor = SAMPredictorVolumeNested(margin=args.margin, aggregate=args.merge_results,
                                                 remove_small_boxes=args.remove_small_boxes)
        elif args.experiment == "combined":
            predictor = SAMPredictorVolumeNestedStochastic(margin=args.margin,
                                                           iterations=args.iterations,
                                                           max_margin=args.max_margin,
                                                           remove_small_boxes=args.remove_small_boxes,
                                                           majority_vote=args.majority_vote,
                                                           aggregate=args.merge_results)
        else:
            logging.info(f"experiment value is not valid: {args.experiment}")
            break
        predictor.read_volume(os.path.join(input_img_dir, im_name.replace(".nii", "_0000.nii")))
        predictor.read_reference(os.path.join(input_seg_dir, im_name))

        if not predictor.check_for_empty_reference(class_labels):
            logging.info(f"no labels {class_labels} present")
            continue

        # iterate through classes
        results_mask = []
        results_prompt = []
        for cls in class_labels:
            b1, b2 = propagate_prompt_per_class(predictor, cls, args.experiment, args.percentage, args.slice_select)
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