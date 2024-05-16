import logging
from collections import OrderedDict

import nibabel as nib
import numpy as np
import json
import cv2
import os
import argparse

from natsort import natsorted
from nnunetv2.paths import nnUNet_raw

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    logging.info(" ")
    logging.info("Create box prompt visualization")
    parser = argparse.ArgumentParser(description="Create box prompt visualization for SAM generated predictions")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--label_order", required=True, nargs='+')
    parser.add_argument("--margin", required=False, default=0, type=int)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--output_folder_prompt", required=False)
    parser.add_argument("--nnunet_suffix", required=False, default="Tr")
    parser.add_argument("--direction", required=False, default="dir1")
    args = parser.parse_args()

    nnunet_folder = "Dataset" + args.dataset
    labels = args.label_order
    direction = args.direction

    files = natsorted([x[:-7].replace("_0000", "") for x in
                       os.listdir(os.path.join(nnUNet_raw, nnunet_folder, f"images{args.nnunet_suffix}"))])

    for sample_idx in files:
        if not os.path.exists(os.path.join(args.output_folder, f"{sample_idx}.nii.gz")):
            continue

        print(f"{sample_idx}")
        nii = nib.load(
            os.path.join(nnUNet_raw, nnunet_folder, f"images{args.nnunet_suffix}", f"{sample_idx}_0000.nii.gz"))
        image = np.asarray(nii.get_fdata())

        if args.output_folder_prompt is None:
            with open(os.path.join(args.output_folder, f"{sample_idx}_prompts.json")) as file:
                prompts = json.load(file)
        else:
            with open(os.path.join(args.output_folder_prompt, f"{sample_idx}_prompts.json")) as file:
                prompts = json.load(file)

        nii = nib.load(os.path.join(args.output_folder, f"{sample_idx}.nii.gz"))
        pred = np.asarray(nii.get_fdata(), dtype=np.uint8)

        # for reference prompts:

        volume = np.zeros_like(image, dtype=np.uint8)
        for slice_idx in range(0, pred.shape[-1]):
            pr = pred[..., slice_idx]
            box = OrderedDict()
            for cls in labels:
                if "idx_first" in prompts[cls].keys() and "idx_last" in prompts[cls].keys():
                    idx_first = int(prompts[cls]["idx_first"])
                    idx_last = int(prompts[cls]["idx_last"])
                    if direction == "dir2":
                        if str(idx_last - slice_idx) not in prompts[cls][direction].keys():
                            continue
                        bbox = prompts[cls][direction][str(idx_last - slice_idx)]
                    else:
                        if str(slice_idx - idx_first) not in prompts[cls][direction].keys():
                            continue
                        bbox = prompts[cls][direction][str(slice_idx - idx_first)]
                    if bbox is not None:
                        box[cls] = bbox
                else:
                    bbox = prompts[cls][str(slice_idx)]
                    if bbox is not None:
                        box[cls] = bbox
            segm = pr.copy()
            for k, v in box.items():
                if len(v) == 0:
                    continue
                if type(v[0]) == list:
                    v_list = v.copy()
                    for v in v_list:
                        cv2.rectangle(segm, (v[0], v[1]), (v[2], v[3]), (int(k), 0, 0), 1)
                else:
                    cv2.rectangle(segm, (v[0], v[1]), (v[2], v[3]), (int(k), 0, 0), 1)
            volume[..., slice_idx] = segm

        # store nii file
        nii = nib.Nifti1Image(volume.astype(np.uint8), affine=nii.affine)
        fn = os.path.join(args.output_folder, f"{sample_idx}_vis.nii.gz")
        nib.save(nii, fn)
