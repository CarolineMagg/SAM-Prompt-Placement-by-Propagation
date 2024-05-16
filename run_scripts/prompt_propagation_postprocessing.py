# Based on collection from nnunet
import argparse
import json
import logging
import multiprocessing
import os
from typing import List

import nibabel as nib
import numpy as np
from acvl_utils.morphology.morphology_helper import label_with_component_sizes
from natsort import natsorted


def keep_initialized_component(segmentation, prompt, class_labels):
    mask_tmp = np.zeros_like(segmentation)
    for cls in class_labels:
        mask_binary = np.array(segmentation == cls, dtype=bool)
        prompt_binary = prompt[str(cls)]
        labeled_image, component_sizes = label_with_component_sizes(mask_binary)
        if len(component_sizes) == 1:
            mask_tmp[labeled_image == 1] = cls
            print(f"... class {cls}: selected component 1 from {component_sizes}")
        else:
            # in the first/last slice there should only be one component (the initialized one)
            # [maybe both directions is an exception?]
            if "dir2" in prompt_binary:
                labels_last_slice = np.unique(labeled_image[:, :, prompt_binary["idx_last"]])
                if len(labels_last_slice) == 2:
                    label2 = [labels_last_slice[1]]
                else:
                    box = prompt_binary["dir2"]["0"]
                    pixel_values = []
                    for b in box:
                        tmp = labeled_image[b[1]:b[3], b[0]:b[2], prompt_binary["idx_last"]]
                        for x in np.unique(tmp):
                            if x != 0:
                                pixel_values.append(x)
                    pixel_values = np.unique(pixel_values)
                    label2 = pixel_values
                for l_value in label2:
                    mask_tmp[labeled_image == l_value] = cls
            if "dir1" in prompt_binary:
                labels_first_slice = np.unique(labeled_image[:, :, prompt_binary["idx_first"]])
                if len(labels_first_slice) == 2:
                    label1 = [labels_first_slice[1]]
                else:
                    box = prompt_binary["dir1"]["0"]
                    pixel_values = []
                    for b in box:
                        tmp = labeled_image[b[1]:b[3], b[0]:b[2], prompt_binary["idx_first"]]
                        for x in np.unique(tmp):
                            if x != 0:
                                pixel_values.append(x)
                    pixel_values = np.unique(pixel_values)
                    label1 = pixel_values
                for l_value in label1:
                    mask_tmp[labeled_image == l_value] = cls
            # random slicing used as initialization
            label = []
            if "dir1" not in prompt_binary and "dir2" not in prompt_binary:
                labels_init_slice = np.unique(labeled_image[:, :, prompt_binary["idx_init"]])
                if len(labels_init_slice) == 2:
                    label = [labels_init_slice[1]]
                else:
                    box = prompt_binary[str(prompt_binary["idx_init"])]
                    pixel_values = []
                    for b in box:
                        tmp = labeled_image[b[1]:b[3], b[0]:b[2], prompt_binary["idx_init"]]
                        largest_key = max(component_sizes, key=component_sizes.get)
                        if largest_key in np.unique(tmp):  # if the largest component is in the bbox
                            pixel_values.append(largest_key)
                        else:  # else, take all the components
                            for x in np.unique(tmp):
                                if x != 0:
                                    pixel_values.append(x)
                    pixel_values = np.unique(pixel_values)
                    label = pixel_values
                for l_value in label:
                    mask_tmp[labeled_image == l_value] = cls
            print(f"... class {cls}: selected component {label} from {component_sizes}")
    return mask_tmp


def load_postprocess_save(segmentation_file: str,
                          output_fname: str,
                          prompts_fname: str,
                          class_labels: List[int]):
    nii = nib.load(segmentation_file)
    seg = np.asarray(nii.get_fdata(), dtype=np.uint8)
    with open(prompts_fname) as f:
        prompt_info = json.load(f)
    out = keep_initialized_component(seg, prompt_info, class_labels)
    write_segm(out, output_fname, nii.affine)


def write_segm(out, output_name, affine):
    nii = nib.Nifti1Image(out, affine=affine)
    print(f"save {output_name}")
    nib.save(nii, output_name)


def apply_postprocessing_to_folder(input_folder: str,
                                   output_folder: str,
                                   class_labels: List[int],
                                   num_processes=8,
                                   file_name_filter=None) -> None:
    os.makedirs(output_folder, exist_ok=True)
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        files = natsorted([x for x in os.listdir(input_folder) if ".nii.gz" in x])
        files_prompts = natsorted([x for x in os.listdir(input_folder) if ".json" in x])
        if file_name_filter is not None:
            for filter_ in file_name_filter:
                files = [f for f in files if filter_ not in f]
                files_prompts = [f for f in files_prompts if filter_ not in f]
        print(f"need to process {len(files)} files")
        _ = p.starmap(load_postprocess_save,
                      zip(
                          [os.path.join(input_folder, i) for i in files],
                          [os.path.join(output_folder, i) for i in files],
                          [os.path.join(input_folder, i) for i in files_prompts],
                          [class_labels] * len(files)
                      )
                      )


if __name__ == '__main__':
    logging.info(" ")
    logging.info("Postprocessing")
    parser = argparse.ArgumentParser('Apply postprocessing to input folder.')
    parser.add_argument('--input_folder', type=str, required=True, help='Input folder')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder')
    parser.add_argument('--label_order', required=True, nargs='+', help='Labels to be postprocessed')
    parser.add_argument("--dataset", required=False)
    parser.add_argument("--nnunet_suffix", required=False, default="Tr")

    args = parser.parse_args()

    # pp_fns, pp_fn_kwargs = load_pickle(args.pp_pkl_file)
    apply_postprocessing_to_folder(args.input_folder, args.output_folder,
                                   class_labels=[int(x) for x in args.label_order],
                                   file_name_filter=["vis", "log"])
