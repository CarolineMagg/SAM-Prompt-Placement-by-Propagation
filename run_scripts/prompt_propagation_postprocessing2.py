import argparse
import json
import logging
import os
from itertools import product
from typing import List

import nibabel as nib
import numpy as np
from natsort import natsorted


def remove_small_box_predictions(segmentation, prompt, class_labels):
    mask_tmp = np.zeros_like(segmentation, dtype=np.uint8)
    pixel_sizes_per_class = {}
    reference_size = {}
    for cls in class_labels:
        # get size of bbox
        prompt_list = prompt[str(cls)].copy()
        idx_init = prompt_list.pop("idx_init")
        time = prompt_list.pop("time")
        sizes_per_class = {}
        box = prompt_list[str(idx_init)][0]
        ref_size = (box[2] - box[0]) * (box[3] - box[1])
        reference_size[cls] = ref_size
        thres_size = ref_size * 0.10
        for k, p in prompt_list.items():
            if p is not None:
                px_size = 0
                for box in p:
                    px_size += (box[2] - box[0]) * (box[3] - box[1])
                sizes_per_class.update({k: px_size})
        pixel_sizes_per_class[cls] = sizes_per_class

        # remove bbox too small
        smaller_than_thres = [k for k, v in sizes_per_class.items() if v < thres_size]
        part1 = [int(x) for x in smaller_than_thres if int(x) > idx_init]
        part2 = [int(x) for x in smaller_than_thres if int(x) < idx_init]
        remove_all_below, remove_all_above = 0, mask_tmp.shape[-1]
        if len(part1) != 0:
            remove_all_above = min(product(part1, [idx_init]), key=lambda t: abs(t[0] - t[1]))[0]
        if len(part2) != 0:
            remove_all_below = min(product(part2, [idx_init]), key=lambda t: abs(t[0] - t[1]))[0]
        segmentation_processed = np.asarray(segmentation == cls, dtype=np.uint8)
        segmentation_processed[..., :remove_all_below] = 0
        segmentation_processed[..., remove_all_above:] = 0
        mask_tmp[segmentation_processed == 1] = cls
        print(f"... class {cls}: boundary slices with boxes too small are {remove_all_below}, {remove_all_above}")
    return mask_tmp


def load_postprocess_save(segmentation_file: str,
                          output_fname: str,
                          prompts_fname: str,
                          class_labels: List[int]):
    nii = nib.load(segmentation_file)
    seg = np.asarray(nii.get_fdata(), dtype=np.uint8)
    with open(prompts_fname) as f:
        prompt_info = json.load(f)
    out = remove_small_box_predictions(seg, prompt_info, class_labels)
    write_segm(out, output_fname, nii.affine)


def write_segm(out, output_name, affine):
    nii = nib.Nifti1Image(out, affine=affine)
    print(f"save {output_name}")
    nib.save(nii, output_name)


def apply_postprocessing_to_folder(input_folder: str,
                                   input_folder_prompt: str,
                                   output_folder: str,
                                   class_labels: List[int],
                                   num_processes=4,
                                   file_name_filter=None) -> None:
    os.makedirs(output_folder, exist_ok=True)
    # with multiprocessing.get_context("spawn").Pool(num_processes) as p:
    files = natsorted([x for x in os.listdir(input_folder) if ".nii.gz" in x])
    files_prompts = natsorted([x for x in os.listdir(input_folder_prompt) if "_prompts.json" in x])
    if file_name_filter is not None:
        for filter_ in file_name_filter:
            files = [f for f in files if filter_ not in f]
            files_prompts = [f for f in files_prompts if filter_ not in f]
    print(f"need to process {len(files)} files")
    assert len(files) == len(
        files_prompts), f"not same number of mask {len(files)} and prompt {len(files_prompts)} files"
    # _ = p.starmap(load_postprocess_save,
    #               zip(
    #                   [os.path.join(input_folder, i) for i in files],
    #                   [os.path.join(output_folder, i) for i in files],
    #                   [os.path.join(input_folder_prompt, i) for i in files_prompts],
    #                   [class_labels] * len(files)
    #               )
    #               )
    for filename, prompt_file in zip(files, files_prompts):
        load_postprocess_save(os.path.join(input_folder, filename),
                              os.path.join(output_folder, filename),
                              os.path.join(input_folder_prompt, prompt_file),
                              class_labels)


if __name__ == '__main__':
    logging.info(" ")
    logging.info("Postprocessing")
    parser = argparse.ArgumentParser('Apply postprocessing to input folder.')
    parser.add_argument('--input_folder', type=str, required=True, help='Input folder')
    parser.add_argument("--input_folder_prompt", required=False, default=None)
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder')
    parser.add_argument('--label_order', required=True, nargs='+', help='Labels to be postprocessed')
    parser.add_argument("--dataset", required=False)
    parser.add_argument("--nnunet_suffix", required=False, default="Tr")

    args = parser.parse_args()
    if args.input_folder_prompt is None:
        args.input_folder_prompt = args.input_folder_prompt

    # pp_fns, pp_fn_kwargs = load_pickle(args.pp_pkl_file)
    apply_postprocessing_to_folder(args.input_folder, args.input_folder_prompt, args.output_folder,
                                   class_labels=[int(x) for x in args.label_order],
                                   file_name_filter=["vis", "log"])
