import json
import multiprocessing
import os

from segment_anything import SamPredictor, sam_model_registry
from natsort import natsorted
import argparse
import numpy as np
import nibabel as nib
import logging

from nnunetv2.paths import nnUNet_raw
from evaluation.nnunet_evaluation_connector import compute_metrics_like_nnunet_from_arrays_simple

logging.basicConfig(level=logging.INFO)


def compute_metrics(reference_mask, prediction_mask, cls):
    reference_masks_cls = reference_mask.copy()
    reference_masks_cls[reference_masks_cls != cls] = 0
    reference_masks_cls[reference_masks_cls == cls] = 1
    prediction_masks_cls = prediction_mask.copy()
    prediction_masks_cls[prediction_masks_cls != cls] = 0
    prediction_masks_cls[prediction_masks_cls == cls] = 1
    blub = compute_metrics_like_nnunet_from_arrays_simple(reference_masks_cls, prediction_masks_cls)
    return {cls: blub}


if __name__ == '__main__':
    logging.info(" ")
    logging.info("Evaluation with Dice, IoU")
    parser = argparse.ArgumentParser(description="Evaluation of SAM generated predictions with Dice, IoU")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--label_order", required=True, nargs='+')
    parser.add_argument("--output_folder", required=False, default="tmp")
    parser.add_argument("--nnunet_suffix", required=False, default="Tr")
    parser.add_argument("--debug", required=False, default=False)
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    # Set up model
    model_type = "vit_b"
    sam = sam_model_registry[model_type](
        checkpoint=os.path.join(os.path.join(os.getcwd(), "sam_vit_b_01ec64.pth")))
    sam.to('cuda')
    predictor = SamPredictor(sam)

    # Set up dataset
    dataset = "Dataset" + args.dataset
    init_path = os.path.join(nnUNet_raw, dataset)
    logging.info(f"dataset to analyze {dataset}")

    input_img_dir = os.path.join(init_path, f"images{args.nnunet_suffix}")
    input_seg_dir = os.path.join(init_path, f"labels{args.nnunet_suffix}")

    output_dir = args.output_folder
    if not os.path.exists(output_dir):
        logging.warning("input reference directory doesn't exist.")

    mask_list = natsorted(os.listdir(input_seg_dir))
    im_list = None  # if only  subset of files should be analyzed
    class_labels = [int(x) for x in args.label_order]
    num_proc = len(class_labels)
    logging.info(f"# of samples {len(mask_list)}")

    # Logging of dataset info
    logging.info(f"input image directory {input_img_dir}")
    logging.info(f"input segmentation directory {input_seg_dir}")
    logging.info(f"input reference directory {output_dir}")
    logging.info(f"labels {class_labels}")
    logging.info(f"output directory {output_dir}")

    # iterate through names
    metrics_log = {}
    for im_idx, im_name in enumerate(mask_list):
        # Skip non-selected images if specified
        if im_list is not None:
            if im_name not in im_list:
                continue

        # Logging
        print(" ")

        # Read images, reference and prediction mask
        fn_image = os.path.join(input_img_dir, im_name.replace(".nii", "_0000.nii"))
        input_image = nib.load(fn_image)
        affine = input_image.affine
        spacing = input_image.header.get_zooms()
        input_image = np.asarray(input_image.get_fdata(), dtype=np.uint16)
        input_images = np.moveaxis(input_image, -1, 0)

        fn_reference = os.path.join(input_seg_dir, im_name)
        reference_masks = np.asarray(nib.load(fn_reference).get_fdata(), dtype=np.uint8)
        reference_masks = np.moveaxis(reference_masks, -1, 0)

        fn_prediction = os.path.join(output_dir, f'{im_name[:-7]}.nii.gz')
        if not os.path.exists(fn_prediction):
            logging.info(f"file {im_name} - no prediction")
            continue
        prediction_masks = np.asarray(nib.load(fn_prediction).get_fdata(), dtype=np.uint8)
        prediction_masks = np.moveaxis(prediction_masks, -1, 0)

        logging.info(f"file {im_name}")
        # iterate through classes
        metrics_log[im_name] = {}
        with multiprocessing.get_context("spawn").Pool(4) as pool:
            results = pool.starmap(
                compute_metrics,
                list(zip([reference_masks] * num_proc, [prediction_masks] * num_proc, class_labels))
            )

        metrics_log[im_name] = {k: v for res in results for k, v in res.items()}

    with open(os.path.join(output_dir, f'{dataset}_log.json'), "w") as file:
        json.dump(metrics_log, file)
