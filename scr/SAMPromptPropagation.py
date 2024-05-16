import logging
import os
import random
import time
from collections import OrderedDict

import cv2
import nibabel as nib
import numpy as np
from skimage.measure import label

from segment_anything import SamPredictor, sam_model_registry

logging.basicConfig(level=logging.INFO)


# # Fix randomness in prompt selection
# np.random.seed(1)
# random.seed(1)


def get_binary_mask(input_mask, cls):
    output = np.zeros_like(input_mask, dtype=np.uint8)
    output[input_mask == cls] = 1
    if len(output.shape) < 3:
        output = output[:, :, np.newaxis]  # height*depth*1, to consistent with multi-class setting
    if len(output.shape) > 3:
        output = np.squeeze(output)
    return output


def get_mask_with_cls_values(input_mask, cls):
    output = np.zeros_like(input_mask, dtype=np.uint8)
    output[input_mask == 1] = cls
    return output


def find_all_disconnected_regions(base_mask, remove_small_boxes=1):
    label_msk, region_ids = label(base_mask, connectivity=2, return_num=True)
    logging.debug(f'num of regions found {region_ids}')
    ratio_list, regionid_list = [], []
    # clean some region that is abnormally small (ration < 0.05 ie 5% of base mask)
    for region_id in range(1, region_ids + 1):
        # find coordinates of points in the region
        binary_msk = np.where(label_msk == region_id, 1, 0)
        # clean some region that is abnormally small
        r = np.sum(binary_msk) / np.sum(base_mask)
        logging.debug(f'curr mask over all mask ratio {r}')
        if remove_small_boxes == 1 and (r < 0.05 or np.sum(binary_msk) < 5):
            continue
        ratio_list.append(r)
        regionid_list.append(region_id)

    if len(ratio_list) == 0:
        return np.zeros_like(label_msk, dtype=np.uint8), regionid_list

    ratio_list, regionid_list = zip(*sorted(zip(ratio_list, regionid_list)))
    regionid_list = regionid_list[::-1]
    return label_msk, regionid_list


def mask_to_box_simple(mask, margin=0):
    mask = mask.squeeze()
    # find coordinates of points in the region
    row, col = np.argwhere(mask).T
    # find the four corner coordinates
    y0, x0 = row.min() - margin, col.min() - margin
    y1, x1 = row.max() + margin, col.max() + margin
    return [x0, y0, x1, y1]


def random_box(box, max_margin=0):
    r1, r2, r3, r4 = random.sample(range(0, max_margin), counts=[4] * max_margin, k=4)
    v1, v2, v3, v4 = random.sample([-1, 1], counts=[4] * 2, k=4)
    return np.array([box[0] + r1 * v1, box[1] + r2 * v2, box[2] + r3 * v3, box[3] + r4 * v4])


def generate_prompt(label_msk, regionid_list, margin=0):
    prompt = []
    for mask_idx in range(3):
        if mask_idx < len(regionid_list):
            binary_msk = np.where(label_msk == regionid_list[mask_idx], 1, 0)
            box = mask_to_box_simple(binary_msk, margin)
            prompt.append(box)
    return np.array(prompt)


def process_input_array(input_array):
    input_array = np.uint8((input_array - input_array.min()) / (np.max(input_array) - input_array.min()) * 255)
    return cv2.cvtColor(input_array, cv2.COLOR_BGR2RGB)


class SAMPredictorVolumeBaseline:
    def __init__(self, margin=0, model_type="vit_b", remove_small_boxes=True):
        self.model_type = model_type
        self.predictor = None
        self._set_sam()

        self.margin = margin
        self.remove_small_boxes = remove_small_boxes

        self.image_volume_full = None
        self.affine = None
        self.image_volume = None
        self.mask_reference_full = None
        self.mask_reference = None
        self.init_slice = None

    def _set_sam(self):
        sam = sam_model_registry[self.model_type](
            checkpoint=os.path.join(os.path.join(os.getcwd(), "sam_vit_b_01ec64.pth")))
        sam.to('cuda')
        self.predictor = SamPredictor(sam)

    def read_volume(self, file_name):
        input_image = nib.load(file_name)
        self.affine = input_image.affine
        input_image = np.asarray(input_image.get_fdata())
        input_images = np.moveaxis(input_image, -1, 0)
        self.image_volume_full = input_images

    def read_reference(self, file_name):
        input_mask = np.asarray(nib.load(file_name).get_fdata(), dtype=np.uint8)
        input_masks = np.moveaxis(input_mask, -1, 0)
        self.mask_reference_full = input_masks

    def check_for_empty_reference(self, labels_to_check):
        if set(labels_to_check).issubset(np.unique(self.mask_reference_full)):
            return True
        else:
            return False

    def set_volume(self, input_volume):
        self.image_volume_full = input_volume
        self.image_volume = input_volume

    def set_reference(self, input_reference):
        self.mask_reference_full = input_reference
        self.mask_reference = input_reference

    def select_init_slice(self, slice_idx, cls):
        self.mask_reference = get_binary_mask(self.mask_reference_full[slice_idx], cls)
        self.image_volume = self.image_volume_full.copy()
        self.init_slice = slice_idx

    def prompt_based_prediction(self, prompt):
        logging.debug(f'prompt: {prompt}')
        preds = None
        if prompt.shape[-1] != 4:
            if len(prompt) != 0:
                # logging.warning("empty prompt")
                # else:
                logging.warning("wrong prompt shape")
        else:
            if len(prompt.shape) == 1:  # one box without nested list
                preds, _, _ = self.predictor.predict(box=prompt, multimask_output=False)
            else:  # box(es) with nested list

                for box in prompt:
                    preds_single, _, _ = self.predictor.predict(box=box, multimask_output=False)
                    if preds is None:
                        preds = preds_single
                    else:
                        preds += preds_single
            preds = preds.transpose((1, 2, 0))
        return preds

    def predict_volume(self):
        if self.image_volume is None or self.mask_reference is None:
            logging.info("volume or reference mask not set.")
            return None

        logging.info("SAM predicts volume for single class ...")

        if len(self.mask_reference) == 0 or np.sum(self.mask_reference) == 0:
            logging.info(f"reference mask is empty.")
            mask_aggregated = np.zeros_like(self.image_volume, dtype=np.uint8)
            prompts_aggregated = {}
            return mask_aggregated, prompts_aggregated

        mask_aggregated = np.zeros_like(self.image_volume, dtype=np.uint8)

        image_shape = self.image_volume[0].shape

        t = time.time()

        # Perform SAM prediction for this volume in both directions
        image_volume1 = self.image_volume[self.init_slice:]  # everything starting from init slice S...N
        image_volume2 = [x for x in
                         reversed(self.image_volume[0:self.init_slice + 1])]  # everything before init slice 0...S-1

        mask_reference = self.mask_reference  # slice S

        mask1, prompts1 = self.predict_volume_one_way(image_volume1, mask_reference, image_shape)
        mask2, prompts2 = self.predict_volume_one_way(image_volume2, mask_reference, image_shape)

        mask_aggregated[self.init_slice:] = mask1
        mask_aggregated[0:self.init_slice + 1] = np.asarray([x for x in reversed(mask2)])
        prompts_aggregated = OrderedDict(sorted({self.init_slice - k: v for k, v in prompts2.items()}.items()))
        prompts_aggregated.update({k + self.init_slice: v for k, v in prompts1.items()})

        elapsed_time = time.time() - t
        prompts_aggregated["time"] = elapsed_time
        logging.info(f"... done in {elapsed_time}")

        return mask_aggregated, prompts_aggregated

    def predict_volume_one_way(self, image_volume, mask_reference, image_shape):
        if image_volume is None or mask_reference is None:
            logging.info("volume or reference mask not set.")
            return None

        # image_volume is list of slices or numpy array
        previous_mask = np.zeros(image_shape, dtype=np.uint8)
        prediction_masks = []
        prompts_full = {}
        for idx, input_array in enumerate(image_volume):
            # Set input array
            if np.max(input_array) == 0:
                logging.debug('Empty image slice, no prompt can be extracted, skipped')
                mask_binary = np.zeros_like(mask_reference, np.uint8).squeeze()
                prediction_masks.append(mask_binary)
                prompts_full[idx] = None
                continue
            self.predictor.set_image(process_input_array(input_array))

            # Get mask
            if idx == 0:
                mask_binary = mask_reference.copy()
            else:
                mask_binary = previous_mask.copy()

            if np.sum(mask_binary) == 0:
                logging.debug('Empty previous slice, no prompt can be extracted, skipped')
                prediction_masks.append(mask_binary)
                prompts_full[idx] = None
                continue

            # Find all disconnected regions
            label_msk, regionid_list = find_all_disconnected_regions(mask_binary, self.remove_small_boxes)

            # box of top-3 largest mask
            prompt = generate_prompt(label_msk, regionid_list, margin=self.margin)

            # Get output based on prompt type
            preds = self.prompt_based_prediction(prompt)
            if preds is None:
                preds_mask_single = np.zeros_like(mask_binary, np.uint8).squeeze()
            else:
                preds_mask_single = np.array(preds[:, :, 0] > 0, dtype=int)

            # Collect slice results
            prediction_masks.append(preds_mask_single)
            prompts_full[idx] = prompt.tolist()
            previous_mask = preds_mask_single

        return np.asarray(prediction_masks), prompts_full


class SAMPredictorVolumeStochastic(SAMPredictorVolumeBaseline):
    def __init__(self, margin=0, model_type="vit_b", remove_small_boxes=True, iterations=10, max_margin=5,
                 majority_vote=0.8):
        super().__init__(margin=margin, model_type=model_type, remove_small_boxes=remove_small_boxes)
        self.iterations = iterations
        self.max_margin = max_margin
        self.majority_vote = majority_vote

    def prompt_based_prediction(self, prompt):
        logging.debug(f'prompt: {prompt}')
        preds = None
        if prompt.shape[-1] != 4:
            if len(prompt) == 0:
                logging.warning("empty prompt")
            else:
                logging.warning("wrong prompt shape")
            return preds
        else:
            if len(prompt.shape) == 1:  # one box without nested list
                for _ in range(self.iterations):
                    new_box = random_box(prompt, max_margin=self.max_margin)
                    preds_single, _, _ = self.predictor.predict(box=new_box, multimask_output=False)
                    if preds is None:
                        preds = preds_single.astype(np.uint8)
                    else:
                        preds += preds_single.astype(np.uint8)
            else:  # box(es) with nested list
                for box in prompt:
                    for _ in range(self.iterations):
                        new_box = random_box(box, max_margin=self.max_margin)
                        preds_single, _, _ = self.predictor.predict(box=new_box, multimask_output=False)
                        if preds is None:
                            preds = preds_single.astype(np.uint8)
                        else:
                            preds += preds_single.astype(np.uint8)

        preds = (preds > int(self.iterations * self.majority_vote)).astype(np.uint8)
        preds = preds.transpose((1, 2, 0))
        return preds


class SAMPredictorVolumeNested(SAMPredictorVolumeBaseline):
    def __init__(self, margin=0, model_type="vit_b", remove_small_boxes=True, aggregate=True):
        super().__init__(margin=margin, model_type=model_type, remove_small_boxes=remove_small_boxes)
        self.aggregate_masks = aggregate

    def predict_volume_one_way(self, image_volume, mask_reference, image_shape):
        if image_volume is None or mask_reference is None:
            logging.info("volume or reference mask not set.")
            return None

        # image_volume is list of slices or numpy array
        previous_mask = np.zeros(image_shape, dtype=np.uint8)
        prediction_masks = []
        prompts_full = {}
        prediction_masks_collection = {}
        for idx in range(len(image_volume)):
            # Set input array for current prediction
            input_array = image_volume[idx]
            if np.max(input_array) == 0:
                logging.debug('Empty image slice, no prompt can be extracted, skipped')
                mask_binary = np.zeros_like(mask_reference, np.uint8).squeeze()
                prediction_masks.append(mask_binary)
                prompts_full[idx] = None
                continue
            self.predictor.set_image(process_input_array(input_array))

            # Get previous mask
            if idx == 0:  # first image -> take directly the reference label and generate prompt
                mask_binary = mask_reference.copy()
            else:  # otherwise take previous_mask
                mask_binary = previous_mask.copy()

            if np.sum(mask_binary) == 0:
                logging.debug('Empty previous slice, no prompt can be extracted, skipped')
                prediction_masks.append(mask_binary)
                prompts_full[idx] = None
                previous_mask = mask_binary.copy()
                continue

            # Perform prediction for current slice with previous mask
            label_msk, regionid_list = find_all_disconnected_regions(mask_binary, self.remove_small_boxes)
            prompt = generate_prompt(label_msk, regionid_list, margin=self.margin)
            preds = self.prompt_based_prediction(prompt)
            if preds is None:
                preds_mask_single1 = np.zeros_like(mask_binary, np.uint8).squeeze()
            else:
                preds_mask_single1 = np.array(preds[:, :, 0] > 0, dtype=int)

            if idx not in prediction_masks_collection.keys():
                prediction_masks_collection[idx] = [preds_mask_single1]
            else:
                prediction_masks_collection[idx] += [preds_mask_single1]

            # Make next prediction based on current slice (if not last slice and if not first slice)
            preds_mask_single2 = None
            if idx + 1 < len(image_volume) and idx != 0:
                input_array2 = image_volume[idx + 1]
                if np.max(input_array) == 0:
                    logging.debug('Empty image slice, no prompt can be extracted, no future prediction generated')
                    preds_mask_single2 = None
                else:
                    self.predictor.set_image(process_input_array(input_array2))

                    if np.sum(preds_mask_single1) != 0:  # if mask vanishes, let it be
                        label_msk2, regionid_list2 = find_all_disconnected_regions(preds_mask_single1,
                                                                                   self.remove_small_boxes)
                        prompt2 = generate_prompt(label_msk2, regionid_list2, margin=self.margin)
                        preds2 = self.prompt_based_prediction(prompt2)
                        if preds2 is not None:  # if next prediction is vanishing, let it be
                            preds_mask_single2 = np.array(preds[:, :, 0] > 0, dtype=int)
                            prediction_masks_collection[idx + 1] = [preds_mask_single2]

                        prompts_full[idx + 1] = prompt2.tolist()

            # If next prediction exists, redo procedure with next mask
            prompt3 = np.array([])
            if idx == 0:
                prompt3 = prompt.copy()
            preds_mask_single1_ = preds_mask_single1.copy()
            if preds_mask_single2 is not None and np.sum(preds_mask_single2) != 0:
                self.predictor.set_image(process_input_array(input_array))

                label_msk, regionid_list = find_all_disconnected_regions(preds_mask_single2, self.remove_small_boxes)
                prompt3 = generate_prompt(label_msk, regionid_list, margin=self.margin)
                preds = self.prompt_based_prediction(prompt3)
                if preds is None:
                    preds_mask_single1_ = np.zeros_like(mask_binary, np.uint8).squeeze()
                else:
                    preds_mask_single1_ = np.array(preds[:, :, 0] > 0, dtype=int)
                prediction_masks_collection[idx] += [preds_mask_single1_]

            # set previous mask; 3 options - first mask, last mask, combination of all masks (current one)
            if self.aggregate_masks == 1:
                v = prediction_masks_collection[idx]
                previous_mask = (np.sum(v, axis=0) == len(v)).astype(np.uint8)
                previous_prompt = [*prompt.tolist(), *prompt3.tolist()]
            else:
                previous_mask = preds_mask_single1_.copy()
                previous_prompt = prompt3.tolist()
                prompts_full.pop(idx, None)  # only keep new prompt3

            # Collect slice results
            prediction_masks.append(previous_mask)
            if idx not in prompts_full.keys():
                prompts_full[idx] = previous_prompt
            else:
                prompts_full[idx] += previous_prompt

        return np.asarray(prediction_masks), prompts_full


class SAMPredictorVolumeNestedStochastic(SAMPredictorVolumeBaseline):
    def __init__(self, margin=0, model_type="vit_b", remove_small_boxes=True, iterations=10, max_margin=5,
                 majority_vote=0.8, aggregate=True):
        super().__init__(margin=margin, model_type=model_type, remove_small_boxes=remove_small_boxes)
        self.aggregate_masks = aggregate
        self.iterations = iterations
        self.max_margin = max_margin
        self.majority_vote = majority_vote

    def prompt_based_prediction_stochastic(self, prompt):
        logging.debug(f'prompt: {prompt}')
        preds = None
        if prompt.shape[-1] != 4:
            if len(prompt) == 0:
                logging.warning("empty prompt")
            else:
                logging.warning("wrong prompt shape")
            return preds
        else:
            if len(prompt.shape) == 1:  # one box without nested list
                for _ in range(self.iterations):
                    new_box = random_box(prompt, max_margin=self.max_margin)
                    preds_single, _, _ = self.predictor.predict(box=new_box, multimask_output=False)
                    if preds is None:
                        preds = preds_single.astype(np.uint8)
                    else:
                        preds += preds_single.astype(np.uint8)
            else:  # box(es) with nested list
                for box in prompt:
                    for _ in range(self.iterations):
                        new_box = random_box(box, max_margin=self.max_margin)
                        preds_single, _, _ = self.predictor.predict(box=new_box, multimask_output=False)
                        if preds is None:
                            preds = preds_single.astype(np.uint8)
                        else:
                            preds += preds_single.astype(np.uint8)

        preds = (preds > int(self.iterations * self.majority_vote)).astype(np.uint8)
        preds = preds.transpose((1, 2, 0))
        return preds

    def predict_volume_one_way(self, image_volume, mask_reference, image_shape):
        if image_volume is None or mask_reference is None:
            logging.info("volume or reference mask not set.")
            return None

        # image_volume is list of slices or numpy array
        previous_mask = np.zeros(image_shape, dtype=np.uint8)
        prediction_masks = []
        prompts_full = {}
        prediction_masks_collection = {}
        for idx in range(len(image_volume)):
            # Set input array for current prediction
            input_array = image_volume[idx]
            if np.max(input_array) == 0:
                logging.debug('Empty image slice, no prompt can be extracted, skipped')
                mask_binary = np.zeros_like(mask_reference, np.uint8).squeeze()
                prediction_masks.append(mask_binary)
                prompts_full[idx] = None
                continue
            self.predictor.set_image(process_input_array(input_array))

            # Get previous mask
            if idx == 0:  # first image -> take directly the reference label and generate prompt
                mask_binary = mask_reference.copy()
            else:  # otherwise take previous_mask
                mask_binary = previous_mask.copy()

            if np.sum(mask_binary) == 0:
                logging.debug('Empty previous slice, no prompt can be extracted, skipped')
                prediction_masks.append(mask_binary)
                prompts_full[idx] = None
                previous_mask = mask_binary.copy()
                continue

            # Perform prediction for current slice with previous mask
            label_msk, regionid_list = find_all_disconnected_regions(mask_binary, self.remove_small_boxes)
            prompt = generate_prompt(label_msk, regionid_list, margin=self.margin)
            preds = self.prompt_based_prediction_stochastic(prompt)
            if preds is None:
                preds_mask_single1 = np.zeros_like(mask_binary, np.uint8).squeeze()
            else:
                preds_mask_single1 = np.array(preds[:, :, 0] > 0, dtype=int)

            if idx not in prediction_masks_collection.keys():
                prediction_masks_collection[idx] = [preds_mask_single1]
            else:
                prediction_masks_collection[idx] += [preds_mask_single1]

            # Make next prediction based on current slice (if not last slice and if not first slice)
            preds_mask_single2 = None
            if idx + 1 < len(image_volume) and idx != 0:
                input_array2 = image_volume[idx + 1]
                if np.max(input_array) == 0:
                    logging.debug('Empty image slice, no prompt can be extracted, no future prediction generated')
                    preds_mask_single2 = None
                else:
                    self.predictor.set_image(process_input_array(input_array2))

                    if np.sum(preds_mask_single1) != 0:  # if mask vanishes, let it be
                        label_msk2, regionid_list2 = find_all_disconnected_regions(preds_mask_single1,
                                                                                   self.remove_small_boxes)
                        prompt2 = generate_prompt(label_msk2, regionid_list2, margin=self.margin)
                        preds2 = self.prompt_based_prediction_stochastic(prompt2)
                        if preds2 is not None:  # if next prediction is vanishing, let it be
                            preds_mask_single2 = np.array(preds[:, :, 0] > 0, dtype=int)
                            prediction_masks_collection[idx + 1] = [preds_mask_single2]

                        prompts_full[idx + 1] = prompt2.tolist()

            # If next prediction exists, redo procedure with next mask
            prompt3 = np.array([])
            if idx == 0:
                prompt3 = prompt.copy()
            preds_mask_single1_ = preds_mask_single1.copy()
            if preds_mask_single2 is not None and np.sum(preds_mask_single2) != 0:
                self.predictor.set_image(process_input_array(input_array))

                label_msk, regionid_list = find_all_disconnected_regions(preds_mask_single2, self.remove_small_boxes)
                prompt3 = generate_prompt(label_msk, regionid_list, margin=self.margin)
                preds = self.prompt_based_prediction_stochastic(prompt3)
                if preds is None:
                    preds_mask_single1_ = np.zeros_like(mask_binary, np.uint8).squeeze()
                else:
                    preds_mask_single1_ = np.array(preds[:, :, 0] > 0, dtype=int)
                prediction_masks_collection[idx] += [preds_mask_single1_]

            # set previous mask; 3 options - first mask, last mask, combination of all masks (current one)
            if self.aggregate_masks == 1:
                v = prediction_masks_collection[idx]
                previous_mask = (np.sum(v, axis=0) == len(v)).astype(np.uint8)
                previous_prompt = [*prompt.tolist(), *prompt3.tolist()]
            else:
                previous_mask = preds_mask_single1_.copy()
                previous_prompt = prompt3.tolist()
                prompts_full.pop(idx, None)  # only keep new prompt3

            # Collect slice results
            prediction_masks.append(previous_mask)
            if idx not in prompts_full.keys():
                prompts_full[idx] = previous_prompt
            else:
                prompts_full[idx] += previous_prompt

        return np.asarray(prediction_masks), prompts_full
