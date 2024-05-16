import logging
import time
from collections import OrderedDict

import cv2
import numpy as np

from scr.SAMPromptPropagation import SAMPredictorVolumeBaseline, get_binary_mask, find_all_disconnected_regions, \
    generate_prompt

logging.basicConfig(level=logging.INFO)


class SAMPredictorVolumeBasedOnReference(SAMPredictorVolumeBaseline):
    def __init__(self, margin=0, model_type="vit_b"):
        super().__init__(margin=margin, model_type=model_type, remove_small_boxes=0)
        self.last_slice = None

    def select_slices(self, idx_first, idx_last, cls):
        self.mask_reference = np.array(
            [get_binary_mask(x, cls).squeeze() for x in self.mask_reference_full[idx_first:idx_last + 1]])
        self.image_volume = self.image_volume_full.copy()
        self.init_slice = idx_first
        self.last_slice = idx_last

    def predict_volume(self):
        if self.image_volume is None or self.mask_reference is None:
            logging.info("volume or reference mask not set.")
            return None

        logging.info("SAM predicts volume for single class based on reference...")

        if len(self.mask_reference) == 0 or np.sum(self.mask_reference) == 0:
            logging.info("reference mask is empty.")
            mask_aggregated = np.zeros_like(self.image_volume, dtype=np.uint8)
            prompts_aggregated = {}
            return mask_aggregated, prompts_aggregated

        mask_aggregated = np.zeros_like(self.image_volume, dtype=np.uint8)
        prompts_aggregated = OrderedDict()

        image_shape = self.image_volume[0].shape

        t = time.time()

        # Perform SAM prediction for this volume bottom to top
        image_volume1 = self.image_volume[self.init_slice:self.last_slice+1]
        mask, prompts_reference = self.predict_volume_one_way(image_volume1, self.mask_reference,
                                                              image_shape)
        mask_aggregated[self.init_slice:self.last_slice+1] = mask
        prompts_aggregated.update({k + self.init_slice: v for k, v in prompts_reference.items()})

        elapsed_time = time.time() - t
        prompts_aggregated["time"] = elapsed_time
        logging.info(f"... done in {elapsed_time}")

        return mask_aggregated, prompts_aggregated

    def predict_volume_one_way(self, image_volume, mask_reference, image_shape):
        if image_volume is None or mask_reference is None:
            logging.info("volume or reference mask not set.")
            return None

        # image_volume is list of slices or numpy array
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
            input_array = np.uint8(input_array / np.max(input_array) * 255)
            input_array = cv2.cvtColor(input_array, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(input_array)

            # Get mask
            mask_binary = mask_reference[idx].copy()

            if np.sum(mask_binary) == 0:
                logging.debug('Empty previous slice, no prompt can be extracted, skipped')
                prediction_masks.append(mask_binary)
                prompts_full[idx] = None
                continue

            # Find all disconnected regions
            label_msk, regionid_list = find_all_disconnected_regions(mask_binary)

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

        return np.asarray(prediction_masks), prompts_full
