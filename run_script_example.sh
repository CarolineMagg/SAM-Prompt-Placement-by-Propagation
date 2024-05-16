#!/bin/bash

CUDA_VISIBLE_DEVICES=7 python -m run_scripts.prompt_reference --dataset 500_TotalSegmentatorBoneSimple --label_order 2 3 4 5 \
   --output_folder /home/caroline/data/sam/test/reference --nnunet_suffix Ts
CUDA_VISIBLE_DEVICES=7 python -m run_scripts.prompt_propagation --dataset 500_TotalSegmentatorBoneSimple --label_order 2 3 4 5 \
   --output_folder /home/caroline/data/sam/test/nested_default --experiment nested --nnunet_suffix Ts
CUDA_VISIBLE_DEVICES=7 python -m run_scripts.prompt_propagation_evaluation --dataset 500_TotalSegmentatorBoneSimple --label_order 2 3 4 5 \
   --output_folder /home/caroline/data/sam/test/nested_default --nnunet_suffix Ts
CUDA_VISIBLE_DEVICES=7 python -m run_scripts.prompt_propagation_evaluation2 --dataset 500_TotalSegmentatorBoneSimple --label_order 2 3 4 5 \
   --output_folder /home/caroline/data/sam/test/nested_default --nnunet_suffix Ts
CUDA_VISIBLE_DEVICES=7 python -m run_scripts.create_box_prompt_visualizations --dataset 500_TotalSegmentatorBoneSimple --label_order 2 3 4 5 \
   --output_folder /home/caroline/data/sam/test/nested_default --nnunet_suffix Ts
CUDA_VISIBLE_DEVICES=7 python -m run_scripts.prompt_propagation_postprocessing --dataset 500_TotalSegmentatorBoneSimple --label_order 2 3 4 5 \
   --input_folder /home/caroline/data/sam/test/nested_default --output_folder /home/caroline/data/sam/test/nested_default_pp
CUDA_VISIBLE_DEVICES=7 python -m run_scripts.prompt_propagation_evaluation --dataset 500_TotalSegmentatorBoneSimple --label_order 2 3 4 5 \
   --output_folder /home/caroline/data/sam/test/nested_default_pp --nnunet_suffix Ts
CUDA_VISIBLE_DEVICES=7 python -m run_scripts.prompt_propagation_evaluation2 --dataset 500_TotalSegmentatorBoneSimple --label_order 2 3 4 5 \
   --output_folder /home/caroline/data/sam/test/nested_default_pp --nnunet_suffix Ts
CUDA_VISIBLE_DEVICES=7 python -m run_scripts.prompt_propagation_postprocessing2 --dataset 500_TotalSegmentatorBoneSimple --label_order 2 3 4 5 \
   --input_folder /home/caroline/data/sam/test/nested_default_pp --input_folder_prompt /home/caroline/data/sam/test/nested_default \
   --output_folder /home/caroline/data/sam/test/nested_default_pp2
CUDA_VISIBLE_DEVICES=7 python -m run_scripts.prompt_propagation_evaluation --dataset 500_TotalSegmentatorBoneSimple --label_order 2 3 4 5 \
    --output_folder /home/caroline/data/sam/test/nested_default_pp2 --nnunet_suffix Ts
CUDA_VISIBLE_DEVICES=7 python -m run_scripts.prompt_propagation_evaluation2 --dataset 500_TotalSegmentatorBoneSimple --label_order 2 3 4 5 \
    --output_folder /home/caroline/data/sam/test/nested_default_pp2 --nnunet_suffix Ts
