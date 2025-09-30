import torch
from task_vectors import TaskVector
import os
# from eval import eval_single_dataset
# from args import parse_arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_checkpoint', type=str, default='Qwen/Qwen2.5-1.5B-Instruct', help='Path to the pretrained checkpoint')
parser.add_argument('--finetuned_checkpoint', type=str, default='LLaMA-Factory/saves/debias_task_vector/Qwen2.5-0.5B-Instruct', help='Path to the finetuned checkpoint')
parser.add_argument('--output_dir', type=str, default='LLaMA-Factory/saves/debias_task_vector/Qwen2.5-0.5B-Instruct/tv', help='Output directory to save the task vector')

args = parser.parse_args()



# Create the task vector
task_vector = TaskVector(args.pretrained_checkpoint, args.finetuned_checkpoint)
# Negate the task vector
neg_task_vector = -task_vector
# Apply the task vector
tv = neg_task_vector.apply_to(args.pretrained_checkpoint, scaling_coef=0.5)
os.makedirs(args.output_dir, exist_ok=True)
tv.save_pretrained(args.output_dir)
print("Down")
