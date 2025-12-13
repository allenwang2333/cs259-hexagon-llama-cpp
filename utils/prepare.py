import os
from datasets import load_dataset

# 1. Config
OUTPUT_FILE = "qwen_calibration.txt"
dataset = load_dataset("lmms-lab/textvqa", split="validation[:2000]")

# Qwen Chat Format
# <|im_start|>user
# Question
# <|im_end|>
# <|im_start|>assistant
# Answer
# <|im_end|>

print(f"Generating calibration data from {len(dataset)} samples...")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for sample in dataset:
        question = sample['question']
        # TextVQA has multiple answers, we pick the most common one or the first one
        answer = sample['answers'][0] 
        
        # We simulate a conversation. 
        # Note: We skip the visual tokens for calibration to focus on language reasoning.
        text = (
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n{answer}<|im_end|>\n"
        )
        
        f.write(text)

print(f"Done! Saved to {OUTPUT_FILE}")