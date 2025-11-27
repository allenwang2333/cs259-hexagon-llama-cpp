import torch
import os
import sys
import random
import evaluate
import subprocess
import re
import json
import numpy as np
import shlex
from tqdm import tqdm
from argparse import ArgumentParser
from datasets import load_dataset, concatenate_datasets
from PIL import Image  # <--- Added for resizing

# Assuming these utils exist in your folder structure
from utils.data_utils import load_yaml, construct_prompt, save_json, process_single_sample, CAT_SHORT2LONG
from utils.model_utils import call_llava_engine_df, llava_image_processor
from utils.eval_utils import parse_multi_choice_response, parse_open_response

def parse_output(output):
    # Try to extract the JSON or text response. 
    # Adjust this based on your specific llama-cli output format
    try:
        # Sometimes llama-cli prints system info first. We look for the generated text.
        # This is a heuristic; might need tuning based on your logs.
        if "<|im_start|>assistant" in output:
            output = output.split("<|im_start|>assistant")[-1]
        
        # Remove trailing system stats if they appear in stdout
        output = output.split("load time =")[0] 
        return output.strip()
    except Exception as e:
        return output.strip()

def main():
    parser = ArgumentParser()
    parser.add_argument('--output_path', type=str, default='qwen_val.json', help='name of saved json')
    parser.add_argument('--data_path', type=str, default="lmms-lab/textvqa") 
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--seed', type=int, default=42)
    # Added argument to specify shell script path easily
    parser.add_argument('--script_path', type=str, default="./run-mtmd-cli.sh")

    args = parser.parse_args()

    # Dataset Loading Logic
    if "MMMU" in args.data_path:
        sub_dataset_list = []
        for subject in CAT_SHORT2LONG.values():
            sub_dataset = load_dataset(args.data_path, subject, split=args.split)
            sub_dataset_list.append(sub_dataset)
            break # Remove break to load full dataset
        dataset = concatenate_datasets(sub_dataset_list)
        question_key = "final_input_prompt"
    elif "textvqa" in args.data_path:
        dataset = load_dataset("lmms-lab/textvqa", split="validation")
        # dataset = dataset.select(range(10)) # Uncomment for testing small batch
        question_key = "question"
    else:
        raise NotImplementedError

    REMOTE_IMG = "/data/local/tmp/image.jpg"
    
    # Metrics containers
    patterns = {
        "load_time_ms": r"load time\s*=\s*([\d.]+)\s*ms",
        "prompt_eval_time_per_token_ms": r"prompt eval time\s*=\s*[\d.]+\s*ms\s*/\s*\d+\s*tokens\s*\(\s*([\d.]+)\s*ms per token",
        "eval_time_per_token_ms": r"eval time\s*=\s*[\d.]+\s*ms\s*/\s*\d+\s*runs\s*\(\s*([\d.]+)\s*ms per token",
        "total_time_ms": r"total time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens"
    }
    
    metrics = {k: [] for k in patterns.keys()}
    metrics["total_tokens"] = []
    correct = []
    rouge_score = []
    rouge = evaluate.load("rouge")

    print(f"Starting evaluation on {len(dataset)} samples...")

    for sample in tqdm(dataset):
        # MMMU Pre-processing
        if "MMMU" in args.data_path:
            sample = process_single_sample(sample)
            sample = construct_prompt(sample, args.config)

        if sample['image']:
            # --- 1. OPTIMIZED IMAGE HANDLING ---
            image = sample['image'].convert("RGB")
            
            # Smart Resize: Cap max dimension to 1280 (Safe for TextVQA)
            # This prevents 4K images from crashing the edge device or causing 10s latency
            max_dim = 1280
            if max(image.size) > max_dim:
                image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            
            image.save("image.jpg", "JPEG", quality=95)
            # -----------------------------------

            # --- 2. QWEN CHAT TEMPLATE ---
            raw_q = sample[question_key]
            # Manually apply Qwen ChatML format
            prompt = (
                "<|im_start|>system\n"
                "You are a helpful assistant. Read the text in the image carefully.<|im_end|>\n"
                "<|im_start|>user\n"
                f"{raw_q}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            # -----------------------------

            # Sanitize for CLI
            prompt_safe = prompt.replace('"', '\\"').replace("`", "\\`").replace("$", "\\$")

            # Push image
            subprocess.run(["adb", "push", "image.jpg", REMOTE_IMG], check=True, stdout=subprocess.DEVNULL)

            try:
                # Use git bash path from your original code, or standard bash if on Linux/Mac
                shell_cmd = ["C:\Program Files\Git\\bin\\bash.exe", args.script_path, "--image", REMOTE_IMG, "--prompt", prompt_safe]
                
                out = subprocess.run(
                    shell_cmd,
                    text=True, capture_output=True, check=True, encoding='utf-8'
                )

                # Output Parsing
                output_text = parse_output(out.stdout)
                
                # Accuracy Check (Exact Match)
                match = 0
                for answer in sample["answers"]:
                    # Simple normalization for comparison
                    if answer.lower().strip() in output_text.lower().strip():
                        match = 1
                        break
                correct.append(match)

                # Performance Metrics Parsing
                found_metrics = {}
                for key, pattern in patterns.items():
                    m = re.search(pattern, out.stderr)
                    if m:
                        val = float(m.group(1))
                        metrics[key].append(val)
                        if key == "total_time_ms":
                            metrics["total_tokens"].append(int(m.group(2)))
                
                # Rouge Score
                score = rouge.compute(predictions=[output_text], references=[sample["answers"]])
                rouge_score.append(score["rougeL"])

                # Optional: Live Print
                # print(f"GT: {sample['answers']} | Pred: {output_text} | Match: {match}")

            except subprocess.CalledProcessError as e:
                print(f"Error processing sample: {e.stderr}", file=sys.stderr)
                continue

    # Final Stats
    print("\n--- RESULTS ---")
    print(f"Average Correct (Exact Match): {np.mean(correct):.4f}")
    print(f"Average Rouge-L: {np.mean(rouge_score):.4f}")
    
    if len(metrics["total_time_ms"]) > 0:
        avg_total_time = np.mean(metrics["total_time_ms"])
        avg_tokens = np.mean(metrics["total_tokens"])
        print(f"Avg Latency (Total Time): {avg_total_time:.2f} ms")
        print(f"Avg Speed: {avg_tokens / (avg_total_time/1000):.2f} tokens/sec")

if __name__ == '__main__':
    main()