# 259 Project README

This repository contains scripts and utilities for running LLM-related tests, quantization, and training flows used in the cs259 project. This README documents the repository layout and shows how to run the most important workflows.

## Model Sources

This project utilizes the **Qwen3-VL** architecture. We employed the following repositories for our optimization pipeline and experimental baselines:

* **Base Model (SFT & Custom Quantization):** We used the unquantized instruct model as the foundation for our Supervised Fine-Tuning (SFT) and Importance Matrix (IMatrix) calibration:  
    [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)

* **Baselines (Ablation Studies):** To conduct performance benchmarks and ablation studies, we utilized the official GGUF implementations provided by the Qwen team:  
    [Qwen/Qwen3-VL-2B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF)

## Repository structure

- `prepare.py` - Preprocessing / setup script used before calibration or quantization steps.
- `quantize.sh` - Shell script that computes an importance matrix and creates a `q4_0` or `q8_0` quantized model.
- `qwen_calibration.txt` - Text file used as calibration input for importance-matrix computation.
- `train_qwen3_lora.py` - Training script to fine-tune QWEN3-VL with LoRA adapters.
- `cs259_project_report.pdf` — project report / write-up located at the repository root.
- `Qwen3-VL-2B-Instruct-q4_0-LORA.gguf` — quantized GGUF model (LoRA-adapted), located at the repository root.
- `cs259-hexagon-llama-cpp/` - Evaluation and utility collection (benchmarks, tests, helpers).
  - `requirement.txt` - Python dependencies for the evaluation scripts.
  - `example-vqa/run_llava.py` - Script for running model on the device.

## Quick setup

1. Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install evaluation/training dependencies (may vary per environment):

```bash
pip install -r cs259-hexagon-llama-cpp/requirement.txt
```

If you see missing packages, install them as needed (e.g., `transformers`, `datasets`, `peft`, `bitsandbytes`, `torch`).

## Running the model on a phone (Android)

You can run inference on an Android device using the helper `example-vqa/run_llava.py`. The script iterates samples from a dataset, uploads each image to the phone with `adb push`, then calls a local shell helper `run-mtmd-cli.sh` to perform remote inference on the device.

To reproduce our best performing result, you may need also utilizing water cooling for the phone.



## How to run: quantization (detailed)

The `quantize.sh` script automates two steps:

1. Compute an importance matrix using `llama-imatrix` and a calibration text file.
2. Run `llama-quantize` with the computed importance matrix to create a `q4_0` or `q8_0` quantized model.

Key variables inside `quantize.sh` (edit as needed):

- `LLAMA_CPP_DIR` — directory containing `llama-imatrix` and `llama-quantize` binaries. Default in the script is `/home/linuxbrew/.linuxbrew/bin`.
- `MODEL_F16` — input F16 model filename (default: `Qwen3VL-2B-Instruct-F16.gguf`).
- `CALIBRATION_FILE` — calibration text (default: `qwen_calibration.txt`).
- `IMATRIX_FILE` — the computed importance matrix output (default: `qwen_imatrix.dat`).
- `OUTPUT_MODEL` — final quantized model filename (default: `Qwen3-VL-2B-Instruct-q4_0-imatrix.gguf`).

Implementation notes observed in the script:

- `llama-imatrix` is run with `--chunks 100` and `-ngl 99` (the `-ngl 99` flag offloads to GPU in that binary; remove or change it if your binary expects different GPU flags or you don't have a GPU).
- `llama-quantize` is invoked with `--imatrix <file>` and `q4_0` or `q8_0` as the quantization target. The script warns not to use certain `iq*` formats for mobile DSPs since they can be slower.

Example usage (from repo root):

```bash
# make sure script is executable
chmod +x quantize.sh

# run it
./quantize.sh
```

If you need to run steps manually, the commands (mirroring the script) look like:

```bash
# Step 1: compute importance matrix
/path/to/llama-imatrix \
  -m Qwen3VL-2B-Instruct-F16.gguf \
  -f qwen_calibration.txt \
  -o qwen_imatrix.dat \
  --chunks 100 \
  -ngl 99

# Step 2: quantize using imatrix
/path/to/llama-quantize --imatrix qwen_imatrix.dat Qwen3VL-2B-Instruct-F16.gguf Qwen3-VL-2B-Instruct-q4_0-imatrix.gguf q4_0
```

## How to run: training QWEN with LoRA (detailed)

`train_qwen3_lora.py` fine-tunes `Qwen/Qwen3-VL-2B-Instruct` using PEFT/LoRA and the `textvqa` dataset. The script is configured via top-level variables (edit before running) and saves the LoRA adapter + processor to `OUTPUT_DIR`.

Important configuration values in the script (edit them if needed):

- `MODEL_ID` — default `Qwen/Qwen3-VL-2B-Instruct`.
- `OUTPUT_DIR` — where the trained LoRA adapter and processor will be saved (default `qwen3-vl-2b-lora-sft`).
- `LORA_RANK`, `LORA_ALPHA` — LoRA parameters (defaults: r=64, alpha=128).
- `BATCH_SIZE`, `GRAD_ACCUMULATION` — batch size and gradient accumulation for VRAM control.
- `LEARNING_RATE`, `MAX_SEQ_LENGTH` — optimizer and sequence length settings.
- `MIN_PIXELS`, `MAX_PIXELS` — image resolution limits used by the processor.

Training behavior / notable flags found in the script:

- The model is loaded with `torch_dtype=torch.bfloat16` and `device_map="auto"`.
- The script ensures gradients flow to embeddings (calls `enable_input_require_grads()` or registers a forward hook).
- LoRA is configured to target projection matrices like `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- `TrainingArguments` in the script use `bf16=True` and `optim="adamw_bnb_8bit"` (bitsandbytes optimizer). It also sets evaluation and save to run every 50 steps.

Run example (recommended):

```bash
# 1) Activate virtualenv
source .venv/bin/activate

# 2) Install dependencies
pip install -r cs259-hexagon-llama-cpp/requirement.txt

# (You may need to install additional packages manually, e.g. bitsandbytes)
pip install bitsandbytes peft transformers datasets accelerate

# 3) Run training
python train_qwen3_lora.py
```
