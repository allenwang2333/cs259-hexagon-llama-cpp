#!/bin/bash

# --- CONFIGURATION ---
LLAMA_CPP_DIR="/home/linuxbrew/.linuxbrew/bin"
MODEL_F16="Qwen3VL-2B-Instruct-F16.gguf"
CALIBRATION_FILE="qwen_calibration.txt"
IMATRIX_FILE="qwen_imatrix.dat"
OUTPUT_MODEL="Qwen3-VL-2B-Instruct-q4_0-imatrix.gguf"

# Check if tools exist
if [ ! -f "$LLAMA_CPP_DIR/llama-imatrix" ]; then
    echo "Error: llama-imatrix not found in $LLAMA_CPP_DIR"
    exit 1
fi

echo "--- STEP 1: COMPUTING IMPORTANCE MATRIX (This takes time!) ---"
# -ngl 99 : Offload to GPU (remove if you don't have a GPU on your PC)
# -c 512  : Context length. 512 is plenty for calibration sentences.
# --chunks 100 : How many chunks of text to process.
$LLAMA_CPP_DIR/llama-imatrix \
    -m $MODEL_F16 \
    -f $CALIBRATION_FILE \
    -o $IMATRIX_FILE \
    --chunks 100 \
    -ngl 99 

echo "--- STEP 2: QUANTIZING WITH IMATRIX ---"
# CRITICAL: We use 'q4_0' for Snapdragon optimization.
# Do NOT use iq4_xs or other 'iq' types, they are slow on mobile DSPs.
$LLAMA_CPP_DIR/llama-quantize \
    --imatrix $IMATRIX_FILE \
    $MODEL_F16 \
    $OUTPUT_MODEL \
    q4_0

echo "--- DONE ---"
echo "New model saved as: $OUTPUT_MODEL"
echo "Transfer this file to your Android device."