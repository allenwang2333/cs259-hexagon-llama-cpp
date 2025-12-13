import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    TrainingArguments,
    Trainer,
    Qwen3VLForConditionalGeneration
)

from peft import LoraConfig, get_peft_model, TaskType
from qwen_vl_utils import process_vision_info
import os
import random

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct" 
OUTPUT_DIR = "qwen3-vl-2b-lora-sft"

# 0. SYSTEM PROMPT
SYSTEM_PROMPT = "Answer the question with a single word or short phrase. Do not provide explanations."

# 1. Capacity
LORA_RANK = 64
LORA_ALPHA = 128

# 2. Speed vs VRAM
BATCH_SIZE = 2           
GRAD_ACCUMULATION = 16    
LEARNING_RATE = 5e-5
MAX_SEQ_LENGTH = 1536    

# 3. Resolution Limits
MIN_PIXELS = 224 * 224
MAX_PIXELS = 560 * 560 
# ==========================================

# --- 1. Load Model & Processor ---
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)

# --- CRITICAL FIX: Ensure Gradients Flow ---
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
else:
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)
    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

processor = AutoProcessor.from_pretrained(MODEL_ID)

# --- 2. Configure LoRA ---
peft_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ],
    task_type=TaskType.CAUSAL_LM,
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# --- 3. Dataset & Formatting ---
def format_data_batch(examples):
    output_messages = []
    output_user_prompts = [] 
    
    for i in range(len(examples['question'])):
        image = examples['image'][i]
        question = examples['question'][i]
        answers = examples['answers'][i] if 'answers' in examples else []
        answer = max(set(answers), key=answers.count) if answers else "Unknown"
        
        # 1. Full Conversation
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant", 
                "content": [{"type": "text", "text": answer}]
            }
        ]
        
        # 2. Context Only (for masking)
        user_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        output_messages.append(messages)
        output_user_prompts.append(user_messages)
        
    return {"messages": output_messages, "user_prompts": output_user_prompts}

print("Loading datasets...")
raw_dataset = load_dataset("textvqa", trust_remote_code=True)
train_dataset = raw_dataset["train"]
eval_dataset = raw_dataset["validation"]

print("Formatting datasets...")
train_dataset.set_transform(format_data_batch)
eval_dataset_sub = eval_dataset.select(range(200))
eval_dataset_sub.set_transform(format_data_batch)

# --- 4. Robust Data Collator ---
def data_collator(examples):
    messages = [x["messages"] for x in examples]
    user_prompts = [x["user_prompts"] for x in examples]
    
    # A. Process Full Text
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        for msg in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    
    batch = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS
    )
    
    # B. Process Context Only (WITH IMAGES)
    user_texts = [
        processor.apply_chat_template(u_msg, tokenize=False, add_generation_prompt=True)
        for u_msg in user_prompts
    ]
    user_image_inputs, user_video_inputs = process_vision_info(user_prompts)
    
    user_batch = processor(
        text=user_texts,
        images=user_image_inputs,
        videos=user_video_inputs,
        padding=True,
        return_tensors="pt",
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        min_pixels=MIN_PIXELS, 
        max_pixels=MAX_PIXELS
    )
    
    # C. Create Labels with Masking
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    for i in range(len(labels)):
        if i < len(user_batch["attention_mask"]):
            prompt_len = user_batch["attention_mask"][i].sum().item()
        else:
            prompt_len = 0
            
        if prompt_len < labels.shape[1]:
            labels[i, :prompt_len] = -100
            
    batch["labels"] = labels
    return batch

# --- 5. Trainer ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    bf16=True,
    logging_steps=10,
    
    # UPDATED: Evaluate every 50 steps so you can see the result immediately
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    
    report_to="none",
    remove_unused_columns=False,
    warmup_steps=100,
    gradient_checkpointing=False,
    optim="adamw_bnb_8bit",
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset_sub, 
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()

print(f"Saving LoRA adapter to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

# ==========================================
# 6. RAW vs FINE-TUNED COMPARISON
# ==========================================
print("\n" + "="*50)
print("RUNNING FINAL VERIFICATION: Raw vs Fine-Tuned")
print("="*50)

model.eval()
torch.cuda.empty_cache()

def generate_answer(model_instance, image, question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    with torch.no_grad():
        generated_ids = model_instance.generate(**inputs, max_new_tokens=128)
        
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

indices = random.sample(range(len(eval_dataset)), 3)

for idx in indices:
    sample = eval_dataset[idx]
    img = sample['image']
    q = sample['question']
    gt = sample['answers']
    
    print(f"\n[Sample {idx}]")
    print(f"Q: {q}")
    print(f"Ground Truths: {gt}")
    
    ft_answer = generate_answer(model, img, q)
    print(f"Fine-Tuned Model: \033[92m{ft_answer}\033[0m")
    
    with model.disable_adapter():
        raw_answer = generate_answer(model, img, q)
        print(f"Raw Base Model:   \033[93m{raw_answer}\033[0m")
    
    print("-" * 50)