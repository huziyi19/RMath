# RMath: A Logic Reasoning-focused Datasets toward Mathematical Multistep Reasoning Tasks.

This is a code file about RMath: A Logic Reasoning-focused Datasets toward Mathematical Multistep Reasoning Tasks. There are three JSON files, representing the RMath dataset in Chinese, the RMath dataset in English and the RMath training dataset.

## Quick Start

### Set Up
Install unsloth by using conda.
```
conda create --name unsloth_env python=3.10
conda activate unsloth_env
 
conda install pytorch-cuda=<12.1/11.8> pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers
 
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

pip install --no-deps trl peft accelerate bitsandbytes
``` 

### Train
By using the code, you can train models with RMath by using unsloth.
```
from unsloth import FastLanguageModel
import torch

from trl import SFTTrainer
from transformers import TrainingArguments

max_seq_length = 2048 
dtype = None
load_in_4bit = True 
model_dir="/tora_70b" #The model_name should be replaced by your model dir.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_dir,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",    
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
    use_rslora = False,  
    loftq_config = None, 
)
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}
### Input:
{}
### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token 
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
      
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset
file_path = "./RMath_train.json"
dataset = load_dataset("json", data_files={"train": file_path}, split="train")

dataset = dataset.map(formatting_prompts_func, batched = True,)
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, 
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer_stats = trainer.train()

FastLanguageModel.for_inference(model) 
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Continue the fibonnaci sequence.", 
        "1, 1, 2, 3, 5, 8", 
        "", 
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)

model.save_pretrained("tora_70b_RMath") 

inputs = tokenizer(
[
    alpaca_prompt.format(
        "请你介绍一下张三", 
        "", 
        "", 
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 512)
```

### Inference
```
from unsloth import FastLanguageModel
import os
import torch
import json
from trl import SFTTrainer
from transformers import TrainingArguments

max_seq_length = 2048 
dtype = None 
load_in_4bit = True 

model, tokenizer = FastLanguageModel.from_pretrained(
model_name = "tora_70b_RMath", 
max_seq_length = max_seq_length,
dtype = dtype,
load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) 

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}
### Input:
{}
### Response:
{}"""

with open('RMATH_in_Chinese.json', 'r', encoding='utf-8') as file:    
    data = json.load(file)
for p in data:
	prompt=p["题面"]
	inputs = tokenizer(
	[
    	alpaca_prompt.format(
        	f"""\\{prompt}""", 
        	"", 
        	"", 
    	)
	], return_tensors = "pt").to("cuda")

	from transformers import TextStreamer
	text_streamer = TextStreamer(tokenizer)
	_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens =1024)

```
