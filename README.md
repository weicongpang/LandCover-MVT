# Openset Remote Sensing Image Tagging 

This repository provides a simple pipeline to perform open-set image tagging on remote sensing datasets, leveraging the Qwen2.5-VL multimodal language model and other MLLMs.

---

## Requirements

### Step 0: Setup Environment

```bash
# Clone the repository
git clone https://github.com/XXX/openset_remotesensing_tagging.git
cd openset

# Create conda environment
conda create -n openset_rs python=3.12 
conda activate openset_rs

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Install LLaMA-Factory package
cd llama_factory/LLaMA-Factory
pip install .
```

---

## Quick Start

### Step 1: Prepare Dataset

Please first download the AID dataset from https://captain-whu.github.io/AID/
Then please download dataset from https://huggingface.co/datasets/image, and **create dataset folder under the repo root folder**. 
Please use ```wget https://huggingface.co/datasets/image/resolve/main/flat_out_without_air.zip.tar.gz```
Inside the dataset folder you should see flat_out folder, root folder, and instance_object_only.json

We will do two stages fine-tuning.
For the first stage fine-tuning, please use the AID Dataset. 
For the second stage fine-tuning, please use our own Dataset.

The followings are for the second stage fine-tuning: 
Navigate to dataset processing scripts and execute them one by one (first combine.py, then rename.py, finally process_json.py):

```bash
cd dataset_processed/code

python combine.py
python rename.py
python process_json.py
```

- **combine.py**: Combines annotations with their respective images.
- **rename.py**: Renames images and updates paths.
- **process_json.py**: Converts data to Alpaca format for inference.

---

### Step 2: Run Inference using Qwen2.5-VL

Run the inference script:

```bash
cd ~/openset/llama_factory/LLaMA-Factory

python scripts/vllm_infer.py \
  --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
  --dataset rs_open_tag_infer \
  --dataset_dir /root/openset/llama_factory/LLaMA-Factory/data \
  --template qwen2_vl \
  --cutoff_len 4096 \
  --save_name /root/openset/llama_factory/LLaMA-Factory/output/qwen25vl_rs_tag_preds.jsonl \
  --enable_thinking false \
  --batch_size 64 \
  --vllm_config '{"gpu_memory_utilization":0.90,"dtype":"bfloat16"}'
```

Inference outputs are saved to:

```
/root/openset/llama_factory/LLaMA-Factory/output/qwen25vl_rs_tag_preds.jsonl
```

---
