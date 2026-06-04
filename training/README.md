# Complete pipeline for fine-tuning Qwen2.5-VL 32B using QLoRA on a single A100 80GB.

## Files

* `train.py` - QLoRA fine-tuning. Loads base model in 4-bit, trains LoRA adapter
* `merge.py` - Merges adapter into base
* `test.py` - test
* `chat.py` - Interactive multi-turn chat loop with history
* `ask.py` - One-shot CLI: pass question as arg, get response. Slow due to fresh load each call. (UNUSED)

## Environment

Tested on GPU pod with:
- A100 SXM4 80GB
- PyTorch 2.5.1 + CUDA 12.4
- transformers 4.49.0
- peft 0.14.0
- trl 0.13.0
- bitsandbytes 0.45.0
- accelerate 1.3.0
- datasets 3.2.0

## Setup

```bash
# Install dependencies
pip install -U torch==2.5.1 torchvision==0.20.1
pip install -U "transformers==4.49.0" "peft==0.14.0" "trl==0.13.0" "bitsandbytes==0.45.0" "accelerate==1.3.0" "datasets==3.2.0"

# HuggingFace auth (token)
huggingface-cli login

# Download base model to Network Volume
huggingface-cli download Qwen/Qwen2.5-VL-32B-Instruct \
    --local-dir /workspace/qwen-base
```
# Upload dataset from local machine (or Jupyter)
```powershell
scp -P <port> example_training_data.jsonl root@<pod-ip>:/workspace/
```

## Training

```bash
python3 train.py 2>&1 | tee training.log
```

Approximately 50-70 minutes on one A100 SXM for 700 examples × 3 epochs.
Expected loss curve: ~3.85 (step 1) → ~0.57 (final).

## Testing

```bash
# sanity check with fixed prompts
python3 test.py

# chat test
python3 chat.py
```

## Inference Configuration

The `chat.py` and `ask.py` scripts include sampling parameters that affect
output quality:

- `temperature=0.7`
- `top_p=0.9`
- `repetition_penalty=1.15`
- `no_repeat_ngram_size=4`

These are the current values. They will be altered in the future as further testing and development takes place.

