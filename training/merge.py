import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

base_path = "/workspace/qwen-base"
adapter_path = "/workspace/onix-adapter-final"
output_path = "/workspace/onix-final"

print("Loading base model in bfloat16...")
base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("Loading adapter...")
model = PeftModel.from_pretrained(base, adapter_path)

print("Merging adapter into base weights...")
merged = model.merge_and_unload()

print("Saving merged model...")
merged.save_pretrained(output_path, safe_serialization=True, max_shard_size="5GB")

processor = AutoProcessor.from_pretrained(base_path)
processor.save_pretrained(output_path)

print(f"Done. Merged model saved to {output_path}")
