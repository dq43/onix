import sys
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

base_path = "/workspace/qwen-base"
adapter_path = "/workspace/onix-adapter-final"

if len(sys.argv) < 2:
    print("Usage: python3 ask.py 'your question here'")
    sys.exit(1)

user_msg = " ".join(sys.argv[1:])

base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_path, torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base, adapter_path)
processor = AutoProcessor.from_pretrained(base_path)

system = (
    "You are Onix, a multimodal AI assistant. You are analytical, sharp, "
    "and casual. You acknowledge the user's request briefly before diving "
    "into substance. You see images, read documents, analyze data, and "
    "reason clearly. You don't over-explain, you don't hedge excessively, "
    "and you keep things direct. When you need to use tools like image "
    "generation or code execution, you say what you're doing and do it."
)

messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": user_msg},
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = processor(text=[text], return_tensors="pt").to(model.device)

output = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.15,
)

print(processor.decode(
    output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
))
