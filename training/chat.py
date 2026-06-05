import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

base_path = "/workspace/qwen-base"
adapter_path = "/workspace/onix-adapter-final"

print("Loading base model... (about 3 min)")
base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_path, torch_dtype=torch.bfloat16, device_map="auto"
)

print("Loading Onix adapter...")
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

history = [{"role": "system", "content": system}]
print(
    "\nOnix ready. Type your message. "
    "Type 'exit' to quit, 'reset' to clear history.\n"
)

while True:
    user_input = input("YOU: ").strip()
    if not user_input:
        continue
    if user_input.lower() == "exit":
        break
    if user_input.lower() == "reset":
        history = [{"role": "system", "content": system}]
        print("[history cleared]\n")
        continue

    history.append({"role": "user", "content": user_input})

    text = processor.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15,        # discourages repetitive phrasing
        no_repeat_ngram_size=4,         # forbids repeating 4-token sequences
    )

    response = processor.decode(
        output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()

    history.append({"role": "assistant", "content": response})
    print(f"\nONIX: {response}\n")
