"""
RunPod serverless handler for Onix.

Loads Qwen2.5-VL 32B base + the Onix LoRA adapter from the Network Volume,
then serves chat completions. Supports streaming token-by-token output.

The Network Volume is mounted at /runpod-volume in serverless workers
(NOT /workspace as in pods — this is a key difference).

Expected input format (job["input"]):
{
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
    ],
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "stream": true
}
"""

import torch
import runpod
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer
from peft import PeftModel
from threading import Thread

# In serverless, the Network Volume mounts at /runpod-volume
BASE_PATH = "/runpod-volume/qwen-base"
ADAPTER_PATH = "/runpod-volume/onix-adapter-final"

DEFAULT_SYSTEM = (
    "You are Onix, a multimodal AI assistant. You are analytical, sharp, "
    "and casual. You acknowledge the user's request briefly before diving "
    "into substance. You see images, read documents, analyze data, and "
    "reason clearly. You don't over-explain, you don't hedge excessively, "
    "and you keep things direct. When you need to use tools like image "
    "generation or code execution, you say what you're doing and do it."
)

# --- Load model once at cold start (stays warm for subsequent requests) ---
print("Loading base model...")
base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print("Loading Onix adapter...")
model = PeftModel.from_pretrained(base, ADAPTER_PATH)
model.eval()
processor = AutoProcessor.from_pretrained(BASE_PATH)
print("Model ready.")


def build_inputs(messages):
    """Apply chat template and tokenize."""
    # Ensure a system prompt is present
    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": DEFAULT_SYSTEM}] + messages
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return processor(text=[text], return_tensors="pt").to(model.device)

def stream_generate(gen_kwargs):
    streamer = TextIteratorStreamer(
        processor.tokenizer,skip_prompt=True,skip_special_tokens=True,
    )
    gen_kwargs["streamer"] = streamer
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()
    for token in streamer:
        if token:
            yield {"token": token}
    thread.join()


def handler(job):
    """
    RunPod serverless entrypoint.
    Yields tokens as they generate (streaming) when stream=True,
    otherwise returns the full response.
    """
    job_input = job["input"]
    messages = job_input.get("messages", [])
    max_new_tokens = job_input.get("max_new_tokens", 512)
    temperature = job_input.get("temperature", 0.7)
    top_p = job_input.get("top_p", 0.9)
    stream = job_input.get("stream", True)

    if not messages:
        return {"error": "No messages provided"}

    inputs = build_inputs(messages)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.15,
        no_repeat_ngram_size=4,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

    if stream:
        return stream_generate(gen_kwargs)
    else:
        # Non-streaming path — return full text
        with torch.no_grad():
            output = model.generate(**gen_kwargs)
        response = processor.decode(
            output[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return {"response": response}


# Start the serverless worker.
# return_aggregate_stream=True lets the streaming generator work over HTTP.
runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True,
})
