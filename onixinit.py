

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import warnings
warnings.filterwarnings('ignore')


class Onix:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
        print(f"Initializing Onix with {model_name}...")
        
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.device == "cpu":
            print("No GPU detected.")
        else:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU detected: {gpu_name} ({gpu_memory:.1f}GB VRAM)")
        
        # Configure 4-bit // can change to 8 or 16 later
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Load tokenizer
        print(" Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Load model with 4-bit quantization
        print(" Loading model (this takes 30-60 seconds first time)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            max_memory={0: "5GB", "cpu": "30GB"}
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Onix initialized")
        print(f"Model loaded on: {self.device}")
        
        if self.device == "cuda":
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            print(f"VRAM used: {memory_allocated:.2f}GB")
    
    
    def generate_response(self, prompt, max_new_tokens=512, temperature=0.7):
        
        # Format prompt for Mistral Instruct
        formatted_prompt = f"[INST] {prompt} [/INST]"
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=200
        )
        
        # Move to GPU
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from response (Mistral includes it)
        response = response.replace(formatted_prompt, "").strip()
        
        return response
    
    
    def chat(self, message):
        """
        Simple chat interface
        
        Args:
            message: User's chat message
        
        Returns:
            AI response
        """
        return self.generate_response(message)
    
    
    def get_memory_usage(self):
        """Get current GPU memory usage"""
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            return {
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2)
            }
        return {"allocated_gb": 0, "reserved_gb": 0}


# Test if run directly
if __name__ == "__main__":
    print("=" * 60)
    print("Onix Standalone Test")
    print("=" * 60)
    
    # Initialize
    onix = Onix()
    
    # Test prompt
    test_prompt = "What are 3 effective strategies for acquiring new clients through SEO?"
    print(f"\n Prompt: {test_prompt}")
    print("\n Onix response:")
    print("-" * 60)
    
    response = onix.chat(test_prompt)
    print(response)
    
    print("-" * 60)
    print(f"\n Memory usage: {onix.get_memory_usage()}")
    print("\n ")
    print("\n ")
    print("-" * 60)
    print("\n Test complete!")
    print("-" * 60)