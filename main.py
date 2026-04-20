from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from onixinit import Onix
import time

class ChatRequest(BaseModel):
     prompt: str

app = FastAPI()

app.add_middleware(
     CORSMiddleware,
     allow_origins=["*"],
     allow_methods=["*"],
     allow_headers=["*"]
)

onix = Onix()

@app.get("/")
async def index():
   return FileResponse("index.html")

@app.get("/status")
async def status():
    memory = onix.get_memory_usage()
    return {
        "model": onix.model_name,
        "device": onix.device,
        "vram_gb": memory
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    start_time = time.time()
    
    response = onix.generate_response(request.prompt)
    
    generation_time = time.time() - start_time
    
    return {
        "response": response,
        "generation_time": round(generation_time, 2)}