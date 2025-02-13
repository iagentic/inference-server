from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
import asyncio
import websockets
import json
import torch
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from faster_whisper import WhisperModel
import numpy as np
import shutil
from typing import List

app = FastAPI(
    title="OpenAI-Compatible FastAPI Server",
    version="0.1.0",
    openapi_version="3.1.0"
)

# Model directories
MODEL_DIR = Path("./models")
LLAMA_MODEL_NAME = "TheBloke/LLaMA-7B-GGUF"  # Replace with desired model
LLAMA_MODEL_PATH = MODEL_DIR / "llama-7b"
WHISPER_MODEL_NAME = "large-v2"  # Use "small", "medium", or "large" based on performance needs

# Ensure model directory exists
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load FasterWhisper model (default: float16, auto-select device)
whisper_model = WhisperModel(WHISPER_MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu")

# Dictionary to track loaded models
loaded_models = {}

def download_model(model_name: str, model_path: Path):
    """Download model from Hugging Face if not present."""
    if not model_path.exists():
        print(f"Downloading model: {model_name}...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_name, local_dir=model_path)
    else:
        print(f"Model found at {model_path}")

def load_llama_model():
    """Load LLaMA model into memory."""
    download_model(LLAMA_MODEL_NAME, LLAMA_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
    )
    return tokenizer, model

# Load LLaMA model
llama_tokenizer, llama_model = load_llama_model()
loaded_models["llama"] = llama_model

async def generate_text(prompt: str):
    """Generate text using LLaMA."""
    inputs = llama_tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = llama_model.generate(**inputs, max_length=500)
    return llama_tokenizer.decode(outputs[0], skip_special_tokens=True)

async def transcribe_audio(audio_bytes):
    """Transcribe audio using FasterWhisper."""
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    segments, _ = whisper_model.transcribe(audio_np)
    
    # Combine transcription results
    transcription = " ".join(segment.text for segment in segments)
    return transcription

# WebSocket for real-time chat
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """Real-time AI interactions via WebSocket."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message["event"] == "conversation.item.create":
                user_message = message["data"]["content"]
                response_text = await generate_text(user_message)
                await websocket.send_text(json.dumps({"response": response_text}))

            elif message["event"] == "input_audio_buffer.append":
                audio_bytes = bytes.fromhex(message["data"]["bytes"])
                transcript = await transcribe_audio(audio_bytes)
                await websocket.send_text(json.dumps({"transcription": transcript}))

    except WebSocketDisconnect:
        print(f"Client {client_id} disconnected.")

# Chat Completion API
@app.post("/v1/chat/completions")
async def chat_completion(payload: dict):
    """Handle chat completions."""
    messages = payload.get("messages", [])
    user_message = messages[-1]["content"] if messages else "Hello"
    response_text = await generate_text(user_message)
    return {"id": "chatcmpl-xyz", "object": "chat.completion", "created": 1234567890, "choices": [{"message": {"role": "assistant", "content": response_text}}]}

# Speech-to-Text API
@app.post("/v1/audio/transcriptions")
async def transcribe_file(file: UploadFile = File(...)):
    """Transcribe an uploaded audio file."""
    temp_audio_path = f"./temp_{file.filename}"
    with open(temp_audio_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    segments, _ = whisper_model.transcribe(temp_audio_path)
    os.remove(temp_audio_path)

    transcription = " ".join(segment.text for segment in segments)
    return {"text": transcription}

# Model Management APIs
@app.get("/v1/models")
async def get_models():
    """List available models."""
    return {"data": [{"id": model, "object": "model"} for model in loaded_models.keys()], "object": "list"}

@app.get("/v1/models/{model_name}")
async def get_model(model_name: str):
    """Retrieve model details."""
    if model_name in loaded_models:
        return {"id": model_name, "object": "model", "created": 1234567890}
    raise HTTPException(status_code=404, detail="Model not found")

@app.post("/api/pull/{model_id}")
async def pull_model(model_id: str):
    """Download model from Hugging Face if not available locally."""
    model_path = MODEL_DIR / model_id
    download_model(model_id, model_path)
    return {"status": "Model downloaded", "model_id": model_id}

@app.get("/api/ps")
async def get_running_models():
    """List currently loaded models."""
    return {"models": list(loaded_models.keys())}

@app.post("/api/ps/{model_id}")
async def load_model_route(model_id: str):
    """Load a model into memory."""
    model_path = MODEL_DIR / model_id
    if not model_path.exists():
        return {"error": "Model not found locally. Please pull the model first."}
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    
    loaded_models[model_id] = model
    return {"status": "Model loaded", "model_id": model_id}

@app.delete("/api/ps/{model_id}")
async def stop_running_model(model_id: str):
    """Unload a model from memory."""
    if model_id in loaded_models:
        del loaded_models[model_id]
        return {"status": "Model unloaded", "model_id": model_id}
    raise HTTPException(status_code=404, detail="Model not found")

# Health check
@app.get("/health")
async def health():
    return {"status": "ok"}
