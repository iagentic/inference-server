from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
import os
import torch
import asyncio
import json
import shutil
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from faster_whisper import WhisperModel
from huggingface_hub import login, snapshot_download

app = FastAPI(
    title="LLaMA 3.2 FastAPI Server",
    version="0.1.0",
    openapi_version="3.1.0"
)

# ✅ Authenticate with Hugging Face
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")  # Ensure token is set
if not HF_TOKEN:
    raise ValueError("Missing Hugging Face Token. Run 'huggingface-cli login' or set HUGGINGFACE_TOKEN.")

login(token=HF_TOKEN)

# ✅ Model Paths & Directories
MODEL_DIR = Path("./models")
LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8"
LLAMA_MODEL_PATH = MODEL_DIR / "llama-3"

# ✅ Ensure model directory exists
MODEL_DIR.mkdir(parents=True, exist_ok=True)
loaded_models = {}
def download_llama_model():
    """Download and load LLaMA 3.2 1B INT4"""
    if not LLAMA_MODEL_PATH.exists():
        print(f"Downloading LLaMA model: {LLAMA_MODEL_NAME}...")
        snapshot_download(repo_id=LLAMA_MODEL_NAME, local_dir=LLAMA_MODEL_PATH)
    
    print(f"Loading model from {LLAMA_MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    loaded_models["llama"] = LLAMA_MODEL_NAME

    return tokenizer, model

# ✅ Load LLaMA 3.2 model
llama_tokenizer, llama_model = download_llama_model()

# ✅ Load FasterWhisper (for speech-to-text)
whisper_model = WhisperModel("large-v2", device="cuda" if torch.cuda.is_available() else "cpu")

async def generate_text(prompt: str):
    """Generate text using LLaMA 3.2"""
    inputs = llama_tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = llama_model.generate(**inputs, max_length=500)
    return llama_tokenizer.decode(outputs[0], skip_special_tokens=True)

async def transcribe_audio(audio_bytes):
    """Transcribe audio using FasterWhisper"""
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    segments, _ = whisper_model.transcribe(audio_np)
    return " ".join(segment.text for segment in segments)

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """Real-time AI interactions via WebSocket"""
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

@app.post("/v1/chat/completions")
async def chat_completion(payload: dict):
    """Handle chat completions"""
    messages = payload.get("messages", [])
    user_message = messages[-1]["content"] if messages else "Hello"
    response_text = await generate_text(user_message)
    return {"id": "chatcmpl-xyz", "object": "chat.completion", "created": 1234567890, "choices": [{"message": {"role": "assistant", "content": response_text}}]}

@app.post("/v1/audio/transcriptions")
async def transcribe_file(file: UploadFile = File(...)):
    """Transcribe an uploaded audio file"""
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
