from fastapi import FastAPI, UploadFile, File
import transformers
import numpy as np
import librosa
import io
import torch

app = FastAPI()

# Load Ultravox model
pipe = transformers.pipeline(model='fixie-ai/ultravox-v0_4', trust_remote_code=True)

# Ensure torch_dtype is set correctly
if isinstance(pipe.model.config.torch_dtype, str):
    pipe.model.config.torch_dtype = getattr(torch, pipe.model.config.torch_dtype, torch.float32)

@app.post("/process-audio/")
async def process_audio(file: UploadFile = File(...)):
    # Read the uploaded file
    audio_bytes = await file.read()
    
    # Convert bytes to numpy array
    audio_np, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)

    # Define turns for Ultravox conversation
    turns = [
        {
            "role": "system",
            "content": "You are a friendly and helpful character. You love to answer questions for people."
        },
    ]

    # Process audio with Ultravox
    response = pipe({'audio': audio_np, 'turns': turns, 'sampling_rate': sr}, max_new_tokens=30)

    return {"response": response}
