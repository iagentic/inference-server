# inference-server
#for kokoro
# https://github.com/remsky/Kokoro-FastAPI.git
# docker run -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:v0.2.2
# docker run --gpus all -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-gpu:v0.2.2
# if necssary install cuda https://developer.nvidia.com/cudnn-9-1-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local
# pip install -r requirements.txt
# pip install setuptools==58.0.0

## uvicorn main:app --host 0.0.0.0 --port 8000 --reload

## python gradio_test.py
# install ultravox 
# sudo apt-get install libportaudio2
# sudo apt install portaudio19-dev

# pip install torch torchaudio einops timm pillow
# pip install git+https://github.com/huggingface/transformers
# pip install git+https://github.com/huggingface/accelerate
# pip install git+https://github.com/huggingface/diffusers
# pip install huggingface_hub
# pip install sentencepiece bitsandbytes protobuf decord
#  pip install librosa peft numpy
#  pip install pyaudio

