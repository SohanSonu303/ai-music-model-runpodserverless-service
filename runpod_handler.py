import runpod
import torch
import torchaudio
import soundfile as sf
import base64
import io
import requests
import uuid
import os
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from huggingface_hub import snapshot_download
from pydub import AudioSegment

# ---------------------------------------------------------
# 🔐 AUTH USING ENV VARIABLE
# ---------------------------------------------------------
ALLOWED_API_KEY = os.environ.get("API_SECRET", None)
if not ALLOWED_API_KEY:
    print("WARNING: Environment variable API_SECRET not set! API will be unprotected.")

# ---------------------------------------------------------
# LOAD MODEL AT WORKER START (NOT IN DOCKER BUILD)
# ---------------------------------------------------------
print("🔥 Initializing worker...")

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = "/root/.cache/huggingface/hub/musicgen-melody"

if not os.path.exists(MODEL_DIR):
    print("📥 Downloading MusicGen model...")
    snapshot_download(
        repo_id="facebook/musicgen-melody",
        local_dir=MODEL_DIR
    )
    print("✅ Model downloaded.")

print("🔧 Loading MusicGen into memory...")
processor = AutoProcessor.from_pretrained(MODEL_DIR)
model = MusicgenForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)
print("✅ MusicGen ready.")


# ---------------------------------------------------------
# HANDLER
# ---------------------------------------------------------
def handler(event):

    # --------------------------------------------
    # 🔐 Authorization (env-based)
    # --------------------------------------------
    if ALLOWED_API_KEY:
        headers = event.get("headers", {}) or {}
        auth_header = headers.get("Authorization") or headers.get("authorization")

        if not auth_header or not auth_header.startswith("Bearer "):
            return {"error": "Missing or invalid Authorization header"}

        api_key = auth_header.split(" ")[1]
        if api_key != ALLOWED_API_KEY:
            return {"error": "Invalid API Key"}

    # --------------------------------------------
    # INPUTS
    # --------------------------------------------
    prompt = event.get("prompt", "calm ambient music")
    duration = int(event.get("duration", 20))
    ref_audio_url = event.get("ref_audio", None)

    melody = None

    # --------------------------------------------
    # OPTIONAL REFERENCE AUDIO
    # --------------------------------------------
    if ref_audio_url:
        audio_bytes = requests.get(ref_audio_url).content
        temp_path = f"/tmp/ref_{uuid.uuid4()}.wav"

        with open(temp_path, "wb") as f:
            f.write(audio_bytes)

        audio = AudioSegment.from_file(temp_path)
        audio = audio.set_channels(1).set_frame_rate(32000)
        audio.export(temp_path, format="wav")

        melody, sr = torchaudio.load(temp_path)
        melody = melody.to(device)

    # --------------------------------------------
    # PROCESS INPUTS
    # --------------------------------------------
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
        sampling_rate=32000
    ).to(device)

    # --------------------------------------------
    # GENERATE AUDIO
    # --------------------------------------------
    audio_values = model.generate(
        **inputs,
        melody=melody,
        do_sample=True,
        max_new_tokens=32000 * duration
    )

    # --------------------------------------------
    # ENCODE WAV
    # --------------------------------------------
    audio = audio_values[0].cpu().numpy()
    buffer = io.BytesIO()
    sf.write(buffer, audio.T, 32000, format="WAV")
    buffer.seek(0)

    audio_b64 = base64.b64encode(buffer.read()).decode()

    return {
        "status": "success",
        "prompt": prompt,
        "audio_base64": audio_b64
    }


# ---------------------------------------------------------
# START SERVERLESS
# ---------------------------------------------------------
runpod.serverless.start({"handler": handler})