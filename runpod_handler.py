import runpod
import torch
import torchaudio
import soundfile as sf
import base64
import io
import requests
import uuid
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from pydub import AudioSegment

# ---------------------------------------------------------
# LOAD MODEL FROM LOCAL CACHE (PRE-BAKED)
# ---------------------------------------------------------
print("Loading MusicGen from local Docker cache...")

device = "cuda" if torch.cuda.is_available() else "cpu"

LOCAL_MODEL_PATH = "/root/.cache/huggingface/hub/musicgen-melody"

processor = AutoProcessor.from_pretrained(LOCAL_MODEL_PATH)

model = MusicgenForConditionalGeneration.from_pretrained(
    LOCAL_MODEL_PATH,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

print("✅ MusicGen ready.")

# ---------------------------------------------------------
# HANDLER FUNCTION
# ---------------------------------------------------------
def handler(event):
    prompt = event.get("prompt", "ambient music")
    duration = event.get("duration", 30)
    ref_audio_url = event.get("ref_audio", None)

    melody = None

    # Optional reference audio
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

    # Prepare inputs
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
        sampling_rate=32000
    ).to(device)

    # Generate
    if melody is not None:
        audio_values = model.generate(
            **inputs,
            melody=melody,
            do_sample=True
        )
    else:
        audio_values = model.generate(
            **inputs,
            do_sample=True
        )

    # Convert to WAV in memory
    audio = audio_values[0].cpu().numpy()
    buffer = io.BytesIO()
    sf.write(buffer, audio.T, 32000, format="WAV")
    buffer.seek(0)

    audio_b64 = base64.b64encode(buffer.read()).decode("utf-8")

    return {
        "status": "success",
        "prompt": prompt,
        "duration": duration,
        "audio_base64": audio_b64
    }

# ---------------------------------------------------------
# START SERVERLESS
# ---------------------------------------------------------
runpod.serverless.start({"handler": handler})