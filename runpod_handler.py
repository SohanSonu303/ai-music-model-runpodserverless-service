import base64
import io
import os
import uuid

import requests
import runpod
import soundfile as sf
import torch

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import generate_music, GenerationParams, GenerationConfig

# ---------------------------------------------------------
# AUTH
# ---------------------------------------------------------
ALLOWED_API_KEY = os.environ.get("API_SECRET")
if not ALLOWED_API_KEY:
    print("WARNING: API_SECRET not set — endpoint is unprotected.")

# ---------------------------------------------------------
# DiT HANDLER (loads once at worker cold-start)
# ---------------------------------------------------------
ACESTEP_CONFIG = os.environ.get("ACESTEP_CONFIG", "acestep-v15-xl-sft")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading ACE-Step DiT ({ACESTEP_CONFIG}) on {DEVICE}...")
dit_handler = AceStepHandler()
dit_handler.initialize_service(
    project_root="/app",
    config_path=ACESTEP_CONFIG,
    device=DEVICE,
)
print("DiT ready.")

# ---------------------------------------------------------
# LM HANDLER (optional — skipped if checkpoint absent)
# LM adds 5Hz musical-structure codes for text2music quality.
# Not used for cover tasks (ACE-Step skips it automatically).
# ---------------------------------------------------------
LM_CHECKPOINT_DIR = os.environ.get("LM_CHECKPOINT_DIR", "/app/checkpoints")
LM_MODEL_PATH = os.environ.get("LM_MODEL_PATH", "acestep-5Hz-lm-0.6B")

_lm_full_path = os.path.join(LM_CHECKPOINT_DIR, LM_MODEL_PATH)
if os.path.isdir(_lm_full_path):
    print(f"Loading LM ({LM_MODEL_PATH}) on {DEVICE}...")
    llm_handler = LLMHandler()
    llm_handler.initialize(
        checkpoint_dir=LM_CHECKPOINT_DIR,
        lm_model_path=LM_MODEL_PATH,
        backend="pt",
        device=DEVICE,
    )
    print("LM ready.")
else:
    print(f"LM checkpoint not found at {_lm_full_path} — running DiT-only.")
    llm_handler = None

SAMPLE_RATE = 48000  # ACE-Step normalises output to stereo 48 kHz

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def _auth_ok(event):
    if not ALLOWED_API_KEY:
        return True
    headers = event.get("headers") or {}
    auth = headers.get("Authorization") or headers.get("authorization") or ""
    return auth.startswith("Bearer ") and auth.split(" ", 1)[1] == ALLOWED_API_KEY


def _download_ref(url):
    path = f"/tmp/ref_{uuid.uuid4().hex}.wav"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return path


# ---------------------------------------------------------
# HANDLER
# ---------------------------------------------------------
def handler(event):
    if not _auth_ok(event):
        return {"error": "Invalid or missing Authorization header"}

    # RunPod wraps payload under "input"; also accept flat payload for local testing
    inp = event.get("input", event)
    prompt = inp.get("prompt", "calm ambient music")
    duration = float(inp.get("duration", 30))
    guidance_scale = float(inp.get("guidance_scale", 7.5))
    seed = inp.get("seed")
    instrumental = bool(inp.get("instrumental", True))
    bpm = inp.get("bpm")
    ref_audio_url = inp.get("ref_audio")

    task_type = "cover" if ref_audio_url else "text2music"
    reference_audio = _download_ref(ref_audio_url) if ref_audio_url else None

    params = GenerationParams(
        task_type=task_type,
        caption=prompt,
        duration=duration,
        reference_audio=reference_audio,
        instrumental=instrumental,
        bpm=bpm,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    config = GenerationConfig(batch_size=1, audio_format="wav")

    result = generate_music(dit_handler, llm_handler, params, config, save_dir="/tmp")

    if not result.success:
        return {"error": result.error or "Generation failed"}

    audio_dict = result.audios[0]
    audio_tensor = audio_dict["tensor"]  # [channels, samples], CPU float32

    # Clean up temp files written by generate_music
    audio_path = audio_dict.get("path")
    if audio_path and os.path.exists(audio_path):
        try:
            os.unlink(audio_path)
        except OSError:
            pass
    if reference_audio and os.path.exists(reference_audio):
        try:
            os.unlink(reference_audio)
        except OSError:
            pass

    buf = io.BytesIO()
    sf.write(buf, audio_tensor.numpy().T, SAMPLE_RATE, format="WAV")
    buf.seek(0)

    return {
        "status": "success",
        "task": task_type,
        "prompt": prompt,
        "duration": duration,
        "sample_rate": SAMPLE_RATE,
        "audio_base64": base64.b64encode(buf.read()).decode(),
    }


# ---------------------------------------------------------
# START SERVERLESS
# ---------------------------------------------------------
runpod.serverless.start({"handler": handler})
