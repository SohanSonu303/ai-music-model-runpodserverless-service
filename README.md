# ai-music-runpod-serverless (ACE-Step v1.5)

RunPod serverless worker that generates music using [ACE-Step v1.5](https://github.com/ace-step/ACE-Step-1.5) (MIT).

**GPU requirement:** 24 GB+ VRAM — A100 or L40S tier on RunPod. The 4B XL SFT model is baked into the Docker image (~15 GB).

---

## 1. Build & push the image

```bash
# Log in to Docker Hub (or any registry)
docker login

# Build for x86_64 (required for RunPod — Apple Silicon Macs are ARM, RunPod is x86_64)
# Expect 20–40 min (downloads ~15 GB of weights)
docker buildx build --platform linux/amd64 -t your-dockerhub-username/ai-music-service:latest .

# Push so RunPod can pull it
docker push your-dockerhub-username/ai-music-service:latest
```

---

## 2. Test before deploying to serverless

The image needs a 24 GB GPU to run. Two ways to test without committing to a full serverless endpoint:

### Option A — RunPod Pod (recommended)

Spin up a temporary **Pod** with your image (~$2–3/hr for A100, pay only while running):

1. Go to **runpod.io → Pods → + Deploy**
2. Select **Custom Image** → paste `your-dockerhub-username/ai-music-service:latest`
3. Choose GPU: **A100 SXM** or **L40S**
4. Under **Environment Variables** add `API_SECRET=test`
5. Under **Expose HTTP Ports** add port `8000`
6. Click **Deploy** and wait for status **Running**
7. Copy the pod's **HTTP URL** (shown on the pod card, looks like `https://abc123-8000.proxy.runpod.net`)

Test it:
```bash
curl -s -X POST https://YOUR_POD_URL/runsync \
  -H "Authorization: Bearer test" \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "lofi piano", "duration": 10, "instrumental": true}}' \
  | jq -r '.output.audio_base64' | base64 -d > out.wav

afplay out.wav   # Mac
# ffplay out.wav  # Linux
```

8. **Terminate the pod** when done to stop billing.

### Option B — Local Docker (only if you have a 24 GB GPU locally)

```bash
docker build -t ai-music-service .

docker run --rm --gpus all \
  -e API_SECRET=test \
  -p 8000:8000 \
  ai-music-service
```

In another terminal:
```bash
curl -s -X POST http://localhost:8000/runsync \
  -H "Authorization: Bearer test" \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "lofi piano", "duration": 10, "instrumental": true}}' \
  | jq -r '.output.audio_base64' | base64 -d > out.wav

afplay out.wav
```

---

## 3. Create the RunPod serverless endpoint

1. Go to **[runpod.io](https://www.runpod.io) → Serverless → + New Endpoint**
2. Select **Custom Source** → paste your image URL (`your-dockerhub-username/ai-music-service:latest`)
3. Set GPU: **A100 SXM 80 GB** or **L40S 48 GB** (both have 24+ GB VRAM)
4. Under **Environment Variables** add:
   ```
   API_SECRET = <choose a secret key>
   ```
5. Click **Deploy**. RunPod pulls the image and starts workers.
6. Copy the **Endpoint ID** shown on the dashboard (looks like `abc123xyz`).

---

## 4. Test on RunPod serverless

Replace `YOUR_ENDPOINT_ID` and `YOUR_RUNPOD_API_KEY` (from runpod.io → Settings → API Keys).

### Text-to-music

```bash
curl -s -X POST \
  "https://api.runpod.io/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "lofi hip hop, slow piano, rain ambience",
      "duration": 20,
      "instrumental": true
    }
  }' | jq .
```

### Reference-audio cover (style transfer)

```bash
curl -s -X POST \
  "https://api.runpod.io/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "make it jazzy",
      "ref_audio": "https://your-public-url.com/melody.wav",
      "duration": 20
    }
  }' | jq .
```

> `runsync` waits for the result (up to 90 s by default). For longer generations use `/run` + poll `/status/{job_id}` — see the async section below.

### Save the audio output

```bash
# Pipe the full curl response and extract the audio
curl -s -X POST \
  "https://api.runpod.io/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "upbeat synth pop", "duration": 15, "instrumental": true}}' \
  | jq -r '.output.audio_base64' \
  | base64 -d > output.wav

# Play it
ffplay output.wav
# or on Mac:
afplay output.wav
```

### Async (for longer generations)

```bash
# 1. Submit job
JOB_ID=$(curl -s -X POST \
  "https://api.runpod.io/v2/YOUR_ENDPOINT_ID/run" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "epic orchestral", "duration": 60}}' \
  | jq -r '.id')

echo "Job: $JOB_ID"

# 2. Poll until status = COMPLETED
curl -s \
  "https://api.runpod.io/v2/YOUR_ENDPOINT_ID/status/$JOB_ID" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" | jq .

# 3. Get audio once COMPLETED
curl -s \
  "https://api.runpod.io/v2/YOUR_ENDPOINT_ID/status/$JOB_ID" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  | jq -r '.output.audio_base64' | base64 -d > output.wav
```

---

## 5. Test via RunPod dashboard (no curl needed)

1. Open your endpoint on runpod.io → click **Run**
2. Paste this into the **Request Input** box:
   ```json
   {
     "input": {
       "prompt": "lofi piano chill",
       "duration": 15,
       "instrumental": true
     }
   }
   ```
3. Click **Run** — the output JSON appears in the **Response** panel.
4. Copy the `audio_base64` value and decode it locally:
   ```bash
   echo "PASTE_BASE64_HERE" | base64 -d > output.wav && afplay output.wav
   ```

---

## Input reference

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `prompt` | string | `"calm ambient music"` | Text description of the music |
| `duration` | float | `30` | Output length in seconds |
| `ref_audio` | string (URL) | — | If set, runs **cover** mode (style transfer) |
| `instrumental` | bool | `true` | Set `false` to allow vocals |
| `bpm` | int | — | Optional tempo hint |
| `guidance_scale` | float | `7.5` | Higher = more faithful to prompt |
| `seed` | int | — | For reproducibility |

Auth header on every request: `Authorization: Bearer <API_SECRET>` (the env var you set on the endpoint).

## Output

```json
{
  "status": "success",
  "task": "text2music",
  "prompt": "lofi piano chill",
  "duration": 15,
  "sample_rate": 48000,
  "audio_base64": "<base64 stereo WAV at 48 kHz>"
}
```

---

## Model configuration (env vars)

| Var | Default | Options |
|-----|---------|---------|
| `ACESTEP_CONFIG` | `acestep-v15-xl-sft` | `acestep-v15-xl-base`, `acestep-v15-xl-turbo` |
| `LM_MODEL_PATH` | `acestep-5Hz-lm-0.6B` | `acestep-5Hz-lm-1.7B` (better quality, needs separate bake step) |
| `LM_CHECKPOINT_DIR` | `/app/checkpoints` | Path where the LM subdirectory lives |
| `API_SECRET` | — | Bearer token clients must send |
