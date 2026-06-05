You're picking up the Groundwork project at https://github.com/jalengg/groundwork
to build the server-side inference API. Read these files first:

  docs/sdxl_controlnet.md       — the working method (what the model does)
  docs/deployment_strategy.md   — the architecture we're building against;
                                  this conversation builds the SERVER half
  model/postprocess.py          — existing road-graph vectorization
                                  (skeleton → graph → simplify), already works
  tools/sdxl_cnet_sample.py     — reference inference loader

Your scope: the cloud inference service. NOT the CS mod client — that's a
separate conversation. You can stub the client; we'll wire it in later.

Existing context:
  - Trained model: jalengg/groundwork-sdxl-cnet-us-suburbs on HF Hub (private)
  - Foundation: stabilityai/stable-diffusion-xl-base-1.0 + madebyollin/sdxl-vae-fp16-fix
  - Inference: ~30s/tile on A100, ~60-90s on consumer 12GB cards
  - Mod will POST 1024x1024 RGB cond images, expect a road graph back
  - Patreon-tier priority queue per deployment_strategy.md
  - Total scale at v1 launch: probably 10s-100s of generations/day, growing

Your task — build:

1. **Inference worker.** Containerized service that:
   - Loads SDXL base + ControlNet head + VAE on startup (~12GB VRAM)
   - Exposes a worker interface that takes a cond image, returns:
     - The generated RGB road network image
     - The vectorized road graph (use existing `model/postprocess.py`)
     - Mapping from our 5 road classes to CS-compatible road type strings
   - Targets: A100 or RTX 4090 worker hardware

2. **API gateway.** FastAPI service exposing:
   - `POST /api/generate` (auth, queue insert)
   - `GET /api/job/{id}` (poll status)
   - `GET /api/health`
   - Patreon OAuth flow for tier verification (`/auth/patreon_callback`)
   - Anonymous tier rate-limiting (5 req/day per IP for v1)
   - Job tracking (Redis or sqlite for v1)

3. **Queue with priority.** Implement FIFO with three priority levels:
   - Anonymous: standard queue
   - Patreon $10/mo: priority queue (cuts ahead of standard)
   - Patreon $20/mo: top priority + concurrent if worker available
   Evaluate Celery vs RQ vs a simple custom asyncio queue for v1.

4. **Deployment.** Decide and document the hosting choice for v1. Realistic
   options from `docs/deployment_strategy.md`:
   - **Replicate** — pay-per-second, easy, ~$0.01-0.02/gen
   - **HuggingFace Spaces (ZeroGPU)** — free tier, slow cold starts
   - **RunPod serverless** — middle ground, more control
   - **Self-hosted A100** — cheapest at scale, most ops burden
   Pick one and justify, then write the actual deployment config.

5. **Output → CS road-class mapping.** Our 5 road palette (residential,
   tertiary, primary, motorway, bg) needs to map to CS1 road prefabs.
   Coordinate with the cs-mod-client conversation on the exact mapping
   table (or define it here and document it). Output format:
   ```json
   {
     "image": "<base64 RGB PNG>",
     "graph": {
       "nodes": [{"id": 0, "x": 123, "y": 456}, ...],
       "edges": [{"from": 0, "to": 1, "cs_road_type": "RoadMedium", "level": 3}, ...]
     }
   }
   ```

6. **Cost monitoring.** Per-tenant generation tracking so we can:
   - Enforce Patreon-tier quotas
   - Surface cost per active patron
   - Cut off abuse from anonymous tier

Honest scope check: this is a real backend engineering project. Build it
phased — milestone 1 is "single-worker FastAPI that takes a cond image,
generates, returns a graph"; milestone 2 is queue + tiers; milestone 3 is
deployment + monitoring. Don't try to one-shot.

New branch: `jalen/cs-server`.

Inference reference code at `tools/sdxl_cnet_sample.py` is a good starting
point — it loads the model and runs inference. Postprocessing at
`model/postprocess.py` already does skeleton→graph→simplify. The new work
is wrapping these in a long-running service with auth/queue/billing.

For local dev: model weights download from HF Hub. Need an HF read token
for the private repo (`HF_TOKEN` env var).
