# Cities Skylines Mod Deployment Strategy

How the trained SDXL ControlNet model will reach players. Not yet
implemented — this is the design we agreed on.

## Constraints

| Constraint | Value |
|---|---|
| Model size | ~12 GB total (SDXL base + ControlNet + VAE + text encoders) |
| Inference VRAM | ~10-12 GB at fp16 |
| Inference latency (A100) | ~30 s / tile |
| Inference latency (RTX 3060) | ~60-90 s / tile (if it fits at all) |
| Steam Workshop size limit | typically <1 GB per item |
| Median CS player GPU | RTX 3060 / GTX 1660 — many <12 GB VRAM |

**Local inference is non-viable** for ~50 % of CS players. Distributing a
12 GB diffusion model through Steam Workshop is unprecedented. Therefore:

## Architecture: cloud inference with priority queue

```
┌───────────────────────┐   1. paints landuse
│ CS player + the mod   ├────────────────────────┐
│ - paintbrush UI       │                        │
│ - Chirpy mascot       │                        ▼
└───▲───────────────┬───┘                  ┌──────────────┐
    │5. accept/     │ 2. POST cond raster  │ Inference    │
    │   reroll      │    (3-channel        │ queue        │
    │               │     1024² PNG)       │ (FIFO with   │
    │               │                      │  Patreon-    │
    │ 4. road graph │                      │  tier        │
    │    + image    │                      │  priority)   │
    │               ▼                      └──────┬───────┘
┌───┴───────────────────┐                         │
│ API gateway           │                         │ 3. dispatch
│ - auth (anon/patron)  │                         ▼
│ - rate limit          │              ┌──────────────────┐
│ - job tracking        │              │ GPU worker(s)    │
│ - Chirpy push notif   │              │ - SDXL+ControlNet│
└───────────────────────┘              │ - postprocess to │
                                       │   road graph     │
                                       └──────────────────┘
```

## Tier strategy

| Tier | Cost | Queue | Concurrency |
|---|---|---|---|
| Anonymous | $0 | Standard FIFO | Heavy rate limit (e.g. 5 generations/day per IP) |
| Patreon $5/mo | Subscription | Priority | 50 gen/day, faster slot |
| Patreon $15/mo | Subscription | Top priority | Unlimited, dedicated worker if available |

Math: if a worker can do ~30 s/tile = ~2 tiles/min = 2,880 tiles/day, one
A100 GPU at ~$1.50/hr on RunPod = ~$36/day = ~$0.01/tile served.

At $5/mo for 50 gens/day cap = up to 1,500/mo per patron = $0.0033/gen
margin against $0.01 cost — **negative margin on heavy use**. Either:
- Bump price ($10-15/mo more reasonable)
- Cap quota lower (10/day instead of 50)
- Average usage will be much lower than cap (most patrons use <10/day)

Anonymous tier subsidized by patrons. Standard SaaS pattern.

## Inference host options

| Option | Cost model | Pros | Cons |
|---|---|---|---|
| **Replicate** | Pay-per-second | Easiest setup, auto-scaling, public model hosting | Margin shrinks at scale (~$0.003/sec on A100) |
| **HuggingFace Spaces (ZeroGPU)** | Free tier + paid | Free for low-traffic, integrated with HF Hub | ZeroGPU cold starts ~30s, premium gets expensive |
| **RunPod** | Per-hour GPU rental | Cheapest per-hour, persistent worker | You manage scaling, ops |
| **Modal** | Per-second + cold start | Good devex, fast cold starts | Pricing mid-tier |
| **Self-host on a $1000 RTX 4090 desktop** | One-time hardware | No recurring cost | Single point of failure, no scale |

**Pragmatic v1**: ship on Replicate. Single A100 worker, ~$0.01-0.02 per
generation, scales automatically. Migrate to RunPod or Modal if traffic
justifies the ops investment.

## API surface

```
POST /api/generate
  body: {
    "cond": "<base64 PNG, 1024×1024 RGB>",
    "guidance_scale": 5.0,
    "controlnet_scale": 1.0,
    "seed": 42  // optional
  }
  auth: anonymous IP-rate-limited OR Patreon JWT
  response: { "job_id": "...", "queue_position": 12 }

GET /api/job/{job_id}
  response: {
    "status": "queued" | "running" | "done" | "error",
    "image_url": "...",      // when done
    "road_graph": [...],     // postprocessed
    "queue_position": 3
  }

POST /api/auth/patreon_callback
  // OAuth2 flow for Patreon tier verification
```

## "Chirpy" notification flow

1. Player paints landuse in CS, clicks "Generate"
2. Mod POSTs cond to API, gets `job_id`, shows "queued" UI
3. Mod polls `/api/job/{job_id}` every 2 s while in queue
4. Chirpy mascot appears in corner with countdown
5. When status flips to `done`, Chirpy plays a chirp sound, shows preview
6. Player clicks accept / reroll
7. On accept: mod consumes `road_graph`, invokes `NetManager.CreateSegment`
   for each edge

## Open implementation questions

| Question | Status |
|---|---|
| HF Hub upload path | Pending (model weights at `/scratch/jalenj4/runs/sdxl_cnet_v1/`) |
| Patreon webhook integration | Not started |
| CS mod codebase (C# / Harmony patches) | Not started |
| `NetManager.CreateSegment` API surface for our segment types | Not researched |
| Postprocessor output → `NetSegment` mapping | Need road-class → CS road-type translation table |
| Inference endpoint host choice | Replicate v1, revisit at scale |
