You're picking up the Groundwork project at https://github.com/jalengg/groundwork
mid-stream. Read these files first to understand where things stand:

  docs/sdxl_controlnet.md       — the working method (SDXL + ControlNet)
  docs/postmortem.md            — 4-month journey, what worked, what didn't
  data_pipeline/                — current pipeline: OSM vectors + SRTM + OSM landuse
                                  → 7-channel cond + 5-channel road rasters
  data_pipeline/prep_flux_dataset.py  — 7→3 RGB + 1024x1024 upscale for SDXL ControlNet

Existing state:
  - Trained model: jalengg/groundwork-sdxl-cnet-us-suburbs on HuggingFace Hub (private)
  - Training data: 17 US Sun Belt cities only (Arlington TX, Mesa AZ, Henderson NV,
    Plano TX, Carlsbad CA, Plymouth MN, etc.) — strong US suburban grid bias
  - 5-class road palette + 7-channel cond pipeline already works end-to-end
  - One model = ~35 GPU-hours of A100 training on UIUC cluster

The goal: expand beyond US suburbs to capture multiple distinct city morphologies
worldwide. The user wants representative samples of:
  - US grid suburbs (already have)
  - Latin America urban (favelas, Spanish colonial grid + organic mix)
  - European (medieval centers + planned bourgeois suburbs + autoroute regions)
  - Organic/medieval (winding non-grid streets — could overlap with European)
  - Corbusierian / Communist bloc (housing estates with discontinuous superblocks)
  - Possibly others you'd add from urban morphology expertise

Your task — produce a phased plan with these deliverables:

1. **Stylistic taxonomy.** Drawing on urban morphology literature (Conzen,
   Hillier's space syntax, Marshall, Mumford), propose a defensible set of
   global city styles. Don't just list "European" — break it down (e.g.
   Haussmann boulevards vs medieval kernel vs satellite housing estates).
   How many distinct style classes does the data actually support? Cite
   academic sources for the typology where useful.

2. **City list per style.** For each style, recommend 10-20 specific cities
   that are good morphological exemplars. Mix size (so we get suburb-scale
   tiles, not just downtown cores). Note OSM data coverage quality per
   region (Latin America OSM is patchier than Western Europe).

3. **Data acquisition feasibility.** For each style, assess:
   - OSM road graph coverage (use OSM Wiki coverage maps / mapathon stats)
   - Landuse polygon coverage (highly variable globally)
   - Terrain data: SRTM (global, 30m) is fine; better local DEMs exist for
     some regions
   - Any region where coverage is so bad we'd need to drop the style
   - Whether OSM landuse tags differ regionally (e.g. European 'industrial'
     vs US 'industrial' — same tag, but the actual land use can look
     different visually)

4. **Architecture decision.** Recommend either:
   - **One big multi-style ControlNet** with a "style code" channel injected
     into cond (would need to extend the 7-ch cond → 8-ch with a one-hot
     style index, or use a CLIP-embedded style prompt). Pros: single model,
     transferable priors. Cons: needs balanced data; may average styles
     together.
   - **N per-style ControlNets** all fine-tuned from the US suburbs model.
     Pros: clean separation, easier to add new styles incrementally. Cons:
     N× inference deployment, N× model storage.
   - Justify with parameter and GPU-hour estimates.

5. **Phased rollout.** Recommend an order: which style to add next given
   the existing US-suburbs base, optimizing for distinct visual payoff per
   training run. Bias toward styles where the existing US model is *most*
   wrong (i.e. European medieval — totally OOD from our training data).

6. **Data pipeline portability check.** Identify whether prep_flux_dataset.py
   and the upstream data_pipeline assume things that will break for non-US
   regions (CRS projections, US-specific OSM tag mappings, etc.). What
   needs to be generalized?

Output: write your findings to a new file `docs/multi_style_plan.md` in the
repo, then commit to a new branch `jalen/multi-style-plan`. Don't start any
data acquisition or training in this conversation — produce the plan first,
then we'll review and stage the actual data work.

Compute budget for context: each new style ≈ 35 A100-hours of training (one
SLURM job + auto-resume on the UIUC `IllinoisComputes-GPU` partition).
Data acquisition is mostly free — OSM + SRTM are public, the question is
preprocessing labor.
