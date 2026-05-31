# Multi-Style Expansion Plan

Produced 2026-05-31. Picks up from the working US suburbs SDXL ControlNet
(`jalengg/groundwork-sdxl-cnet-us-suburbs`, trained on 17 Sun Belt cities,
25 k steps, ~35 A100-hours). Goal: expand to capture representative global
city morphologies for the Cities Skylines mod.

**Scope:** plan only — no data acquisition or training in this document.

---

## 1. Stylistic Taxonomy

### Framing

The existing model encodes one morphological type: the **post-war US automobile
suburb** — cul-de-sac collector networks, wide arterials, large superblocks,
low connectivity, strong motorway presence. Everything else is out-of-distribution.

Four scholarly frameworks bear directly on what makes road networks *visually
and structurally distinct* at the raster scale we operate at (5 m/px, ~2.5 km
tiles):

- **Conzen (1960)** — "town plan" = street pattern + plot pattern + building
  pattern. His three-part morphological region framework identifies the "burgage
  cycle" (organic medieval accumulation vs. planned extensions vs. fringe belts).
  At our resolution, the street pattern component is what matters.
- **Hillier & Hanson (1984), Hillier (1996)** — Space syntax. Global integration
  vs. local integration, axial maps. Medieval organics have low global integration;
  grids are globally integrated; housing estates have discontinuous structure.
- **Marshall (2005)** — *Streets & Patterns*. "Directional" (strong through-routes),
  "grid" (uniform), "organically planned" (irregular but coherent), "cellular"
  (superblocks, internal access only).
- **Moudon (1994)** — Morphological periods tied to transport technology eras.
  Pre-automobile (foot/horse) → tram/streetcar → car. Each produces a visually
  distinct tile signature.

### Proposed Style Classes

The key question is: what is *actually distinguishable* in a 512×512 px
5 m/pixel raster of road networks? Two styles that look the same at this
resolution shouldn't be separate training classes — the model can't learn the
distinction. Based on that constraint:

| # | Style ID | Description | Morphological basis |
|---|----------|-------------|---------------------|
| 0 | `us_suburb` | **(existing)** Cul-de-sac collector network, large blocks, arterial/motorway hierarchy | Moudon post-car; Marshall "cellular" |
| 1 | `euro_grid` | Regular medium-density grid, wide through-boulevards, small-to-medium blocks | Haussmann (Paris), Cerdà (Barcelona Eixample), Laws of the Indies; Marshall "directional" |
| 2 | `medieval_organic` | Irregular, winding streets, no grid, high intersection density, small blocks | Conzen organic; Hillier low global integration; Marshall "organically planned" |
| 3 | `soviet_microrayon` | Superblocks with internal loops, few through-streets, collector-dominated, very low road density | Mikrorayon/Plattenbau typology; Marshall "cellular" but with discontinuous internal structure |
| 4 | `latam_informal` | Dense organic mesh, terrain-following, high cul-de-sac and dead-end ratio, minimal hierarchy | Self-organized (Turner 2001), Hillier "deformed grids" |
| 5 | `east_asian_dense` | Very high road density, fine-grain grid or near-grid, minimal motorway within tile, alleys present | Japanese *chō* subdivision, Korean *hanok* districts; Moudon pre-car+car hybrid |

**6 total styles including existing**, each producing a visually distinct raster
signature at our tile scale.

**Styles not added and why:**

- *Latin American colonial grid* — visually nearly identical to `euro_grid` at
  5 m/px. Same tag distribution, same block proportions. Not worth a separate
  training run; include as cities in `euro_grid`.
- *Suburban UK / Australian* — intermediate between `us_suburb` and `medieval_organic`.
  Adds noise rather than a new mode.
- *MENA medina* (Fez, Marrakech) — extremely high dead-end ratio (Islamic
  "khuttas") is structurally interesting but OSM coverage of medina streets is
  patchy and the visual signature overlaps substantially with `medieval_organic`
  at 5 m/px. Defer to v2.
- *South/Southeast Asian mixed* — interesting but very high OSM tag variability
  and coverage gaps. Defer to v2.

### Distinguishability at 5 m/px

At 512 px × 5 m/px = 2.56 km tiles:

- `us_suburb` vs `soviet_microrayon`: both "cellular" but differ in block size
  (US: ~200-300 m blocks; Soviet: ~400-600 m superblocks) and internal access
  density (US has many cul-de-sacs; Soviet has few internal roads)
- `euro_grid` vs `east_asian_dense`: differ in intersection spacing (~80-120 m
  Japan/Korea vs ~120-200 m Europe) and alley presence
- `medieval_organic` is visually unmistakable — no parallel lines, highly
  irregular, dense intersections

---

## 2. City List Per Style

### Style 1: `euro_grid`

Target: ~150 tiles/city, 12-15 cities, mix of downtown and periphery.

| City | Country | Notes | OSM coverage |
|------|---------|-------|--------------|
| Paris 13e/14e (not 1er) | France | Haussmann grid without tourist congestion | Excellent |
| Barcelona Eixample | Spain | Cerdà grid, chamfered corners, textbook exemplar | Excellent |
| Brussels inner ring | Belgium | Mixed Haussmann + organic fringe | Excellent |
| Vienna Margareten/Favoriten | Austria | Gründerzeit grid, good mix | Excellent |
| Milan Porta Vittoria/Navigli | Italy | Radial+grid mix, not downtown core | Excellent |
| Buenos Aires Almagro/Boedo | Argentina | Laws of the Indies + 20c grid; tests LatAm colonial | Very good |
| Montevideo Cordón | Uruguay | Clean colonial grid, good OSM | Very good |
| Lyon Part-Dieu | France | Modern French grid, contrast with medieval Vieux-Lyon | Excellent |
| Bordeaux Saint-Michel | France | Mix of medieval + Haussmann, manageable | Excellent |
| Porto Bonfim | Portugal | Hilly grid, tests elevation interaction | Good |
| Athens Kallithéa | Greece | Dense Athenian grid, no motorways inside tiles | Good |
| Turin Crocetta/Nizza | Italy | Rational Piedmontese grid | Excellent |

**Tile strategy:** sample from residential and commercial districts, not historic
centers (those belong in `medieval_organic`). Use the city admin boundary minus
~1 km inward to avoid both the medieval core and outer periurban.

**OSM landuse coverage note:** European cities have excellent `landuse=residential`,
`landuse=commercial`, and `landuse=industrial` coverage. `landuse=retail` is less
consistently tagged in Greece and Portugal — the `commercial` channel will be
sparser. Acceptable.

---

### Style 2: `medieval_organic`

These cities have well-preserved pre-automobile historic cores. Sample only
from within the medieval walled boundary, or its functional equivalent.

| City | Country | Notes | OSM coverage |
|------|---------|-------|--------------|
| Bologna centro storico | Italy | Largest medieval center in Italy, well-mapped | Excellent |
| Bruges | Belgium | UNESCO site, exceptional OSM | Excellent |
| Chester | UK | Roman+ medieval, walled city | Excellent |
| Colmar | France | Exceptionally intact Alsatian medieval core | Excellent |
| Siena | Italy | UNESCO, minimal car penetration preserves street structure | Excellent |
| Toledo | Spain | Islamic+medieval palimpsest | Very good |
| Faro old town | Portugal | Small but clean exemplar | Good |
| Ghent Patershol | Belgium | Irregular medieval quarter, adjacent to grid | Excellent |
| York Shambles area | UK | Well-mapped | Excellent |
| Regensburg Altstadt | Germany | UNESCO Danube crossing | Excellent |
| Lucca | Italy | Intact city walls, Roman street grid fossilized into medieval | Excellent |
| Tallinn Vanalinn | Estonia | Well-mapped, northern European | Excellent |

**Tile strategy:** tile centers placed inside the medieval perimeter. At 512 px
@ 5 m/px = 2.56 km, most medieval cores fit within 1-3 tiles. Will produce
fewer tiles per city than suburban styles (~50-80 vs 150). To compensate,
use more cities.

**Important:** many medieval cores include large plazas (piazze, market squares)
which will appear as large `bg` (background) areas. This is correct and the model
should learn that `medieval_organic` tiles have a different bg/road ratio than
suburban tiles.

---

### Style 3: `soviet_microrayon`

Eastern European and Soviet-era housing estates. Characterized by large
residential slabs set in open space, served by collector roads with no
through-street grid.

| City | Country | Notes | OSM coverage |
|------|---------|-------|--------------|
| Warsaw Ursynów | Poland | Large, well-documented mikrorayon | Excellent |
| Warsaw Praga-Południe | Poland | Mix of pre-war and Soviet, tiles from postwar part | Excellent |
| Prague Jižní Město | Czech Republic | Canonical Czechoslovak panelák estate | Excellent |
| Bratislava Petržalka | Slovakia | Europe's largest housing estate by population | Excellent |
| Budapest Újpalota | Hungary | Large Hungarian housing estate | Very good |
| Vilnius Fabijoniškės | Lithuania | Well-mapped, appeared in Chernobyl TV series | Excellent |
| Tallinn Lasnamäe | Estonia | Baltic Soviet estate, excellent OSM | Excellent |
| Bucharest Drumul Taberei | Romania | Ceaușescu-era, variable OSM | Good |
| Kyiv Troieshchyna | Ukraine | Large Soviet estate; OSM quality may vary post-2022 | Variable |
| Leipzig Grünau | Germany | DDR Plattenbau, excellent German OSM | Excellent |
| Erfurt Johannesvorstadt | Germany | East German estate | Excellent |
| Krakow Nowa Huta | Poland | Planned socialist city, UNESCO candidate | Excellent |

**OSM note:** German DDR cities have the best coverage. Baltic states are excellent.
Bucharest and Kyiv are acceptable but landuse polygon coverage is spottier.

**Road channel note:** `soviet_microrayon` tiles will have almost no `motorway`
channel hits (class 4). The road hierarchy is shallow: mostly `residential`
(class 1) and a few `primary` (class 3). This means the model needs to learn
a different class distribution than `us_suburb`. Per-style ControlNet handles
this naturally; a multi-style model would need balancing.

---

### Style 4: `latam_informal`

Informal settlements (favelas, comunas, villas miserias). Very high road
density from organic growth, often on steep terrain. OSM coverage is the
most variable of all styles — see §3.

| City | Country | Notes | OSM coverage |
|------|---------|-------|--------------|
| Medellín Comunas 1-3 (NE hillside) | Colombia | HDM4/OpenCities project improved OSM significantly | Good–Very good |
| Bogotá Ciudad Bolívar (periphery) | Colombia | Large, tiles from lower slopes | Fair–Good |
| Rio de Janeiro Rocinha | Brazil | Most-mapped favela globally | Very good |
| Rio de Janeiro Complexo do Alemão | Brazil | HOT Tasking Manager coverage | Good |
| Lima Villa El Salvador | Peru | Planned informal grid hybrid, interesting edge case | Good |
| Lima Comas | Peru | Less planned, more organic | Fair |
| Santiago La Pintana | Chile | Chilean población, OSM improving | Good |
| Caracas Petare | Venezuela | Massive; OSM patchy in interior | Fair |
| São Paulo Heliópolis | Brazil | Brasil mapping activities | Good |
| Fortaleza Bom Jardim | Brazil | Less studied, may be sparse | Fair |

**Coverage strategy:** use the HOT Tasking Manager completion map to select
specific districts with ≥70% road mapping completeness. The `overpy` / `osmnx`
query will succeed but return a sparse graph in under-mapped areas — sparse
graphs produce mostly-background tiles that pollute training. Filter: drop tiles
where total road pixel fraction < 3% of tile area (road rasterization gives a
usable signal; below 3% is just background noise).

**Landuse note:** `landuse=residential` dominates almost entirely with nearly
zero `commercial` or `industrial`. The B channel in cond will be nearly uniform.
This is fine — the ControlNet must learn to generate informal organic road
patterns given a mostly-residential cond input.

---

### Style 5: `east_asian_dense`

Fine-grain high-density grids. Includes Japanese *chō*-based subdivisions
(very small blocks, alleys = `highway=service`), Korean residential, and
Taiwanese districts.

| City | Country | Notes | OSM coverage |
|------|---------|-------|--------------|
| Tokyo Nerima ward | Japan | Mid-density residential, *chō* block structure | Excellent |
| Tokyo Suginami ward | Japan | Residential+commercial mix | Excellent |
| Osaka Naniwa/Nishi ward | Japan | Denser than Tokyo wards | Excellent |
| Kyoto Fushimi ward | Japan | Includes traditional *machi* blocks | Excellent |
| Seoul Mapo-gu | South Korea | Post-war grid + informal infill | Excellent |
| Seoul Nowon-gu | South Korea | 1980s suburban apartment + grid | Excellent |
| Busan Busanjin-gu | South Korea | Hillside + grid mix | Excellent |
| Taipei Zhongzheng District | Taiwan | Colonial Japanese grid preserved | Excellent |
| Taipei Neihu | Taiwan | Post-war dense grid | Excellent |
| Incheon Michuhol-gu | South Korea | Industrial + residential mix | Very good |
| Sapporo Toyohira-ku | Japan | Post-war Hokkaido grid | Excellent |
| Nagoya Midori-ku | Japan | Suburban Japanese grid | Excellent |

**OSM note:** Japan has extraordinary OSM coverage — among the best globally.
Korea and Taiwan are very good. China mainland is explicitly excluded: OSM
coverage is systematically incomplete due to legal restrictions on surveying
(China's Surveying and Mapping Law), and the geographic coordinate offset
(GCJ-02 vs WGS84) creates systematic position errors in OSM-derived data.

**Road channel note:** `highway=service` (alleys, driveways) is extremely common
in Japanese tiles and is binned into road class 1 (`residential`). Japanese
tiles will have the highest road pixel density of any style — likely 2-3× the
road fraction of US suburbs.

---

## 3. Data Acquisition Feasibility

### OSM Road Graph Coverage

| Style | OSM road quality | Key risks |
|-------|-----------------|-----------|
| `euro_grid` | Excellent in W/C Europe; Good in SE Europe | None for recommended cities |
| `medieval_organic` | Excellent for all recommended cities | Very short alleys may be missing in some cities |
| `soviet_microrayon` | Excellent in Baltic/Poland/Czech; Variable in Ukraine/Romania | Kyiv post-2022 quality unknown — consider dropping or verifying |
| `latam_informal` | Highly variable, use HOT completion % filter | Must apply road-density drop filter (see §2) |
| `east_asian_dense` | Excellent for Japan/Korea/Taiwan | Do not use Chinese mainland cities |

**Practical filter for sparse tiles:** in `data_pipeline/dataset.py`, add a
`min_road_fraction` parameter. Load the road `.npy`, compute
`(road[1:].sum(0) > 0).mean()` and skip tiles below threshold. Recommend 0.03
(3%) for informal styles, 0.05 for all others.

### OSM Landuse Polygon Coverage

Landuse coverage is globally inconsistent. The pipeline's 5 landuse categories
use these OSM tags:

```
residential  → landuse=residential, landuse=apartments
commercial   → landuse=commercial, landuse=retail
industrial   → landuse=industrial, landuse=warehouse
parkland     → landuse=park, recreation_ground, nature_reserve, forest, grass, meadow
agricultural → landuse=farmland, landuse=farmyard
```

**European cities:** `landuse=residential` and `landuse=commercial` are
well-populated. `leisure=park` is common but `landuse=park` sometimes not tagged;
the `LANDUSE_TAGS` query in `osm_layers.py` already includes
`"leisure": ["park", "recreation_ground", "golf_course"]` which captures this.
No changes needed.

**Latin American cities:** `landuse=residential` dominates. `landuse=commercial`
mapping is inconsistent (many shops are individual POIs, not area polygons).
The B channel in cond will be sparser but this is acceptable — the ControlNet
learns from what's there.

**East Asian cities (Japan):** Japan has good `landuse` coverage at the broad
polygon level but uses a different breakdown than Europe. `landuse=residential`
and `landuse=commercial` are present; `landuse=industrial` is well-tagged.
However, Japanese cities are dense enough that the `commercial` channel will be
unusually large (dense shop districts).

**Soviet microrayon:** `landuse=residential` covers the housing estates well.
Green space between slabs is often tagged `landuse=grass` or `leisure=park`.
The G channel (green/parks) will be high. Industrial zones are typically
adjacent, not mixed.

### Terrain Data (SRTM1)

`elevation_layer.py` downloads SRTM1 (30 m resolution) from
`elevation-tiles-prod` on AWS. SRTM1 covers 60°S–60°N with no gaps for any
recommended city. No terrain data changes needed.

One caveat: the current code downloads only the single SRTM tile for the tile
center point. If a data tile spans a SRTM tile boundary (e.g., a 2.56 km tile
near a 1°×1° SRTM grid edge), the elevation data may have a hard edge mid-tile.
For US Sun Belt cities this was never an issue. For cities near SRTM tile
boundaries (many European cities span boundaries more often due to smaller tiles),
consider upgrading to multi-tile fetch. This is a moderate-priority fix.

### Regional Tag Differences

The most important regional OSM tag differences:

| Region | Issue | Impact | Mitigation |
|--------|-------|--------|------------|
| Japan | `highway=service` accounts for alleys, driveways, parking lot aisles — very common | Road class 1 count inflated vs US | Acceptable; model learns Japanese road density natively |
| Germany | `landuse=allotments` (Kleingärten) very common; not in our `parkland` category | Allotments appear as uncategorized pixels | Low impact; add `allotments` to parkland category (see §6) |
| France/Spain | `landuse=farmland` common near tile edges in suburban peripheries | Agricultural channel active, looks like US rural | No issue; channel is present |
| Latin America | `landuse=residential` is tagged but often as a large undifferentiated polygon | B channel lower than visual reality | Accept; informal neighborhoods are genuinely residential-dominated |
| East Europe (Soviet) | `landuse=grass` between housing slabs well-tagged; G channel higher | Model should learn green-between-slabs as a style feature | Desired behavior |
| UK | `landuse=brownfield`, `landuse=construction` not in our taxonomy | Miscellaneous untagged pixels | Low impact; add `brownfield` → industrial as low-priority |

---

## 4. Architecture Decision

### Option A: One Multi-Style ControlNet (8-channel cond)

Add a style index as an 8th channel: `cond[7] = style_id / (N-1)` normalized to
`[0,1]`. The ControlNet encoder sees 8 channels. Requires re-training from
scratch (our current ControlNet head is initialized from SDXL UNet encoder
weights which expect 3-channel RGB input — changing to 8 channels invalidates
those weights).

Alternatively: encode style via the **text prompt** instead of a new channel.
SDXL already has a CLIP text encoder. Changing the prompt to
`"top-down raster of a European medieval road network, ..."` per-style is
architecturally free.

**Problem:** the postmortem established that *"The text-prompt is irrelevant in
this setup. ControlNet with a constant per-tile caption doesn't use the text
encoder for class differentiation — all of it comes from the ControlNet input."*
Text-based style conditioning would require actively re-training the model to
attend to the text encoder output for style, which is a different problem from
what our current setup does.

**Verdict on Option A:** The text-conditioning approach requires recovering
text-encoder guidance (non-trivial, high risk of style bleeding); the 8-channel
approach requires re-training the ControlNet head from scratch (loses the
US-suburbs warm start). Neither is appropriate for this phase.

### Option B: N Per-Style ControlNets Fine-Tuned from US Suburbs

Each new style is a fine-tuning job starting from `sdxl_cnet_v1`:
- Initialize new ControlNet from the US suburbs checkpoint
- Train on the new style's tiles for ~25 k steps (~35 A100-hours)
- Save as a separate `~3 GB` safetensors file

**Parameter count:** unchanged — still ~1.4 B trainable ControlNet params per
model. No architecture changes required.

**Compute budget for 5 new styles:**

| Style | Cities | Est. tiles | Est. A100-hours |
|-------|--------|-----------|-----------------|
| `euro_grid` | 12 | ~1,800 | 35 |
| `medieval_organic` | 12 | ~700 | 35 |
| `soviet_microrayon` | 12 | ~1,500 | 35 |
| `latam_informal` | 8 (filtered) | ~800 | 35 |
| `east_asian_dense` | 12 | ~2,000 | 35 |
| **Total** | | **~6,800 new tiles** | **175 A100-hours** |

**Storage:** 5 × ~3 GB = ~15 GB on HF Hub (plus ~45 GB scratch for checkpoints
during training, purged after upload).

**Inference deployment:** 5 model checkpoints loaded per-request. Since the
Cities Skylines mod uses a cloud inference endpoint, the deployment can load one
model at a time based on the user's style selection. Memory footprint: ~12 GB
VRAM per active model (same as current). No change to inference architecture.

**Recommendation: Option B.** Per-style ControlNets are the right call because:

1. The US suburbs prior is a useful warm start even for very OOD styles —
   the ControlNet head's early layers encode road-detection priors (parallel
   lines, junctions, hierarchy by color) that transfer across styles.
2. Each model is independently tunable. If `medieval_organic` under-trains,
   we run it for 10 k more steps without touching the other models.
3. No risk of style bleeding. A multi-style model trained on unbalanced data
   risks averaging medieval organic + suburban grid into something that looks
   like neither.
4. Incremental: we can ship `euro_grid` while `soviet_microrayon` is still
   training.
5. Text-based style conditioning is empirically broken in our setup per the
   postmortem. Recovering it is a separate research problem.

---

## 5. Phased Rollout

Order by: (a) visual payoff — how different from `us_suburb`, (b) OSM coverage
confidence, (c) training difficulty.

### Phase 1: `euro_grid` — add immediately

**Why first:** highest visual contrast with US suburbs + best OSM coverage of
any non-US style. US suburbs have no continuous block fronts, no mixed-use
ground floors, no Haussmann 8-story cornice lines reflected in road hierarchy.
The `euro_grid` produces distinctly different outputs because:
- Blocks are ~100 m × 100 m vs US ~200-300 m
- No cul-de-sacs
- `tertiary` and `secondary` roads form the primary grid (not arterials)
- `motorway` channel is absent inside most tiles (motorways ring cities, don't
  penetrate the grid)

The US model is *maximally wrong* on European grid input because it has never
seen tight uniform blocks without cul-de-sacs.

**Data work:** cities.yaml entry for each of 12 cities, run the standard
pipeline, prep dataset, single SLURM job.

### Phase 2: `soviet_microrayon`

**Why second:** visually very distinct from both US suburbs and European grid —
the superblock topology produces large empty areas between sparse collector roads,
which is completely OOD from the model trained on tight grids. The tile signature
looks like open-field terrain with a few roads floating in it.

Also structurally the most interesting from a Cities Skylines perspective —
superblock city designs are a popular CS challenge.

**OSM risk is low** for the Baltic/Polish/Czech subset.

### Phase 3: `medieval_organic`

**Why third:** high visual payoff but produces fewer tiles per city (small area
covered), so we need more cities to hit tile count targets. Data acquisition
is more labor-intensive (manual review of city boundaries to exclude non-medieval
areas). Run after Phase 2 infrastructure is proven.

### Phase 4: `east_asian_dense`

**Why fourth:** Japan/Korea OSM is excellent and the fine-grain alley structure
is visually compelling. But `east_asian_dense` requires verifying that Japanese
`highway=service` inclusion is sensible (it may produce extremely high road
density tiles that confuse the model). Test this in Phase 4 only after we know
our road-density filter works from Phase 1.

### Phase 5: `latam_informal`

**Why last:** highest OSM coverage uncertainty. Run after the pipeline
improvements needed for sparse-tile filtering (§6) are validated in earlier
phases. Also the most unusual in road palette distribution (nearly all class 1,
no class 4), which might require tuning training steps.

---

## 6. Data Pipeline Portability Check

The existing `data_pipeline/` was designed for US Sun Belt cities. Below are
specific breakage points for non-US regions, ordered by severity.

### Critical: PROMPT hardcoded to "US suburban" in `prep_flux_dataset.py`

```python
PROMPT = (
    "top-down satellite-style raster of a US suburban road network, "
    "high-contrast color-coded road class map, flat color, vector style, "
    "no texture, no shading"
)
```

This is written to every `meta/{city}_{id}.txt`. For non-US styles, the text
conditioning is weak (per postmortem) but baking "US suburban" into all metadata
for a European medieval training run is actively wrong — the 10% non-empty
caption dropout means the model will occasionally see "US suburban" during
medieval training and receive conflicting signal.

**Fix:** Parameterize `prep_flux_dataset.py` with a `--style` argument that
maps to a prompt template:

```python
PROMPTS = {
    "us_suburb":       "top-down satellite-style raster of a US suburban road network, ...",
    "euro_grid":       "top-down satellite-style raster of a European grid city road network, ...",
    "medieval_organic":"top-down satellite-style raster of a medieval European road network, ...",
    "soviet_microrayon":"top-down satellite-style raster of a Soviet housing estate road network, ...",
    "latam_informal":  "top-down satellite-style raster of a Latin American informal settlement road network, ...",
    "east_asian_dense":"top-down satellite-style raster of a dense East Asian city road network, ...",
}
```

Also update `cities.yaml` to include per-city `style:` annotation (already
has top-level `style: us_suburb`; make it per-city or per-file).

### Moderate: `LANDUSE_TAGS` missing regionally common tags in `osm_layers.py`

The `LANDUSE_TAGS` dict is `{"landuse": True, "leisure": ["park", "recreation_ground", "golf_course"]}`.
Missing for non-US regions:

- `landuse=allotments` — extremely common in Germany, Poland, Czech Republic
  (allotment gardens / Kleingärten). These visually break up residential areas.
  **Add to `parkland` category** (they're green space):
  ```python
  ("parkland", ["park", "recreation_ground", "nature_reserve", "forest",
                "grass", "meadow", "allotments", "village_green"]),
  ```
- `landuse=brownfield`, `landuse=construction` — common in Eastern Europe during
  post-Soviet redevelopment. Not critical but currently they appear as
  uncategorized. Add to `industrial` channel as a low-priority mapping.
- `landuse=cemetery` — common in European tiles. Currently uncategorized.
  Visually similar to parkland (green space). Consider adding to `parkland`.

### Moderate: elevation_layer.py downloads only the center SRTM tile

```python
name, lat_i, _lon_i, ns, _ew = _srtm_tile(center_lat, center_lon)
hgt_path = _download_hgt(name, lat_i, ns, cache_dir)
```

This loads a single 1°×1° SRTM tile. For cities near latitude/longitude degree
boundaries (common in dense European cities on river crossings), the 2.56 km
tile might straddle two SRTM tiles, leaving one half with wrong or zero elevation.

**Fix:** compute the four corners of the bounding box, identify all SRTM tiles
they fall in, download and mosaic them before reprojecting.

### Low: `_bbox_latlon` flat-earth approximation

```python
lat_deg = half_m / 111_000
lon_deg = half_m / (111_000 * math.cos(math.radians(center_lat)))
```

This is the same approximation used in both `osm_layers.py` and `road_layers.py`.
At ±45° latitude (most of Europe, most of Japan), the error is <0.5% — acceptable.
At 60°N (Helsinki, Tallinn), error is ~2% on longitude, which at 2.56 km tile
scale = ~50 m positional error. Fine for a generative model training set.

This would only matter at >70°N (Tromsø, Murmansk) — none of the recommended
cities are near that. No action needed.

### Low: `cities.yaml` has global `style: us_suburb`

The `style:` field at the top level is unused by the pipeline code (nothing in
`dataset.py` or `prep_flux_dataset.py` reads it), but it's a convention to set up.

**Fix:** add `style:` as a per-city field in YAML and read it in
`prep_flux_dataset.py` to select the prompt template. Also add a CLI arg
`--style` as override for bulk runs.

### Low: `dataset.py` has no sparse-tile filter

```python
self.samples.append((os.path.join(d, cf), os.path.join(d, rf)))
```

No filter on tile quality. For US suburbs this is fine (all tiles have roads).
For `latam_informal` and `medieval_organic` (small core area), some tiles will
be mostly background.

**Fix:** add `min_road_fraction` parameter to `RoadLayoutDataset.__init__`:

```python
if min_road_fraction > 0:
    road = np.load(rp)
    frac = (road[1:].sum(0) > 0).mean()
    if frac < min_road_fraction:
        continue
```

Recommended thresholds: 0.03 for informal, 0.04 for medieval, 0.05 for all
others.

### Non-issue: CRS / projection

`_bbox_latlon` computes bounding boxes in WGS84 (lat/lon degrees). OSMnx queries
use lat/lon natively. SRTM is WGS84. Rasterio reprojects to target CRS on load.
There are **no hardcoded UTM or US-specific projections** in the pipeline. The
pipeline is already projection-neutral.

### Non-issue: `ROAD_CHANNELS` tag completeness globally

The road type filter in `road_layers.py`:

```python
cf = '["highway"~"motorway|trunk|primary|secondary|tertiary|residential|unclassified|living_street|service|motorway_link|..."]'
```

These highway types exist in OSM globally. Non-US regions may use them in
different proportions (more `living_street` in Europe, more `unclassified`
in South Asia) but the tag set is complete. The binning into 5 classes
(`bg/residential/tertiary/primary/motorway`) may over-bin some European road
types (many European `secondary` roads map to class 3 `primary` — correct).
No changes needed.

---

## Summary Checklist for Implementation

**Before first non-US data acquisition run:**

- [ ] Parameterize `--style` in `prep_flux_dataset.py` with prompt map
- [ ] Add `allotments`, `cemetery`, `village_green` to `parkland` category in `osm_layers.py`
- [ ] Add `brownfield`, `construction` to `industrial` category in `osm_layers.py`
- [ ] Add `min_road_fraction` filter to `RoadLayoutDataset`
- [ ] Add `style:` per-city field to `cities.yaml` format, or create per-style YAML files
- [ ] Update `build_hf_manifest.py` to pass through style-appropriate prompts from `meta/` files

**Before training `euro_grid`:**

- [ ] Create `data_pipeline/cities_euro_grid.yaml` with Phase 1 cities
- [ ] Run pipeline, verify landuse coverage on 5-10 sample tiles (visual check)
- [ ] Confirm SRTM tiles download correctly for European lat/lon (test with Paris tile)
- [ ] Create `data/flux_cnet_euro_grid_hf/` with `build_hf_manifest.py`
- [ ] Update SLURM script with new `OUT_DIR` and `--resume_from_checkpoint` pointing to `sdxl_cnet_v1`

**Before training `medieval_organic`:**

- [ ] Audit tile yield per city (expect ~50-80 tiles vs 150 for suburban)
  and expand city list if total tiles <1,000
- [ ] Investigate SRTM multi-tile mosaic fix for cities near lat/lon boundaries
