# Multi-Style Expansion Plan

Produced 2026-05-31; revised 2026-05-31. Picks up from the working US suburbs
SDXL ControlNet (`jalengg/groundwork-sdxl-cnet-us-suburbs`, trained on 17 Sun
Belt cities, 25 k steps, ~35 A100-hours). Goal: expand to capture representative
global city morphologies for the Cities Skylines mod.

**Scope:** plan only — no data acquisition or training in this document.

---

## 1. Stylistic Taxonomy

### Literature Survey

The initial draft of this plan cited only four sources (Conzen, Hillier,
Marshall, Moudon). That's the Anglophone morphology canon but leaves out
major work on American urban structure, post-colonial urbanism, African cities,
and Latin American urban form. A broader survey:

**Form typologies and street structure:**

- **Kostof (1991, 1992)** — *The City Shaped* / *The City Assembled*. The most
  comprehensive historical typology. Kostof identifies: organic pattern, the
  grid, the city of grand design (baroque/Haussmann), the industrial city, the
  automobile city. His grid chapter distinguishes the Greek Hippodamian grid,
  the Roman castrum, the colonial Spanish grid, and the American section-line
  grid — these look similar at street scale but differ in block proportions,
  hierarchy, and plot structure.
- **Conzen (1960)** — "town plan" analysis: street pattern + plot pattern +
  building pattern. The "burgage cycle" and "morphological frame" concepts.
  His fringe belt concept explains why European cities have concentric rings of
  different morphological character.
- **Hillier & Hanson (1984), Hillier (1996)** — Space syntax. Global vs. local
  integration, axial maps, the "deformed grid." Medieval cities tend toward low
  global integration; grids tend toward high; housing estates have disconnected
  local structure.
- **Marshall (2005)** — *Streets & Patterns*. "Directional" (strong
  through-routes), "grid" (uniform), "organically planned" (irregular but
  coherent), "cellular" (superblocks, internal access only). Directly applicable
  at our tile scale.
- **Moudon (1994)** — Morphological periods tied to transport technology.
  Pre-automobile organic → streetcar suburb → automobile suburb. Each era leaves
  a visible physical trace.
- **Panerai, Castex & Depaule (1977/2004)** — *Urban Forms: The Death and Life
  of the Urban Block*. The evolution of the European perimeter block and its
  dissolution under CIAM modernism (superblocks). Directly explains the visual
  difference between `euro_grid` and `soviet_microrayon`.
- **Habraken (1998)** — *Structure of the Ordinary*. Type/fabric theory.
  Thematic structure (the controlled) vs. tissue structure (the ordinary).

**American urban structure:**

- **Vance (1990)** — *The Continuing City*. Traces the American grid from the
  Land Ordinance of 1785 through the streetcar suburb. The section-line grid
  (mile-square sections) is the organizing structure of most US cities —
  arterials at 1-mile intervals, internal curvilinear residential streets are
  a *deliberate design reaction* against the grid, not its absence.
- **Jackson (1985)** — *Crabgrass Frontier*. The political economy of US
  suburban sprawl. FHA underwriting guidelines (1934–1960s) explicitly required
  curvilinear streets and cul-de-sacs to qualify for mortgage insurance —
  this is why post-war suburbs look the way they do.
- **Southworth & Ben-Joseph (2003)** — *Streets and the Shaping of Towns and
  Cities*. The evolution from grid (pre-1920) → curvilinear (1920–1960) →
  cul-de-sac (1960–present). The key distinction for our taxonomy: "curvilinear"
  ≠ "cul-de-sac." Curvilinear streets are winding but through-connected;
  cul-de-sac streets terminate. Pre-war streetcar suburbs and LA's postwar
  inner neighborhoods are curvilinear-through; post-1970 Sun Belt suburbs
  are cul-de-sac-cellular.
- **Soja (1996, 2000)** — *Thirdspace*, *Postmetropolis*. Los Angeles as the
  paradigmatic postmodern city. The "exopolis" structure: decentralized
  employment nodes, edge cities, arterial commercial at section-line intervals,
  no dominant downtown. The LA basin's street network is structurally the
  section-line arterial grid with residential infill that is curvilinear but
  not cul-de-sac — unlike the Sun Belt suburbs that our current model was
  trained on.
- **Davis (1990)** — *City of Quartz*. The security/fortress urbanism of LA's
  neighborhoods, which includes fortified residential enclaves that are partial
  cul-de-sac networks (gated). A complication for tile sampling.

**Latin American urban form:**

- **Griffin & Ford (1980), Ford (1996)** — *A model of Latin American city
  structure*. The Latin American city has a distinctive zonal structure: CBD +
  elite spine + concentric zones of decreasing formal quality → peripheral
  informal. The colonial Laws of the Indies grid (1573 Ordinances) governs the
  formal urban core (central plaza, orthogonal blocks ~80–100 m). The periphery
  grows organically under different logic entirely.
- **Hardoy (1982)** — *Urbanization in Latin America*. Pre-Colombian and colonial
  urban forms. Distinguishes indigenous organic urban forms (Tenochtitlan) from
  Spanish imposition of the orthogonal grid.
- **Caldeira (2000)** — *City of Walls* (São Paulo). Fortress urbanism, walled
  condominiums, and the spatial politics of exclusion. Relevant for understanding
  why the formal/informal boundary is so sharp in Brazilian cities.
- **Roy (2011)** — *Slumdog Cities*. Global comparison of informal urbanization
  processes. Shows that LatAm favelas, South Asian informal settlements, and
  African informal settlements have structurally different growth patterns.
- **Davis (2006)** — *Planet of Slums*. Global survey of informal settlement
  morphology and density. Key point: favelas in Rio/Medellín grow up steep
  hillsides as terrain-constrained organic structures; African informal
  settlements (Nairobi, Lusaka) often grow on flat terrain as planned-but-
  unserviced subdivisions or spontaneous flat-terrain organic settlements.

**African urban form:**

- **UN-Habitat (2014)** — *The State of African Cities*. Comprehensive survey.
  Identifies four broadly different African urban morphological contexts: (1)
  colonial-planned formal cities (Nairobi CBD, Harare, Accra); (2) indigenous
  West African compound city (Ibadan, Kano, Lagos Island); (3) informal
  peripheral settlements (Kibera, Mathare, Khayelitsha); (4) post-independence
  planned townships (Soweto, Cape Flats).
- **Rakodi (1997)** — *The Urban Challenge in Africa*. Argues that African cities
  cannot be understood through a single "African city" archetype — there are at
  minimum three distinct morphological families (Anglophone colonial, Francophone
  colonial, and indigenous).
- **Fourchard (2011)** — *Urban History of Sub-Saharan Africa*. The South African
  township (apartheid-era formal planning) is structurally distinct from both
  informal settlements and indigenous organic cities. It uses a regular
  orthogonal grid with very small plots — visually similar to American urban
  grid but at much smaller scale.
- **Myers (2011)** — *African Cities*. Challenges the idea that African cities
  are defined by informality. Many major African cities (Kigali, Addis Ababa)
  have undergone massive formal planning transformations in the 2000s–2020s.

**Key insight from the literature for our taxonomy:** the error in the first
draft was treating "Africa" and "LatAm informal" as single categories. The
literature shows three distinct African morphologies (colonial grid, informal
flat-terrain, indigenous organic) and two distinct LatAm morphologies
(Laws-of-the-Indies colonial grid, terrain-constrained informal). Additionally,
the US has three distinct morphologies within Moudon's automobile era:
pre-war grid, inner curvilinear (LA-type), and post-war cul-de-sac (existing
training data).

### Proposed Style Classes

The organizing question remains: **what is actually distinguishable in a
512×512 px, 5 m/px raster of road networks?** Two styles that look the same
at this scale don't need separate training runs.

At 512 px × 5 m/px = 2.56 km tiles, the distinguishable features are:
- Road density (total road pixels / tile area)
- Connectivity (intersection density, dead-end fraction)
- Block size (typical void area between roads)
- Hierarchy presence (which of the 5 road classes appear)
- Regularity (parallel lines, right angles vs. organic)
- Terrain coupling (roads following contours vs. cutting through)

With those axes, the proposed taxonomy expands from 6 to **11 styles** (10 new
+ existing):

| # | Style ID | Family | Road signature at 5 m/px |
|---|----------|--------|--------------------------|
| 0 | `us_suburb` | US | **Existing.** Low density, cul-de-sac terminals, strong class-3/4 arterial, nearly no class-2 through-grid |
| 1 | `us_grid` | US | Medium-high density, uniform right-angle grid, class-1/2/3 all present, alley network in many cities |
| 2 | `us_arterial` | US | Class-3 arterials at regular ~1.6 km intervals, internal curvilinear class-1 that are through-connected (not cul-de-sac) |
| 3 | `euro_grid` | European | Dense medium blocks ~100–150 m, class-2 roads form the primary grid, wide class-3 boulevard axis, no cul-de-sacs |
| 4 | `medieval_organic` | European | Irregular, winding, no parallel lines, very high class-1/2 intersection density, small irregular blocks |
| 5 | `soviet_microrayon` | Post-socialist | Very low density, large open voids, class-1 loops internal to superblocks, class-3 collector ring |
| 6 | `latam_colonial` | Latin American | Regular orthogonal grid ~80–100 m blocks, dense class-2 network, central plaza void, flat terrain typically |
| 7 | `latam_informal` | Latin American | Dense class-1 organic mesh, terrain-following winding streets, very high dead-end fraction, minimal class-3/4 |
| 8 | `africa_informal` | African | Moderate-density class-1 organic mesh, typically flat terrain (unlike LatAm hillside), compound-wall topology, more regular than LatAm informal |
| 9 | `africa_township` | African | Small regular orthogonal grid (apartheid-era South African township), very small blocks ~50 m, class-1 dominant, uniform |
| 10 | `east_asian_dense` | East Asian | Very high density, class-1/service alley network, fine-grain near-grid ~60–80 m blocks, minimal class-4 |

**11 styles total** including existing. At ~35 A100-hours each, 10 new runs =
350 GPU-hours.

### Why Each Style Is Distinguishable

**`us_grid` vs `us_suburb` vs `us_arterial`:**
- `us_suburb` (existing): tile contains cul-de-sacs visible as terminal
  branches, no through-streets in residential zones, strong motorway presence
- `us_grid`: all streets are through-connected at right angles; alleys appear
  as dense class-1 mesh behind the main grid; block size ~60–100 m (smaller
  than suburb superblocks)
- `us_arterial`: wide class-3 arterials visible at regular ~1.6 km intervals,
  internal residential streets winding but passing through (no terminals at
  residential streets), large internal blocks ~300–500 m

These are distinguishable because the cul-de-sac terminal signature and the
1-mile arterial grid rhythm are both visible at 5 m/px resolution.

**`latam_colonial` vs `euro_grid`:**
The first draft wrongly folded these together. They are distinguishable:
- `latam_colonial`: block size ~80–100 m (slightly smaller than Haussmann's
  ~120–150 m), uniform orthogonal grid covering the entire tile with minimal
  hierarchy variation, central plaza creates a characteristic void at the grid
  center, typically flat terrain → uniform G (green/parks) channel
- `euro_grid`: larger blocks, more hierarchy variation (Haussmann boulevards
  are visually wider than cross-streets), more complex landuse patterning
  (mixed-use dense ground floors = `commercial` B-channel alongside residential)

**`latam_informal` vs `africa_informal`:**
- `latam_informal`: frequently steep terrain → roads visible following contours
  diagonally across the tile; very high elevation channel variance
- `africa_informal`: typically flat terrain → elevation channel flat;
  compound-wall topology creates larger internal voids than LatAm favelas
  (compounds = walled clusters with one entrance, creating quasi-dead-ends at
  a different scale than hillside favelas)

**`africa_township` vs `us_grid`:**
- `africa_township`: extremely small blocks (~50 m vs. ~80 m US grid), much
  lower road class diversity (almost entirely class-1), little to no commercial
  landuse B-channel, very uniform residential G-channel
- `us_grid`: alley network creates a two-tier density pattern, more landuse
  diversity, class-2/3 arterials present at regular intervals

### Styles Deferred

- **West African compound city** (`ibadan_organic`, Kano traditional core):
  structurally interesting (compound walling creates a topology distinct from
  both medieval European organic and African informal), but OSM coverage of
  traditional cores in Ibadan, Kano, and Lagos Island is poor to fair. Defer
  to v2 pending OSM improvement or dedicated mapping campaign.
- **MENA medina** (Fez, Marrakech): very high dead-end fraction (Islamic
  *khuttas*) would create a distinctive tile signature, but OSM street coverage
  inside medinas is patchy. Defer to v2.
- **Post-independence African formal** (Kigali Vision 2020 new developments,
  Addis Ababa Bole): recently-planned African urban extensions resemble
  `soviet_microrayon` or `euro_grid` depending on the project. Not a distinct
  style; sample from existing styles for these areas.
- **South/Southeast Asian mixed** (Mumbai, Jakarta): very high OSM tag
  variability and coverage gaps. Defer to v2.
- **Australian/UK suburb**: intermediate between `us_suburb` and
  `medieval_organic`. Too ambiguous to train a clean model on.

---

## 2. City List Per Style

### Style 1: `us_grid`

Pre-automobile American city grids. Sample from residential/commercial
neighborhoods, not downtown cores (where office towers dominate and landuse
patterns are unusual). Avoid cities already in `us_suburb` training data.

| City | State | Notes | OSM coverage |
|------|-------|-------|--------------|
| Portland (Sellwood, Woodstock) | OR | Textbook American grid with alley system, residential | Excellent |
| Denver (Capitol Hill, Highlands) | CO | 1880s grid, alleys common, good terrain variation | Excellent |
| Chicago (Bridgeport, Pilsen) | IL | The canonical American alley grid; dense class-1 | Excellent |
| Philadelphia (West Philly, Fishtown) | PA | Rowhouse grid, very uniform blocks ~75 m | Excellent |
| Minneapolis (Powderhorn, Longfellow) | MN | Clean American grid, good landuse mix | Excellent |
| Salt Lake City (Sugar House, Liberty) | UT | Very wide grid (132-foot streets from Plat of Zion); distinctive | Excellent |
| Cincinnati (Clifton, Norwood) | OH | Hilly American grid, tests terrain interaction | Excellent |
| Kansas City (Hyde Park, Westport) | MO | Streetcar-era grid | Excellent |
| Baltimore (Hampden, Remington) | MD | Dense rowhouse grid, clear landuse delineation | Excellent |
| St. Louis (Maplewood, Webster Groves) | MO | Inner-ring streetcar suburb grid | Excellent |
| Buffalo (Elmwood, Allentown) | NY | Olmsted parkways embedded in grid | Excellent |
| Milwaukee (Bay View, Walker's Point) | WI | Dense Great Lakes grid, industrial+residential mix | Excellent |

**Tile strategy:** use city district boundaries, not admin city limits. The admin
limits include modern suburbs that look like `us_suburb`. Sample within the
1880–1930 built area (roughly inside the pre-WWII expansion ring).

**OSM note:** US cities have excellent OSM coverage. Alley systems are tagged as
`highway=service` in OSM, which bins into road class 1 — alley-heavy cities
(Chicago, Denver, KC, Baltimore) will have noticeably higher class-1 density
than non-alley cities (Portland, Minneapolis). This is correct behavior.

---

### Style 2: `us_arterial`

The LA-type inner suburban zone: section-line arterials at ~1.6 km intervals,
large residential superblocks with curvilinear through-streets (not cul-de-sacs).
Post-war (1945–1975) automobile-era but before the switch to pure cul-de-sac.
Associated with Soja's "postmodern city" and Vance's "automobile city" without
the full cul-de-sac commitment.

| City | State | Notes | OSM coverage |
|------|-------|-------|--------------|
| Los Angeles (Culver City, Mar Vista) | CA | Canonical example; flat residential with arterial rhythm | Excellent |
| Los Angeles (Palms, Westchester) | CA | More uniform, tests variation within LA basin | Excellent |
| Los Angeles (Van Nuys, Reseda) | CA | San Fernando Valley, flat section-line structure | Excellent |
| Glendale (residential interior) | CA | Avoids downtown, captures the inner-suburban zone | Excellent |
| Burbank (residential east) | CA | Industrial+residential mix, good B-channel variation | Excellent |
| Phoenix (Arcadia neighborhood) | AZ | Pre-cul-de-sac Phoenix; different from Henderson training data | Excellent |
| Tucson (Sam Hughes, Rincon Heights) | AZ | Streetcar-era Tucson, curvilinear through-streets | Excellent |
| Long Beach (Bixby Knolls, Wrigley) | CA | Flat residential, clear arterial structure | Excellent |
| San Jose (Willow Glen, Cambrian) | CA | Silicon Valley inner suburban | Excellent |
| Sacramento (Elmhurst, Oak Park) | CA | Grid+curvilinear hybrid zone | Excellent |
| Albuquerque (Nob Hill, Ridgecrest) | NM | Flat SW inner suburban, minimal terrain variation | Excellent |
| Denver (Barnum, Harvey Park) | CO | Post-war Denver inner ring, curvilinear but through | Excellent |

**What makes this different from `us_suburb`:** In training data from
Henderson NV / Chandler AZ, residential streets terminate at cul-de-sacs.
In these cities, residential streets wind but connect to the next arterial.
The OSM `highway=residential` network will show a different connectivity
graph — fewer dead-ends, more 4-way or T intersections at non-arterial
junctions.

**OSM note:** Excellent for all. The critical feature — through-connectivity
of residential streets — is well-captured in OSM because mappers trace
drivable streets regardless of curvilinearity.

---

### Style 3: `euro_grid`

Planned bourgeois grids: Haussmann, Cerdà, Gründerzeit, and their
equivalents. Revised to exclude LatAm colonial cities (now their own style)
but can include colonial African planned cities as secondary examples.

| City | Country | Notes | OSM coverage |
|------|---------|-------|--------------|
| Paris 13e/14e | France | Haussmann grid away from tourist core | Excellent |
| Barcelona Eixample | Spain | Cerdà grid with chamfered corners, textbook | Excellent |
| Brussels inner ring | Belgium | Haussmann-influenced grid + fringe | Excellent |
| Vienna Margareten/Favoriten | Austria | Gründerzeit ring-road grid | Excellent |
| Milan Porta Vittoria/Navigli | Italy | Radial+grid, varied landuse | Excellent |
| Lyon Part-Dieu | France | Modern French grid | Excellent |
| Bordeaux Saint-Michel | France | Haussmann + organic fringe mix | Excellent |
| Porto Bonfim | Portugal | Hilly grid, terrain interaction | Good |
| Athens Kallithéa | Greece | Dense Athenian grid, no motorway penetration | Good |
| Turin Crocetta/Nizza | Italy | Rational Piedmontese grid | Excellent |
| Addis Ababa Bole (formal areas) | Ethiopia | Post-2000 planned African grid, good OSM | Good |
| Nairobi Westlands/Parklands | Kenya | Colonial British grid, good OSM, diverse landuse | Very good |

**Note on African colonial grids:** Nairobi Parklands and Addis Ababa's formal
planned zones have the same visual signature as European planned grids — regular
~120–150 m blocks, class-2/3 road hierarchy. Including them in `euro_grid`
training data adds morphological diversity without requiring a separate training
run. These are *not* the same as `africa_township` (which is smaller-block,
class-1 dominant).

---

### Style 4: `medieval_organic`

Pre-automobile organic European cores. Sample only from within the medieval
walled boundary or its functional equivalent.

| City | Country | Notes | OSM coverage |
|------|---------|-------|--------------|
| Bologna centro storico | Italy | Largest medieval center in Italy | Excellent |
| Bruges | Belgium | UNESCO site, exceptional OSM | Excellent |
| Chester | UK | Roman+medieval walled city | Excellent |
| Colmar | France | Intact Alsatian medieval core | Excellent |
| Siena | Italy | UNESCO, minimal car penetration | Excellent |
| Toledo | Spain | Islamic+medieval palimpsest | Very good |
| Ghent Patershol | Belgium | Irregular medieval quarter | Excellent |
| York Shambles | UK | Well-mapped | Excellent |
| Regensburg Altstadt | Germany | UNESCO Danube crossing city | Excellent |
| Lucca | Italy | Intact Roman→medieval street grid | Excellent |
| Tallinn Vanalinn | Estonia | Northern European medieval, excellent OSM | Excellent |
| Strasbourg Grande Île | France | UNESCO island medieval core | Excellent |

**Tile strategy:** medieval cores are small (~50–80 tiles per city vs. ~150 for
suburban). Tile centers within medieval perimeter only. More cities compensate
for lower tile yield.

---

### Style 5: `soviet_microrayon`

Eastern European and Soviet-era housing estates.

| City | Country | Notes | OSM coverage |
|------|---------|-------|--------------|
| Warsaw Ursynów | Poland | Large, well-documented | Excellent |
| Prague Jižní Město | Czech Republic | Canonical panelák estate | Excellent |
| Bratislava Petržalka | Slovakia | Europe's largest housing estate | Excellent |
| Budapest Újpalota | Hungary | Hungarian housing estate | Very good |
| Vilnius Fabijoniškės | Lithuania | Well-mapped | Excellent |
| Tallinn Lasnamäe | Estonia | Baltic Soviet estate | Excellent |
| Bucharest Drumul Taberei | Romania | Ceaușescu-era | Good |
| Leipzig Grünau | Germany | DDR Plattenbau, excellent German OSM | Excellent |
| Erfurt Johannesvorstadt | Germany | East German estate | Excellent |
| Krakow Nowa Huta | Poland | Planned socialist city | Excellent |
| Riga Purvciems | Latvia | Soviet-era Latvian estate | Excellent |
| Kyiv Troieshchyna | Ukraine | Large Soviet estate; verify OSM quality post-2022 | Variable |

---

### Style 6: `latam_colonial`

The Laws of the Indies colonial grid (Spanish Royal Ordinances of 1573).
Characteristic features: strict orthogonality, ~80–100 m blocks (half the
size of Haussmann), central plaza void, typically flat terrain, dense
uniform `residential`/`commercial` B-channel distribution.

This is morphologically distinct from `euro_grid` (see §1 taxonomy note).
The Spanish colonial grid predates Haussmann by 250 years, produces smaller
tighter blocks, and the central-plaza void is a recurring feature.

| City | Country | Notes | OSM coverage |
|------|---------|-------|--------------|
| Buenos Aires San Telmo/Almagro | Argentina | Colonial+19c grid, best OSM in LatAm | Very good |
| Montevideo Ciudad Vieja/Cordón | Uruguay | Clean colonial grid | Very good |
| Bogotá La Candelaria/Chapinero | Colombia | Colonial core + planned extensions | Good |
| Lima Cercado/Barranco | Peru | Colonial Lima, flat, dense | Good |
| Oaxaca Centro | Mexico | UNESCO; small-city colonial grid, plaza-dominant | Very good |
| Guadalajara Analco/San Juan | Mexico | Large colonial grid city | Good |
| Cusco Centro | Peru | Inca+Spanish grid hybrid, distinctive OSM | Very good |
| Santiago (Barrio Italia, Yungay) | Chile | 19c colonial extension grid | Very good |
| Quito La Mariscal/La Floresta | Ecuador | UNESCO; Andean colonial grid | Good |
| Córdoba (Argentina) Nueva Córdoba | Argentina | Good OSM, classic grid | Very good |
| Asunción Centro | Paraguay | Colonial grid, improving OSM | Fair–Good |
| Valparaíso Plan area | Chile | Port grid on flat lower city (not hillside) | Good |

**OSM note:** Argentina, Uruguay, and Chile have the best LatAm OSM coverage.
Bogotá, Quito, Lima are acceptable. Use HOT Tasking Manager completion maps
before finalizing city list. Prefer larger cities for tile count.

**Tag note:** `landuse=commercial` and `landuse=retail` are less consistently
tagged as area polygons in LatAm — shops exist as POIs, not mapped areas.
The B channel will be sparser than European grids for equivalent commercial
activity. This is a dataset characteristic, not a pipeline bug.

---

### Style 7: `latam_informal`

Favelas, comunas, villas miserias, asentamientos. Characterized by
terrain-constrained organic growth, steep elevation channel variation, very
high class-1 road density, minimal class-3/4 presence.

| City | Country | Notes | OSM coverage |
|------|---------|-------|--------------|
| Medellín Comunas 1-3 (NE hills) | Colombia | OpenCities Medellín significantly improved OSM | Good–Very good |
| Rio de Janeiro Rocinha | Brazil | Most-mapped favela globally | Very good |
| Rio de Janeiro Complexo do Alemão | Brazil | HOT Tasking Manager coverage | Good |
| Lima Villa El Salvador | Peru | Planned informal grid hybrid | Good |
| Santiago La Pintana/El Bosque | Chile | Chilean población, OSM improving | Good |
| São Paulo Heliópolis | Brazil | Brasil mapping activities | Good |
| Bogotá Ciudad Bolívar | Colombia | Large, sample lower slopes for OSM quality | Fair–Good |
| Caracas Petare | Venezuela | Massive; exterior zones better mapped | Fair |
| Fortaleza Bom Jardim | Brazil | Less studied | Fair |
| Medellin El Popular/Santa Cruz | Colombia | Additional high-hill communes | Good |

**Tile filter:** drop tiles with road pixel fraction <3% (under-mapped areas
produce mostly-background tiles; see §6). Kyiv data quality note does not apply
here but Caracas and Fortaleza need pre-sampling verification.

---

### Style 8: `africa_informal`

Sub-Saharan African informal settlements. Key morphological difference from
`latam_informal`: typically **flat terrain** (Kibera, Mathare, Khayelitsha all
on flat or gently rolling ground), compound-wall structure creates larger walled
voids than LatAm favelas, road class distribution is even flatter (almost
entirely class-1 tracks and paths, some of which are tagged `highway=path`
rather than `highway=residential` in OSM).

**Important OSM caveat:** `highway=path` and `highway=track` (common in African
informal settlements) are *not* included in the road channel filter in
`road_layers.py`. A significant fraction of internal circulation in these
settlements would be invisible to the pipeline. This requires a pipeline change:
either include `path`+`track` as class-1 roads for this style, or accept
that the pipeline captures only the vehicle-accessible network (which may still
be sufficient to capture the morphological signature).

| City | Country | Notes | OSM coverage |
|------|---------|-------|--------------|
| Nairobi Kibera | Kenya | Most-mapped informal settlement globally; HOT | Very good |
| Nairobi Mathare | Kenya | Adjacent to Kibera, different topology | Good |
| Nairobi Korogocho | Kenya | Smaller, denser | Good |
| Dar es Salaam Manzese | Tanzania | Flat informal, well-mapped | Good |
| Dar es Salaam Tandale | Tanzania | Dense, flat | Good |
| Kampala Kisenyi/Katanga | Uganda | Hilly-ish informal, tests variation | Good |
| Lagos Makoko (accessible areas) | Nigeria | Water-adjacent informal; partial | Fair |
| Accra Nima/Mamobi | Ghana | Flat informal, improving OSM | Fair–Good |
| Maputo Chamanculo C/D | Mozambique | Flat, HOT coverage | Good |
| Lusaka Kanyama | Zambia | Large flat informal settlement | Good |
| Khayelitsha (Cape Town) | South Africa | South African township+informal mix; see also `africa_township` | Very good |
| Harare Epworth | Zimbabwe | Peri-urban informal | Fair |

---

### Style 9: `africa_township`

South African apartheid-era townships and similar post-colonial planned-but-
unserviced residential estates. Structurally distinct from informal: regular
small orthogonal grid (~40–60 m blocks), class-1 dominant but through-connected
(unlike cul-de-sac suburbs), very uniform residential landuse, typically flat.

Visible difference from `us_grid`: much smaller blocks (US grid ~75–100 m);
almost no commercial/industrial B-channel diversity (entire township is
`landuse=residential`); no alley system.

| City | Country | Notes | OSM coverage |
|------|---------|-------|--------------|
| Soweto (formal sections: Meadowlands, Diepkloof) | South Africa | Canonical township, very well mapped | Excellent |
| Khayelitsha (formal grid sections) | South Africa | Separate tiles from informal sections | Very good |
| Mitchell's Plain (Cape Town) | South Africa | Large township, flat grid | Very good |
| Tembisa (Ekurhuleni) | South Africa | Large Gauteng township | Very good |
| Mamelodi (Pretoria) | South Africa | Apartheid-era formal | Good |
| Harare Mbare/Highfield | Zimbabwe | Colonial Rhodesian township grid | Good |
| Lusaka Chilenje/Mandevu | Zambia | Zambian township grid | Fair–Good |
| Nairobi Eastlands (Buruburu) | Kenya | Post-independence planned estate | Good |
| Bulawayo Makokoba | Zimbabwe | One of the oldest African townships | Good |

**OSM note:** South African townships are exceptionally well-mapped (active
South African OSM community). Zimbabwe and Zambia are fair.

**Pipeline note:** The `africa_township` small block size (~50 m) means tiles
will have higher road density than `us_grid` even though they appear visually
sparser (narrower streets, class-1 only). The road rasterization line width
(`class_widths = {1: 2, ...}` pixels) remains appropriate.

---

### Style 10: `east_asian_dense`

Fine-grain high-density grids. Japan/Korea/Taiwan only; mainland China excluded
(GCJ-02 coordinate offset + legally restricted surveying creates systematic
OSM errors; see §3).

| City | Country | Notes | OSM coverage |
|------|---------|-------|--------------|
| Tokyo Nerima ward | Japan | Mid-density residential *chō* | Excellent |
| Tokyo Suginami ward | Japan | Residential+commercial mix | Excellent |
| Osaka Naniwa/Nishi ward | Japan | Denser than Tokyo wards | Excellent |
| Kyoto Fushimi ward | Japan | Traditional *machi* blocks | Excellent |
| Seoul Mapo-gu | South Korea | Post-war grid + informal infill | Excellent |
| Seoul Nowon-gu | South Korea | 1980s suburban apartment + grid | Excellent |
| Busan Busanjin-gu | South Korea | Hillside + grid mix | Excellent |
| Taipei Zhongzheng District | Taiwan | Japanese colonial grid preserved | Excellent |
| Taipei Neihu | Taiwan | Post-war dense grid | Excellent |
| Sapporo Toyohira-ku | Japan | Post-war Hokkaido grid | Excellent |
| Nagoya Midori-ku | Japan | Suburban Japanese grid | Excellent |
| Incheon Michuhol-gu | South Korea | Industrial + residential mix | Very good |

---

## 3. Data Acquisition Feasibility

### OSM Road Graph Coverage

| Style | OSM road quality | Key risks |
|-------|-----------------|-----------|
| `us_grid` | Excellent | None |
| `us_arterial` | Excellent | None |
| `euro_grid` | Excellent W/C Europe; Good SE Europe | None for recommended cities |
| `medieval_organic` | Excellent for all recommended cities | Narrow alleys sometimes missing |
| `soviet_microrayon` | Excellent Baltic/Poland/Czech; Variable Ukraine/Romania | Verify Kyiv post-2022 |
| `latam_colonial` | Very good Argentina/Uruguay/Chile; Good elsewhere | Pre-sample Asunción; drop if sparse |
| `latam_informal` | Variable; use HOT completion % filter | Apply 3% road-density drop filter |
| `africa_informal` | Variable; HOT campaigns helped Nairobi/DSM | `highway=path` not in pipeline; see §6 |
| `africa_township` | Excellent South Africa; Good elsewhere | Zambia/Zimbabwe need verification |
| `east_asian_dense` | Excellent Japan/Korea/Taiwan | Do not use Chinese mainland |

### OSM Landuse Coverage

No new issues beyond what was identified for the original 6 styles. The new US
and African styles introduce one additional tag gap:

- **US cities with alleys:** `highway=service` roads include both alleys and
  driveways/parking aisles. This inflates class-1 count. No fix needed — the
  model learns alley density as a style feature.
- **Africa informal + township:** `landuse=residential` is the dominant tag
  and is consistently mapped, but `landuse=commercial` is almost absent (shops
  are POIs, not area polygons). The B channel will be near-zero for most African
  tiles. This is the actual ground truth — these areas are residential-only —
  so it's not a pipeline defect.

### Terrain Data

SRTM1 (30 m, global 60°S–60°N) covers all recommended cities. No changes.

`latam_informal` (hillside favelas) will exercise the elevation channel heavily —
tile-level normalization in `encode_cond` means a steep hillside tile will
use the full `[0,1]` R channel range, while flat African tiles will be ~uniform.
This is the desired behavior.

### Regional Tag Differences

| Region | Issue | Impact | Mitigation |
|--------|-------|--------|------------|
| Germany/Poland | `landuse=allotments` very common | Parkland channel sparse without it | Add to `parkland` tags (§6) |
| Africa informal | `highway=path`/`track` = primary circulation | Road network appears sparse | Add path/track to class-1 for this style; see §6 |
| Japan | `highway=service` = alleys (very dense) | Class-1 inflated; correct behavior | None needed |
| LatAm all | `landuse=commercial` rare as polygon | B channel sparse | Accept; informative feature |
| South Africa | `landuse=residential` dominates uniformly | B channel flat for entire township | Accept; correct |

---

## 4. Architecture Decision

*(Unchanged from first draft — reasoning still holds with expanded style set.)*

### Option A: One Multi-Style ControlNet

Adding a style-code channel or relying on text conditioning runs into the same
problem at 10 styles as at 5: the text encoder's contribution is empirically
weak in this setup (postmortem: *"The text-prompt is irrelevant"*). An 8th
channel style code requires re-training the ControlNet head from scratch, losing
the US-suburbs warm start. Option A is not appropriate for this phase.

### Option B: N Per-Style ControlNets

**Recommendation: 10 per-style ControlNets, each fine-tuned from `sdxl_cnet_v1`.**

Updated compute budget:

| Style | Cities | Est. tiles | A100-hours |
|-------|--------|-----------|------------|
| `us_grid` | 12 | ~1,800 | 35 |
| `us_arterial` | 12 | ~1,800 | 35 |
| `euro_grid` | 12 | ~1,800 | 35 |
| `medieval_organic` | 12 | ~700 | 35 |
| `soviet_microrayon` | 12 | ~1,500 | 35 |
| `latam_colonial` | 12 | ~1,600 | 35 |
| `latam_informal` | 10 (filtered) | ~800 | 35 |
| `africa_informal` | 12 (filtered) | ~700 | 35 |
| `africa_township` | 9 | ~1,200 | 35 |
| `east_asian_dense` | 12 | ~2,000 | 35 |
| **Total** | | **~13,900 tiles** | **350 A100-hours** |

Storage: 10 × ~3 GB = ~30 GB on HF Hub.

---

## 5. Phased Rollout

Order by: (a) visual distance from `us_suburb`, (b) OSM confidence, (c)
pipeline changes required. Styles requiring new pipeline work (§6) deferred
until those changes are validated.

### Phase 1: `us_grid` — immediate

**Why first:** highest OSM confidence (US), requires zero pipeline changes
(same OSM tag set, same SRTM range), and is visually very OOD from `us_suburb`
(through-connected grid, alley network, small blocks). The model has never seen
a through-connected residential grid — all training data has cul-de-sac
topology. Validating the pipeline on a US city before attempting non-US cities
is also a useful sanity check: we can visually compare output on Chicago
neighborhoods against training data we know well.

### Phase 2: `us_arterial`

Same reasoning as Phase 1. US data, zero pipeline risk. The arterial grid
rhythm is a very different topological signal from both `us_suburb` and
`us_grid`. This is the most common morphology in the western US that the
current model gets wrong (most of suburban LA, inner Phoenix, inner Tucson
tiles fed to the current model would produce `us_suburb`-style cul-de-sac
outputs).

### Phase 3: `euro_grid`

First non-US style. Requires prompt parameterization fix (§6). Best OSM of any
non-US style, most visually distinct from all US styles (no cul-de-sacs, uniform
blocks, boulevard hierarchy). The US model is *maximally wrong* on European grid
because it predicts cul-de-sac terminals wherever it sees residential inputs.

### Phase 4: `latam_colonial`

Colonial grid after European grid. The distinction is visible at 5 m/px (smaller
blocks, plaza void). OSM is good for Argentina/Uruguay/Chile. Requires verifying
that the prompt correctly signals "colonial grid" vs "European grid" — may reveal
whether text conditioning has any discriminative power for structurally-similar
styles.

### Phase 5: `soviet_microrayon`

Superblock topology after we have grids working. German DDR cities first (best
OSM). Most visually distinct from everything: the large void between roads is
unlike any grid style.

### Phase 6: `medieval_organic`

Small tile yield per city means more data acquisition overhead. Run after
Phase 5 validates the per-style training pipeline.

### Phase 7: `east_asian_dense`

Excellent data; requires verifying that Japanese alley density doesn't cause
rasterization artifacts (very high class-1 pixel fraction). Defer until Phase 4
sparse-tile filter is validated.

### Phase 8: `africa_township`

South African data is excellent. Run after pipeline's `highway=path` issue is
addressed (§6) so we understand what road class handling looks like for Africa.
Township OSM doesn't use paths heavily so this could be Phase 6 — but sequencing
it after informal styles avoids confusion between township and informal tiles.

### Phase 9: `latam_informal`

Requires HOT completion filter and sparse-tile filter. Run after Phase 3
validates non-US pipeline.

### Phase 10: `africa_informal`

Most pipeline-fragile style (path/track issue) and most variable OSM. Last.

---

## 6. Data Pipeline Portability Check

*(Expanded from first draft to cover new styles.)*

### Critical: PROMPT hardcoded in `prep_flux_dataset.py`

```python
PROMPT = (
    "top-down satellite-style raster of a US suburban road network, "
    "high-contrast color-coded road class map, flat color, vector style, "
    "no texture, no shading"
)
```

**Fix:** `--style` argument with a prompt map:

```python
PROMPTS = {
    "us_suburb":        "top-down raster of a US suburban road network, ...",
    "us_grid":          "top-down raster of an American urban grid city road network, ...",
    "us_arterial":      "top-down raster of an American inner suburban arterial grid road network, ...",
    "euro_grid":        "top-down raster of a European planned grid city road network, ...",
    "medieval_organic": "top-down raster of a medieval European road network, ...",
    "soviet_microrayon":"top-down raster of a Soviet housing estate road network, ...",
    "latam_colonial":   "top-down raster of a Latin American colonial grid city road network, ...",
    "latam_informal":   "top-down raster of a Latin American informal settlement road network, ...",
    "africa_informal":  "top-down raster of a sub-Saharan African informal settlement road network, ...",
    "africa_township":  "top-down raster of a sub-Saharan African township road network, ...",
    "east_asian_dense": "top-down raster of a dense East Asian city road network, ...",
}
```

Also add `style:` per-city field to `cities.yaml` or use per-style YAML files.

### Critical (Africa styles): `highway=path`/`track` not in road filter

`road_layers.py` fetches only:
```
motorway|trunk|primary|secondary|tertiary|residential|unclassified|living_street|service|...
```

In African informal settlements, much of the internal circulation network is
tagged `highway=path` (footpath used by motorcycles) or `highway=track` (dirt
track). These are not in the filter and will be invisible to the pipeline.

**Fix option A:** Add `path` and `track` to road class 1 for `africa_informal`
style only (via a `--style`-conditional road filter in `road_layers.py`).

**Fix option B:** Accept that the pipeline captures vehicle-accessible roads
only. In Kibera, the mapped vehicle road network is sparse but structurally
representative. Given the postmortem lesson that the model doesn't need
perfectly complete data — it needs enough signal to learn the morphological
style — Option B may be acceptable. Test by visual inspection of 10 tiles
before deciding.

### Moderate: Missing landuse tags in `osm_layers.py`

Add to `LANDUSE_CATEGORIES`:

```python
("parkland", ["park", "recreation_ground", "nature_reserve", "forest",
              "grass", "meadow", "allotments", "village_green", "cemetery"]),
("industrial", ["industrial", "warehouse", "brownfield", "construction"]),
```

`allotments` is very common in German/Polish/Czech tiles (`soviet_microrayon`
phase). `cemetery` appears frequently in European and African tiles. `brownfield`
and `construction` are common in Eastern European tiles.

### Moderate: `RoadLayoutDataset` has no sparse-tile filter

**Fix:** add `min_road_fraction` to `dataset.py`:
```python
if min_road_fraction > 0:
    road = np.load(rp)
    frac = (road[1:].sum(0) > 0).mean()
    if frac < min_road_fraction:
        continue
```

Recommended thresholds: 0.03 for `latam_informal`/`africa_informal`, 0.04 for
`medieval_organic`, 0.05 for all others.

### Low: Single SRTM tile fetch in `elevation_layer.py`

The current code downloads only the SRTM tile for the tile center point. Tiles
near lat/lon degree boundaries may have an elevation discontinuity mid-tile.
More relevant for European cities (denser on SRTM tile boundaries) than US.

**Fix:** compute all four bbox corners, identify all SRTM tiles they fall in,
download and mosaic before reprojecting. Medium-priority before `euro_grid` run.

### Non-issues (unchanged)

- **CRS/projection:** pipeline is WGS84 throughout — no US-specific projections.
- **Road channel global coverage:** OSM highway types used are globally standard.
- **Flat-earth bbox approximation:** <2% error at any recommended city latitude.

---

## Summary Checklist

**Before first non-US data acquisition run (`euro_grid`):**
- [ ] Parameterize `--style` in `prep_flux_dataset.py`
- [ ] Add `allotments`, `cemetery`, `village_green` to `parkland` in `osm_layers.py`
- [ ] Add `brownfield`, `construction` to `industrial` in `osm_layers.py`
- [ ] Add `min_road_fraction` filter to `RoadLayoutDataset`
- [ ] Add `style:` per-city field to YAML format
- [ ] Fix SRTM multi-tile mosaic (before `euro_grid` run)

**Before `africa_informal` run:**
- [ ] Decide Option A vs B for `highway=path`/`track` (visual inspection of 10 Kibera tiles)
- [ ] If Option A: add style-conditional road filter to `road_layers.py`

**Before `us_grid` run (Phase 1):**
- [ ] No pipeline changes needed
- [ ] Create `data_pipeline/cities_us_grid.yaml`
- [ ] Verify alley (`highway=service`) density is reasonable in 5 sample Chicago tiles
