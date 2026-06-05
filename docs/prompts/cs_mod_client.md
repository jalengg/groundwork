You're picking up the Groundwork project at https://github.com/jalengg/groundwork
to build the Cities Skylines mod client. Read these files first:

  docs/sdxl_controlnet.md       — the working method (what produces road networks)
  docs/deployment_strategy.md   — the cloud-inference architecture we're building
                                  against; this conversation builds the CLIENT half
  docs/postmortem.md            — context on the long road here (optional)

Your scope: the Cities Skylines 1 mod (Steam Workshop). NOT the server — that's
a separate conversation. You depend on a server API spec but you can stub it
out and assume the contract; we'll wire it in later.

Existing context:
  - CS1 modding uses C# + Unity Mono + Harmony patches (DLL injection at runtime)
  - The mod will live in Steam Workshop, ~50-100 MB typical size
  - Target audience: median CS player on RTX 3060 / GTX 1660 hardware
  - Model lives server-side; mod just sends cond images, polls a job, gets back
    a road graph
  - Mascot character "Chirpy" handles user-facing notifications

Your task — build:

1. **Mod project scaffold.** Set up the Visual Studio / .NET project for a CS1
   mod with:
   - Harmony patches into the appropriate ToolControllers (research which)
   - References to `Assembly-CSharp.dll` and `ColossalManaged.dll`
   - Hot-reload during dev (if practical with CS1's modding setup)
   - Builds to a Workshop-deployable bundle

2. **Paintbrush UI.** Add a tool to CS1's tool-bar that lets the player paint:
   - Landuse zones (residential / commercial / industrial / parkland /
     agricultural) — 5 brushes, similar to existing zoning tool
   - Optional: terrain & water are already in CS so we just read them
   - On submit, rasterize the painted region to a 3-channel RGB matching
     our cond encoding (see `data_pipeline/prep_flux_dataset.py` for the
     exact 7→3 RGB mapping — replicate that in C#)

3. **API client.** Wire up:
   - POST to a server-side /api/generate endpoint with the cond PNG
   - Poll /api/job/{id} every 2s
   - Authenticate via either anonymous (rate-limited) or stored Patreon JWT
   - Return: road graph JSON

4. **Road graph → CS NetManager.** Translate the received road graph (list
   of edges with road class + coordinates) into actual CS road segments:
   - Map our 5 road classes to CS road prefabs (Basic Road, Medium Road,
     Large Road, Highway). Research the prefab IDs.
   - Snap coordinates to CS's grid
   - Call `NetManager.CreateSegment` per edge, handling intersections
   - Group as an atomic action so the player can ctrl+Z

5. **Chirpy notification system.** Build a friendly in-game notification
   widget that:
   - Shows queue position while waiting ("8th in line")
   - Plays a chirp sound when generation completes
   - Shows preview of the generated road network as a 3D overlay before
     committing
   - Buttons: Accept | Reroll | Cancel

6. **Settings panel.** Mod options screen for:
   - Patreon login (store JWT in Settings)
   - Default road class mappings
   - Anonymous tier rate-limit warnings

Stub the server API for development:
```csharp
// You can mock locally with a static JSON response while building the client
```

Honest scope check: this is a real CS1 modding project that will take 1-3 weeks
of focused work. Don't try to one-shot it. Produce a phased plan first
(MILESTONES.md or similar in a new branch), then start with milestone 1
(scaffold + builds + Workshop publishable).

New branch: `jalen/cs-mod-client`.

Useful references:
  - CS1 modding guide: https://skylines.paradoxwikis.com/Modding_API
  - Harmony patching: https://harmony.pardeike.net/
  - Existing mods to study: RoadAnarchy, MoveIt, Network Multitool (Workshop)
