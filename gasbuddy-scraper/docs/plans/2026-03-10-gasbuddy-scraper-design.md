# GasBuddy Scraper — Design Doc
**Date:** 2026-03-10

## Goal

Collect daily gas price snapshots from all available gas stations across five metro areas: Bay Area, SoCal, Phoenix Metro, Chicago Metro. Output structured Parquet files for research and time-series analysis.

---

## Architecture

A Python CLI script using **Patchright** (undetected Playwright fork) that runs a stealth Chromium browser. Rather than parsing HTML, the scraper intercepts the GraphQL network responses that GasBuddy's React app makes automatically when a search/map page loads — capturing clean JSON directly.

**Flow:**
```
Run script
  → for each metro anchor point (zip codes covering the area)
      → navigate to gasbuddy.com/gas-prices/{zip}
      → wait for GraphQL response with station list
      → capture JSON via network interception
      → collect station records
  → deduplicate by station ID
  → write Parquet snapshot with date stamp
```

---

## Metro Coverage

Each metro is covered by a curated list of zip codes that tile the area. GasBuddy returns ~20-50 stations per zip search.

| Metro | Approx. Zips | Coverage |
|---|---|---|
| Bay Area | ~30 | SF, Oakland, San Jose, peninsula, East Bay, North Bay |
| SoCal | ~60 | LA metro, Orange County, San Diego, Inland Empire |
| Phoenix Metro | ~25 | Phoenix, Scottsdale, Tempe, Mesa, Chandler, Glendale |
| Chicago Metro | ~30 | Chicago, Evanston, Naperville, Aurora, Joliet |

**Total:** ~145 searches per run. At 4-8s human-like delay per page: ~20-25 minutes per full run.

Stations are deduplicated by `station_id` after all zips are processed.

---

## Data Schema

One row per station per snapshot. Output: `snapshots/YYYY-MM-DD.parquet`.

| Field | Type | Notes |
|---|---|---|
| `snapshot_date` | date | Date the scrape ran |
| `station_id` | str | GasBuddy internal station ID |
| `name` | str | Station display name |
| `brand` | str | Brand/chain (Chevron, Shell, etc.) |
| `address` | str | Street address |
| `city` | str | City |
| `state` | str | State abbreviation |
| `zip` | str | ZIP code |
| `lat` | float | Latitude |
| `lon` | float | Longitude |
| `metro` | str | `bay_area`, `socal`, `phoenix`, `chicago` |
| `regular` | float | Regular unleaded price ($/gal), null if unavailable |
| `midgrade` | float | Midgrade price, null if unavailable |
| `premium` | float | Premium price, null if unavailable |
| `diesel` | float | Diesel price, null if unavailable |
| `price_updated_at` | datetime | Timestamp of last reported price |
| `price_reporter` | str | GasBuddy username who reported the price |

---

## Anti-Detection Approach

- **Patchright** — patches `navigator.webdriver` and other browser fingerprint leaks
- **Human-like delays** — random 4-8s between page navigations (not perfectly regular)
- **Single persistent browser session** — appears as one user browsing, not parallel requests
- **Stealth headless mode** — Patchright's headless avoids standard headless fingerprints
- **Standard Chrome user-agent** — current, realistic UA string
- **Resume on failure** — scraper saves progress after each zip; can resume mid-run if blocked

---

## Output

```
gasbuddy-scraper/
  snapshots/
    2026-03-10.parquet
    2026-03-11.parquet
    ...
  src/
    scraper.py
    zips.py        # zip code lists per metro
    schema.py      # Parquet schema definition
  docs/
    plans/
      2026-03-10-gasbuddy-scraper-design.md
```

Each Parquet file is a standalone daily snapshot. Analysis across dates is done by loading multiple files and concatenating.
