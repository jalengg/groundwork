# GasBuddy Scraper Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Playwright-based scraper that collects daily gas price snapshots from all GasBuddy stations across Bay Area, SoCal, Phoenix, and Chicago metros, outputting dated Parquet files for analysis.

**Architecture:** A Patchright (stealth Playwright) browser navigates GasBuddy search pages for ~145 zip codes across 4 metros, intercepts GraphQL network responses, parses station price records, deduplicates by station ID, and writes a dated Parquet snapshot.

**Tech Stack:** Python 3.11+, patchright (stealth Playwright), polars (Parquet output), pytest

---

### Task 1: Project scaffold

**Files:**
- Create: `gasbuddy-scraper/pyproject.toml`
- Create: `gasbuddy-scraper/src/__init__.py`
- Create: `gasbuddy-scraper/src/zips.py`
- Create: `gasbuddy-scraper/src/schema.py`
- Create: `gasbuddy-scraper/src/parser.py`
- Create: `gasbuddy-scraper/src/scraper.py`
- Create: `gasbuddy-scraper/src/writer.py`
- Create: `gasbuddy-scraper/src/cli.py`
- Create: `gasbuddy-scraper/scripts/__init__.py`
- Create: `gasbuddy-scraper/tests/__init__.py`
- Create: `gasbuddy-scraper/tests/fixtures/` (directory)
- Create: `gasbuddy-scraper/.gitignore`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "gasbuddy-scraper"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "patchright",
    "polars",
    "pyarrow",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-asyncio",
]

[project.scripts]
gasbuddy-scrape = "src.cli:main"

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

**Step 2: Install dependencies**

```bash
cd gasbuddy-scraper
pip install -e ".[dev]"
patchright install chromium
```

Expected: no errors. If `patchright` is not available on PyPI, install `playwright` + `playwright-stealth` instead and substitute `from patchright.async_api import async_playwright` with `from playwright.async_api import async_playwright` throughout (stealth patching is less critical but still apply `playwright_stealth(page)` if using that fallback).

**Step 3: Create empty module files**

Touch all files listed above. Leave them empty for now.

**Step 4: Create .gitignore**

```
snapshots/
.progress-*.json
*.pyc
__pycache__/
.pytest_cache/
*.egg-info/
```

**Step 5: Commit**

```bash
git add gasbuddy-scraper/
git commit -m "feat: scaffold gasbuddy-scraper project"
```

---

### Task 2: Zip code lists

**Files:**
- Modify: `gasbuddy-scraper/src/zips.py`
- Create: `gasbuddy-scraper/tests/test_zips.py`

**Step 1: Write failing tests**

```python
# tests/test_zips.py
from src.zips import METRO_ZIPS

def test_all_metros_present():
    assert set(METRO_ZIPS.keys()) == {"bay_area", "socal", "phoenix", "chicago"}

def test_metro_zip_counts():
    assert len(METRO_ZIPS["bay_area"]) >= 25
    assert len(METRO_ZIPS["socal"]) >= 50
    assert len(METRO_ZIPS["phoenix"]) >= 20
    assert len(METRO_ZIPS["chicago"]) >= 25

def test_zips_are_5_digit_strings():
    for metro, zips in METRO_ZIPS.items():
        for z in zips:
            assert isinstance(z, str) and len(z) == 5 and z.isdigit(), \
                f"Bad zip {z!r} in {metro}"

def test_no_duplicate_zips_within_metro():
    for metro, zips in METRO_ZIPS.items():
        assert len(zips) == len(set(zips)), f"Duplicate zips in {metro}"
```

**Step 2: Run to verify they fail**

```bash
cd gasbuddy-scraper
pytest tests/test_zips.py -v
```

Expected: ImportError or AssertionError.

**Step 3: Implement zips.py**

```python
# src/zips.py
METRO_ZIPS: dict[str, list[str]] = {
    "bay_area": [
        # San Francisco
        "94102", "94103", "94107", "94110", "94112", "94114", "94117", "94122", "94124",
        # Oakland / East Bay
        "94601", "94603", "94605", "94609", "94611", "94621",
        # Berkeley / Albany
        "94702", "94703", "94704", "94710",
        # San Jose
        "95112", "95116", "95122", "95128", "95148",
        # Peninsula (Daly City, San Mateo, Redwood City)
        "94015", "94401", "94403", "94063", "94065",
        # North Bay (San Rafael, Santa Rosa)
        "94901", "94903", "95401", "95404",
        # Fremont / Hayward
        "94536", "94538", "94541", "94544",
    ],
    "socal": [
        # Los Angeles core
        "90001", "90011", "90021", "90031", "90044", "90057", "90065",
        # Hollywood / Mid-City
        "90028", "90036", "90046",
        # West LA / Santa Monica
        "90025", "90034", "90064", "90291", "90401",
        # San Fernando Valley
        "91331", "91340", "91342", "91352", "91401", "91405", "91423",
        # Long Beach / South LA
        "90713", "90805", "90808",
        # Orange County
        "92801", "92805", "92614", "92648", "92704",
        # Inland Empire
        "92401", "92503", "91764", "92507",
        # San Diego core
        "92101", "92103", "92111", "92115", "92126",
        # San Diego suburbs
        "91910", "92020", "92025",
        # Pasadena / SGV
        "91101", "91103", "91731", "91745",
        # South Bay
        "90501", "90745", "90250",
        # Ventura County
        "93001", "93003", "93010",
    ],
    "phoenix": [
        # Phoenix core
        "85003", "85006", "85008", "85013", "85015", "85017", "85019", "85031", "85033",
        # Scottsdale
        "85251", "85257", "85260",
        # Tempe
        "85281", "85282", "85283",
        # Mesa
        "85201", "85203", "85205", "85213",
        # Chandler / Gilbert
        "85224", "85226", "85233",
        # Glendale / Peoria
        "85301", "85303", "85345",
        # Surprise / Goodyear
        "85374", "85338",
    ],
    "chicago": [
        # Chicago core
        "60601", "60607", "60608", "60609", "60616", "60619", "60621", "60623", "60625",
        # North Side
        "60613", "60614", "60640", "60657",
        # Northwest Side
        "60630", "60634", "60639",
        # South Side
        "60628", "60636", "60643",
        # Evanston
        "60201",
        # Oak Park
        "60304",
        # Berwyn
        "60402",
        # Naperville / Aurora / Joliet
        "60540", "60506", "60431",
        # O'Hare area
        "60018",
        # South suburbs
        "60426", "60452",
    ],
}
```

**Step 4: Run tests**

```bash
pytest tests/test_zips.py -v
```

Expected: all PASS.

**Step 5: Commit**

```bash
git add src/zips.py tests/test_zips.py
git commit -m "feat: add zip code lists for all 4 metros"
```

---

### Task 3: GraphQL discovery (spike)

**Goal:** Identify which network request GasBuddy makes when you search a zip, and what the response JSON looks like. Output: a fixture file for later tests.

**Files:**
- Create: `gasbuddy-scraper/scripts/discover.py`
- Create: `gasbuddy-scraper/tests/fixtures/sample_response.json` (generated by running the script)

**Step 1: Write the discovery script**

```python
# scripts/discover.py
"""
Run once to capture GasBuddy's GraphQL traffic for a sample zip.
Saves all captured requests/responses to tests/fixtures/sample_response.json.

Usage:
    cd gasbuddy-scraper
    python scripts/discover.py
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from patchright.async_api import async_playwright


async def main():
    captured = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()

        async def handle_response(response):
            if "gasbuddy.com/graphql" in response.url:
                try:
                    body = await response.json()
                    req_body = response.request.post_data
                    captured.append({
                        "url": response.url,
                        "status": response.status,
                        "request_body": req_body,
                        "response_body": body,
                    })
                    print(f"[captured] {response.url} — status {response.status}")
                    # Print top-level keys to help navigate the structure
                    if isinstance(body, dict):
                        print(f"  top-level keys: {list(body.keys())}")
                        data = body.get("data", {})
                        if isinstance(data, dict):
                            print(f"  data keys: {list(data.keys())}")
                except Exception as e:
                    print(f"[skip] {response.url}: {e}")

        page.on("response", handle_response)

        print("Navigating to GasBuddy zip search...")
        await page.goto(
            "https://www.gasbuddy.com/home?search=94103&fuel=1",
            wait_until="networkidle",
            timeout=30000,
        )
        print("Waiting 10s for all GraphQL calls to fire...")
        await page.wait_for_timeout(10000)

        out_path = Path("tests/fixtures/sample_response.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(captured, indent=2))
        print(f"\nSaved {len(captured)} captured responses to {out_path}")
        print("\nNext: inspect the file to find the station array and field names.")

        await browser.close()


asyncio.run(main())
```

**Step 2: Run the discovery script**

```bash
cd gasbuddy-scraper
python scripts/discover.py
```

A real Chromium window will open and navigate to GasBuddy. Watch the terminal output — it prints each captured GraphQL call and its top-level keys. After 10s it saves to `tests/fixtures/sample_response.json`.

**Step 3: Inspect the output**

```bash
python -m json.tool tests/fixtures/sample_response.json | head -150
```

Find the response that contains a list of stations. Look for:
- An array of objects with fields like `id`, `name`, `lat`/`latitude`, `lng`/`longitude`, `prices`
- Inside each station, a `prices` array with fuel grade + price + timestamp + reporter

Note the exact JSON path, e.g.:
```
response_body → data → locationBySearchTerm → stations → results → [array of stations]
```

**Step 4: Update parser.py field paths (in Task 4) based on what you found.**

**Step 5: Commit**

```bash
git add scripts/discover.py tests/fixtures/sample_response.json
git commit -m "feat: add discovery script, capture sample GasBuddy GraphQL response"
```

---

### Task 4: Station data model + parser

> **Depends on Task 3.** Use the actual field names you found in `tests/fixtures/sample_response.json`. The placeholder paths in this task will likely need adjusting.

**Files:**
- Modify: `gasbuddy-scraper/src/schema.py`
- Modify: `gasbuddy-scraper/src/parser.py`
- Create: `gasbuddy-scraper/tests/test_parser.py`

**Step 1: Define the schema**

```python
# src/schema.py
from dataclasses import dataclass
from datetime import date, datetime


@dataclass
class StationRecord:
    snapshot_date: date
    station_id: str
    name: str
    brand: str | None
    address: str | None
    city: str | None
    state: str | None
    zip: str | None
    lat: float | None
    lon: float | None
    metro: str
    regular: float | None
    midgrade: float | None
    premium: float | None
    diesel: float | None
    price_updated_at: datetime | None
    price_reporter: str | None
```

**Step 2: Write failing parser tests**

```python
# tests/test_parser.py
import json
from datetime import date
from pathlib import Path

import pytest

from src.parser import parse_stations_from_response
from src.schema import StationRecord

FIXTURE_PATH = Path("tests/fixtures/sample_response.json")


@pytest.fixture
def sample_response_body():
    """Load the first captured response that contains station data."""
    captured = json.loads(FIXTURE_PATH.read_text())
    # Find the response that has station data (non-empty results)
    for entry in captured:
        records = parse_stations_from_response(
            entry["response_body"], metro="bay_area", snapshot_date=date(2026, 3, 10)
        )
        if records:
            return entry["response_body"]
    pytest.skip("No station data found in fixture — re-run scripts/discover.py")


def test_parse_returns_list_of_station_records(sample_response_body):
    records = parse_stations_from_response(
        sample_response_body, metro="bay_area", snapshot_date=date(2026, 3, 10)
    )
    assert isinstance(records, list)
    assert len(records) > 0
    assert all(isinstance(r, StationRecord) for r in records)


def test_station_has_required_fields(sample_response_body):
    records = parse_stations_from_response(
        sample_response_body, metro="bay_area", snapshot_date=date(2026, 3, 10)
    )
    r = records[0]
    assert r.station_id
    assert r.name
    assert r.metro == "bay_area"
    assert r.snapshot_date == date(2026, 3, 10)


def test_prices_are_float_or_none(sample_response_body):
    records = parse_stations_from_response(
        sample_response_body, metro="bay_area", snapshot_date=date(2026, 3, 10)
    )
    for r in records:
        for field in (r.regular, r.midgrade, r.premium, r.diesel):
            assert field is None or isinstance(field, float), \
                f"Expected float or None, got {type(field)}"


def test_missing_price_is_none_not_zero(sample_response_body):
    records = parse_stations_from_response(
        sample_response_body, metro="bay_area", snapshot_date=date(2026, 3, 10)
    )
    # At least some stations won't report all grades
    has_none = any(
        r.regular is None or r.midgrade is None or r.premium is None or r.diesel is None
        for r in records
    )
    assert has_none, "Expected at least some None prices (not all grades reported at every station)"


def test_empty_response_returns_empty_list():
    records = parse_stations_from_response({}, metro="bay_area", snapshot_date=date(2026, 3, 10))
    assert records == []
```

**Step 3: Run to verify they fail**

```bash
pytest tests/test_parser.py -v
```

Expected: ImportError (parser.py is empty).

**Step 4: Implement parser.py**

> The field paths below are placeholders based on common GasBuddy structures. **You must update them to match what you found in Task 3.** Common variants:
> - `data.locationBySearchTerm.stations.results`
> - `data.nearbyStationsWithPrices.results`
> - `data.stations.results`

```python
# src/parser.py
from datetime import date, datetime
from src.schema import StationRecord


def _safe_price(value) -> float | None:
    """Return float price or None if missing/zero/negative."""
    if value is None:
        return None
    try:
        f = float(value)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def _safe_datetime(value) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _extract_stations(response_body: dict) -> list[dict]:
    """
    Navigate the GraphQL response to find the stations array.
    UPDATE THIS FUNCTION based on the actual structure in your fixture.
    """
    data = response_body.get("data", {})
    if not isinstance(data, dict):
        return []

    # Try known GasBuddy response shapes — update based on Task 3 findings:
    for path in [
        ["locationBySearchTerm", "stations", "results"],
        ["nearbyStationsWithPrices", "results"],
        ["stations", "results"],
    ]:
        node = data
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, list) and node:
            return node

    return []


def parse_stations_from_response(
    response_body: dict,
    metro: str,
    snapshot_date: date,
) -> list[StationRecord]:
    stations = _extract_stations(response_body)
    records = []

    for s in stations:
        prices = s.get("prices") or []
        price_map: dict[str, float | None] = {
            "regular": None, "midgrade": None, "premium": None, "diesel": None
        }
        reporter: str | None = None
        updated_at: datetime | None = None

        for p in prices:
            # Fuel grade name — update key based on fixture (common: "fuel_product", "name", "grade")
            fuel = (p.get("fuel_product") or p.get("name") or "").lower()
            price_val = _safe_price(p.get("price") or p.get("cash_price"))

            if "regular" in fuel or fuel == "unleaded":
                price_map["regular"] = price_val
            elif "midgrade" in fuel or "mid" in fuel:
                price_map["midgrade"] = price_val
            elif "premium" in fuel:
                price_map["premium"] = price_val
            elif "diesel" in fuel:
                price_map["diesel"] = price_val

            # Capture reporter + timestamp from first non-null price
            if price_val is not None and updated_at is None:
                updated_at = _safe_datetime(p.get("time") or p.get("posted_time"))
                posted_by = p.get("posted_by") or p.get("user") or {}
                reporter = posted_by.get("nick") or posted_by.get("username") if isinstance(posted_by, dict) else None

        # Address — update key based on fixture (common: "address", nested object)
        addr = s.get("address") or {}
        address_line = addr.get("line1") or addr.get("street") if isinstance(addr, dict) else s.get("address")

        records.append(StationRecord(
            snapshot_date=snapshot_date,
            station_id=str(s.get("id") or s.get("station_id") or ""),
            name=s.get("name") or s.get("display_name") or "",
            brand=s.get("brand") or s.get("chain_brand"),
            address=address_line,
            city=addr.get("city") if isinstance(addr, dict) else None,
            state=addr.get("state") if isinstance(addr, dict) else None,
            zip=addr.get("zip") or addr.get("postal_code") if isinstance(addr, dict) else None,
            lat=s.get("latitude") or s.get("lat"),
            lon=s.get("longitude") or s.get("lng"),
            metro=metro,
            regular=price_map["regular"],
            midgrade=price_map["midgrade"],
            premium=price_map["premium"],
            diesel=price_map["diesel"],
            price_updated_at=updated_at,
            price_reporter=reporter,
        ))

    return records
```

**Step 5: Run tests**

```bash
pytest tests/test_parser.py -v
```

If tests fail due to wrong field paths, open `tests/fixtures/sample_response.json`, find the station array, and update `_extract_stations` and the field accesses in `parse_stations_from_response` to match. This is expected work — the placeholder paths are best guesses.

**Step 6: Commit**

```bash
git add src/schema.py src/parser.py tests/test_parser.py
git commit -m "feat: add StationRecord schema and GraphQL response parser"
```

---

### Task 5: Progress checkpoint (resume support)

**Files:**
- Modify: `gasbuddy-scraper/src/scraper.py`
- Create: `gasbuddy-scraper/tests/test_checkpoint.py`

**Step 1: Write failing tests**

```python
# tests/test_checkpoint.py
from src.scraper import Checkpoint


def test_checkpoint_saves_and_loads(tmp_path):
    cp = Checkpoint(tmp_path / "progress.json")
    cp.mark_done("bay_area", "94102")
    cp.mark_done("bay_area", "94103")

    cp2 = Checkpoint(tmp_path / "progress.json")  # reload from disk
    assert cp2.is_done("bay_area", "94102")
    assert cp2.is_done("bay_area", "94103")
    assert not cp2.is_done("bay_area", "94104")


def test_checkpoint_remaining_zips(tmp_path):
    cp = Checkpoint(tmp_path / "progress.json")
    cp.mark_done("bay_area", "94102")
    remaining = cp.remaining("bay_area", ["94102", "94103", "94104"])
    assert remaining == ["94103", "94104"]


def test_checkpoint_does_not_error_on_missing_file(tmp_path):
    path = tmp_path / "nonexistent.json"
    assert not path.exists()
    cp = Checkpoint(path)
    assert not cp.is_done("bay_area", "94102")


def test_checkpoint_no_duplicate_entries(tmp_path):
    cp = Checkpoint(tmp_path / "progress.json")
    cp.mark_done("bay_area", "94102")
    cp.mark_done("bay_area", "94102")  # duplicate
    cp2 = Checkpoint(tmp_path / "progress.json")
    assert cp2.remaining("bay_area", ["94102"]) == []
```

**Step 2: Run to verify they fail**

```bash
pytest tests/test_checkpoint.py -v
```

Expected: ImportError.

**Step 3: Implement Checkpoint in scraper.py**

```python
# src/scraper.py
import json
from pathlib import Path


class Checkpoint:
    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, list[str]] = {}
        if path.exists():
            self._data = json.loads(path.read_text())

    def mark_done(self, metro: str, zip_code: str) -> None:
        if metro not in self._data:
            self._data[metro] = []
        if zip_code not in self._data[metro]:
            self._data[metro].append(zip_code)
        self.path.write_text(json.dumps(self._data, indent=2))

    def is_done(self, metro: str, zip_code: str) -> bool:
        return zip_code in self._data.get(metro, [])

    def remaining(self, metro: str, zips: list[str]) -> list[str]:
        return [z for z in zips if not self.is_done(metro, z)]
```

**Step 4: Run tests**

```bash
pytest tests/test_checkpoint.py -v
```

Expected: all PASS.

**Step 5: Commit**

```bash
git add src/scraper.py tests/test_checkpoint.py
git commit -m "feat: add Checkpoint class for scraper resume support"
```

---

### Task 6: Parquet writer

**Files:**
- Modify: `gasbuddy-scraper/src/writer.py`
- Create: `gasbuddy-scraper/tests/test_writer.py`

**Step 1: Write failing tests**

```python
# tests/test_writer.py
from datetime import date, datetime
from pathlib import Path

from src.schema import StationRecord
from src.writer import write_snapshot, read_snapshot


def _record(**kwargs) -> StationRecord:
    defaults = dict(
        snapshot_date=date(2026, 3, 10),
        station_id="123",
        name="Test Shell",
        brand="Shell",
        address="100 Main St",
        city="San Francisco",
        state="CA",
        zip="94102",
        lat=37.77,
        lon=-122.41,
        metro="bay_area",
        regular=4.29,
        midgrade=4.49,
        premium=4.69,
        diesel=None,
        price_updated_at=datetime(2026, 3, 10, 8, 0),
        price_reporter="GasUser42",
    )
    return StationRecord(**{**defaults, **kwargs})


def test_write_and_read_roundtrip(tmp_path):
    records = [_record(station_id="1"), _record(station_id="2")]
    out = tmp_path / "2026-03-10.parquet"
    write_snapshot(records, out)
    df = read_snapshot(out)
    assert len(df) == 2
    assert set(df["station_id"].to_list()) == {"1", "2"}


def test_deduplication_keeps_newest(tmp_path):
    older = _record(station_id="1", price_updated_at=datetime(2026, 3, 10, 6, 0), regular=4.39)
    newer = _record(station_id="1", price_updated_at=datetime(2026, 3, 10, 9, 0), regular=4.19)
    out = tmp_path / "2026-03-10.parquet"
    write_snapshot([older, newer], out)
    df = read_snapshot(out)
    assert len(df) == 1
    assert df["regular"][0] == 4.19


def test_null_prices_preserved(tmp_path):
    out = tmp_path / "2026-03-10.parquet"
    write_snapshot([_record(diesel=None, midgrade=None)], out)
    df = read_snapshot(out)
    assert df["diesel"][0] is None
    assert df["midgrade"][0] is None


def test_empty_input_does_not_write_file(tmp_path):
    out = tmp_path / "empty.parquet"
    write_snapshot([], out)
    assert not out.exists()
```

**Step 2: Run to verify they fail**

```bash
pytest tests/test_writer.py -v
```

Expected: ImportError.

**Step 3: Implement writer.py**

```python
# src/writer.py
from dataclasses import asdict
from pathlib import Path

import polars as pl

from src.schema import StationRecord


def write_snapshot(records: list[StationRecord], path: Path) -> None:
    """Deduplicate by station_id (keep newest price_updated_at) and write Parquet."""
    if not records:
        return

    df = pl.DataFrame([asdict(r) for r in records])

    df = (
        df.sort("price_updated_at", descending=True, nulls_last=True)
        .unique(subset=["station_id"], keep="first")
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def read_snapshot(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path)
```

**Step 4: Run tests**

```bash
pytest tests/test_writer.py -v
```

Expected: all PASS.

**Step 5: Commit**

```bash
git add src/writer.py tests/test_writer.py
git commit -m "feat: add Parquet snapshot writer with deduplication"
```

---

### Task 7: Browser scraper

**Files:**
- Modify: `gasbuddy-scraper/src/scraper.py`

**Step 1: Append the async browser scraper to scraper.py**

This component isn't unit-tested (requires a live browser) — the integration test is the smoke test in Step 2.

```python
# append to src/scraper.py (below the Checkpoint class)
import asyncio
import random
from datetime import date

from patchright.async_api import async_playwright, Page

from src.parser import parse_stations_from_response
from src.schema import StationRecord
from src.zips import METRO_ZIPS


_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


async def _scrape_zip(
    page: Page,
    zip_code: str,
    metro: str,
    snapshot_date: date,
) -> list[StationRecord]:
    """Navigate to a GasBuddy zip search and capture station data via network interception."""
    collected: list[dict] = []

    async def handle_response(response):
        if "gasbuddy.com/graphql" in response.url:
            try:
                body = await response.json()
                collected.append(body)
            except Exception:
                pass

    page.on("response", handle_response)

    await page.goto(
        f"https://www.gasbuddy.com/home?search={zip_code}&fuel=1",
        wait_until="networkidle",
        timeout=30000,
    )
    await page.wait_for_timeout(random.randint(4000, 8000))  # human-like delay

    page.remove_listener("response", handle_response)

    records = []
    for body in collected:
        records.extend(
            parse_stations_from_response(body, metro=metro, snapshot_date=snapshot_date)
        )
    return records


async def run_scraper(
    metros: list[str] | None = None,
    checkpoint_path: Path | None = None,
    snapshot_date: date | None = None,
) -> list[StationRecord]:
    """
    Scrape all zips across the given metros (default: all 4).
    Returns a flat list of StationRecords (deduplication happens in the writer).
    """
    if metros is None:
        metros = list(METRO_ZIPS.keys())
    if snapshot_date is None:
        snapshot_date = date.today()
    if checkpoint_path is None:
        checkpoint_path = Path(f".progress-{snapshot_date}.json")

    checkpoint = Checkpoint(checkpoint_path)
    all_records: list[StationRecord] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent=_USER_AGENT)
        page = await context.new_page()

        for metro in metros:
            zips = checkpoint.remaining(metro, METRO_ZIPS[metro])
            print(f"\n[{metro}] {len(zips)} zips remaining")

            for zip_code in zips:
                print(f"  {zip_code}...", end=" ", flush=True)
                try:
                    records = await _scrape_zip(page, zip_code, metro, snapshot_date)
                    all_records.extend(records)
                    checkpoint.mark_done(metro, zip_code)
                    print(f"{len(records)} stations")
                except Exception as e:
                    print(f"ERROR: {e}")
                    # Don't mark done — will retry on resume

        await browser.close()

    return all_records
```

**Step 2: Smoke test with one zip**

```bash
cd gasbuddy-scraper
python -c "
import asyncio, sys
import src.zips as z
z.METRO_ZIPS = {'bay_area': ['94103']}  # override to 1 zip
from src.scraper import run_scraper
records = asyncio.run(run_scraper(metros=['bay_area']))
print(f'Got {len(records)} records')
if records: print(records[0])
"
```

Expected: prints 10-50 station records. If 0 records, the parser field paths need updating (go back to Task 4 and fix `_extract_stations`).

**Step 3: Commit**

```bash
git add src/scraper.py
git commit -m "feat: add Patchright browser scraper with GraphQL network interception"
```

---

### Task 8: CLI

**Files:**
- Modify: `gasbuddy-scraper/src/cli.py`

**Step 1: Implement CLI**

```python
# src/cli.py
import argparse
import asyncio
from datetime import date
from pathlib import Path

from src.scraper import run_scraper
from src.writer import write_snapshot
from src.zips import METRO_ZIPS


def main():
    parser = argparse.ArgumentParser(
        description="Scrape daily gas prices from GasBuddy"
    )
    parser.add_argument(
        "--metros",
        nargs="+",
        choices=list(METRO_ZIPS.keys()),
        default=list(METRO_ZIPS.keys()),
        help="Which metros to scrape (default: all)",
    )
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        default=date.today(),
        help="Snapshot date (default: today, format: YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("snapshots"),
        help="Directory to write Parquet files (default: snapshots/)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint file if present",
    )
    args = parser.parse_args()

    checkpoint_path = Path(f".progress-{args.date}.json")
    if not args.resume and checkpoint_path.exists():
        checkpoint_path.unlink()

    output_path = args.output_dir / f"{args.date}.parquet"

    print(f"Scraping metros: {args.metros}")
    print(f"Snapshot date:   {args.date}")
    print(f"Output:          {output_path}")

    records = asyncio.run(run_scraper(
        metros=args.metros,
        checkpoint_path=checkpoint_path,
        snapshot_date=args.date,
    ))

    print(f"\nTotal records collected (before dedup): {len(records)}")
    write_snapshot(records, output_path)
    print(f"Written to {output_path}")


if __name__ == "__main__":
    main()
```

**Step 2: Re-install to register the entrypoint**

```bash
pip install -e ".[dev]"
```

**Step 3: Verify the CLI**

```bash
python -m src.cli --help
```

Expected: prints usage without errors.

**Step 4: Full smoke test (one metro)**

```bash
python -m src.cli --metros bay_area
```

Expected: runs ~30 zip searches, writes `snapshots/YYYY-MM-DD.parquet`.

Load the result in Python to verify:
```python
import polars as pl
df = pl.read_parquet("snapshots/2026-03-10.parquet")
print(df.shape)
print(df.head())
```

**Step 5: Commit**

```bash
git add src/cli.py
git commit -m "feat: add CLI with metro, date, output-dir, and resume flags"
```

---

## Running a Full Scrape

```bash
# All 4 metros
python -m src.cli

# Single metro
python -m src.cli --metros chicago

# Resume an interrupted run
python -m src.cli --resume

# Specific date
python -m src.cli --date 2026-03-10

# Multiple specific metros
python -m src.cli --metros bay_area phoenix
```

Results land in `snapshots/YYYY-MM-DD.parquet`. Load for analysis:

```python
import polars as pl
df = pl.read_parquet("snapshots/2026-03-10.parquet")
print(df.shape)           # (n_stations, 17)
print(df["metro"].value_counts())
print(df.filter(pl.col("regular").is_not_null()).sort("regular"))
```
