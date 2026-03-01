# data pipeline tests

import os
import yaml

def test_cities_yaml_loads():
    path = os.path.join(os.path.dirname(__file__), "../data_pipeline/cities.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    assert cfg["tile_size_px"] == 512
    assert cfg["pixel_size_m"] == 5
    assert len(cfg["cities"]) == 8
    val_cities = [c for c in cfg["cities"] if c["split"] == "val"]
    assert len(val_cities) == 1
    assert val_cities[0]["name"] == "irving_tx"
    assert all("query" in c and c["query"] for c in cfg["cities"])
    assert all(c["split"] in {"train", "val"} for c in cfg["cities"])
