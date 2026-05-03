# POI Management System

**Task:** A command line system for points of interest on a 1000x1000 grid. Manage types, POIs, visitors and visits, and answer queries like closest pair, POIs within a radius, k nearest and top visitors.

**What I did:** A `Registry` class keeps everything in memory and enforces the rules (POI ids can't be reused and a type can't be deleted while POIs still use it). The queries are plain functions in `storage.py` and ties are sorted by id so results are always the same. Data can be loaded from a YAML file.

```
pip install -r requirements.txt
python -m poi_manager
python -m unittest discover -s tests
```

In the menu, choose option 1 and load `sample_config.yaml` to get some data.
