# POI management system

**task:** command line system for points of interest on a 1000x1000 grid. manage types, POIs, visitors and visits, and answer queries like closest pair, POIs in a radius, k nearest, top visitors etc.

**what i did:** a `Registry` class holds everything in memory and checks the rules (no reusing POI ids, can't delete a type that's still used). queries are plain functions in `storage.py`, ties are sorted by id so results are always the same. data can be loaded from a yaml file.

```
pip install -r requirements.txt
python -m poi_manager      # option 1 -> sample_config.yaml
python -m unittest discover -s tests
```
