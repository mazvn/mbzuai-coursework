import yaml
from typing import Any, Dict
from .models import Registry

def load_config(reg: Registry, path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    for tname, attrs in (data.get("types") or {}).items():
        reg.add_type(tname, attrs or [])
    for poi in data.get("pois") or []:
        reg.add_poi(
            id=int(poi["id"]),
            name=str(poi["name"]),
            type_name=str(poi["type"]),
            location=(int(poi["x"]), int(poi["y"])),
            attributes=poi.get("attributes") or {},
        )
    for v in data.get("visitors") or []:
        reg.add_visitor(int(v["id"]), str(v["name"]), str(v["nationality"]))
        for vis in v.get("visits") or []:
            reg.add_visit(int(v["id"]), int(vis["poi_id"]), str(vis["date"]), vis.get("rating"))
    return data
