from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from .utils import clamp_coord

Coord = Tuple[int, int]

@dataclass(frozen=True)
class POIType:
    name: str
    attributes: List[str] = field(default_factory=list)

@dataclass
class POI:
    id: int
    name: str
    type_name: str
    location: Coord
    attributes: Dict[str, object] = field(default_factory=dict)

@dataclass
class Visit:
    poi_id: int
    date: str  # dd/mm/yyyy
    rating: Optional[int] = None

@dataclass
class Visitor:
    id: int
    name: str
    nationality: str
    visits: List[Visit] = field(default_factory=list)

class Registry:
    # everything is kept in memory. poi ids can never be reused, even after delete
    def __init__(self):
        self.types: Dict[str, POIType] = {}
        self.pois: Dict[int, POI] = {}
        self.visitors: Dict[int, Visitor] = {}
        self._used_poi_ids: Set[int] = set()

    # types / attributes
    def add_type(self, type_name: str, attributes: Optional[List[str]] = None):
        if type_name in self.types:
            raise ValueError(f"Type '{type_name}' already exists")
        self.types[type_name] = POIType(type_name, list(attributes or []))

    def delete_type(self, type_name: str):
        for p in self.pois.values():
            if p.type_name == type_name:
                raise ValueError(f"Cannot delete type '{type_name}': existing POIs reference it")
        if type_name in self.types:
            del self.types[type_name]

    def add_attribute(self, type_name: str, attr: str):
        t = self.types.get(type_name)
        if not t:
            raise ValueError(f"Unknown type '{type_name}'")
        if attr not in t.attributes:
            t.attributes.append(attr)

    def delete_attribute(self, type_name: str, attr: str):
        t = self.types.get(type_name)
        if not t:
            raise ValueError(f"Unknown type '{type_name}'")
        if attr in t.attributes:
            t.attributes.remove(attr)
        for poi in self.pois.values():
            if poi.type_name == type_name:
                poi.attributes.pop(attr, None)

    # renaming
    def rename_attribute(self, type_name: str, old: str, new: str):
        t = self.types.get(type_name)
        if not t:
            raise ValueError(f"Unknown type '{type_name}'")
        if old not in t.attributes:
            raise ValueError(f"Attribute '{old}' not in type '{type_name}'")
        if new in t.attributes:
            raise ValueError(f"New attribute '{new}' already exists")
        idx = t.attributes.index(old)
        t.attributes[idx] = new
        for poi in self.pois.values():
            if poi.type_name == type_name:
                if old in poi.attributes:
                    poi.attributes[new] = poi.attributes.pop(old)
                else:
                    poi.attributes[new] = None

    def rename_type(self, old: str, new: str):
        if old not in self.types:
            raise ValueError(f"Unknown type '{old}'")
        if new in self.types:
            raise ValueError(f"Type '{new}' already exists")
        t = self.types.pop(old)
        self.types[new] = POIType(new, list(t.attributes))
        for poi in self.pois.values():
            if poi.type_name == old:
                poi.type_name = new

    # pois
    def add_poi(self, id: int, name: str, type_name: str, location: Tuple[int, int], attributes: Optional[Dict[str, object]] = None):
        if id in self._used_poi_ids or id in self.pois:
            raise ValueError("POI id already used and cannot be reused")
        if type_name not in self.types:
            raise ValueError(f"Unknown type '{type_name}'")
        x, y = location
        location = (clamp_coord(x), clamp_coord(y))
        attr = attributes or {}
        allowed = set(self.types[type_name].attributes)
        filtered = {k: v for k, v in attr.items() if k in allowed}
        poi = POI(id=id, name=name, type_name=type_name, location=location, attributes=filtered)
        self.pois[id] = poi
        self._used_poi_ids.add(id)

    def delete_poi(self, id: int):
        if id in self.pois:
            del self.pois[id]

    # visitors
    def add_visitor(self, id: int, name: str, nationality: str):
        if id in self.visitors:
            raise ValueError("Visitor id already exists")
        self.visitors[id] = Visitor(id=id, name=name, nationality=nationality)

    def add_visit(self, visitor_id: int, poi_id: int, date: str, rating: Optional[int] = None):
        if visitor_id not in self.visitors:
            raise ValueError("Unknown visitor id")
        if poi_id not in self.pois:
            raise ValueError("Unknown POI id")
        day, month, year = date.split("/")
        assert len(day)==2 and len(month)==2 and len(year)==4, "Date must be dd/mm/yyyy"
        if rating is not None:
            if not isinstance(rating, int) or rating < 1 or rating > 10:
                raise ValueError("Rating must be int 1..10")
        self.visitors[visitor_id].visits.append(Visit(poi_id=poi_id, date=date, rating=rating))
