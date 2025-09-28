from collections import defaultdict
from .models import Registry
from .utils import dist, equal_float, EPSILON

def list_pois_by_type(reg: Registry, type_name: str):
    res = []
    for p in reg.pois.values():
        if p.type_name == type_name:
            res.append((p.id, p.name, p.location, dict(p.attributes)))
    return sorted(res, key=lambda x: x[0])

def closest_pair(reg: Registry):
    items = list(reg.pois.values())
    best = None
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            a, b = items[i], items[j]
            d = dist(a.location, b.location)
            if best is None or d < best[2] - EPSILON or equal_float(d, best[2]):
                best = ((a.id, a.name, a.location), (b.id, b.name, b.location), d)
    return best

def count_pois_per_type(reg: Registry):
    c = defaultdict(int)
    for p in reg.pois.values():
        c[p.type_name] += 1
    return dict(sorted(c.items()))

def pois_within_radius(reg: Registry, c0, r: float):
    res = []
    for p in reg.pois.values():
        d = dist(c0, p.location)
        if d <= r + EPSILON:
            res.append((p.id, p.name, p.location, p.type_name, d))
    return sorted(res, key=lambda x: (x[4], x[0], x[1]))

def k_nearest(reg: Registry, c0, k: int):
    res = []
    for p in reg.pois.values():
        d = dist(c0, p.location)
        res.append((p.id, p.name, p.location, p.type_name, d))
    res.sort(key=lambda x: (x[4], x[0], x[1]))
    return res[:k]

def boundary_exact(reg: Registry, c0, r: float):
    res = []
    for p in reg.pois.values():
        d = dist(c0, p.location)
        if equal_float(d, r):
            res.append((p.id, p.name, p.location, p.type_name, d))
    return sorted(res, key=lambda x: (x[4], x[0], x[1]))

def visits_for_visitor(reg: Registry, visitor_id: int):
    v = reg.visitors.get(visitor_id)
    if not v: return []
    seen = {}
    for vis in v.visits:
        seen.setdefault(vis.poi_id, vis.date)
    out = []
    for poi_id, date in seen.items():
        p = reg.pois.get(poi_id)
        if p:
            out.append((p.id, p.name, date))
    return sorted(out, key=lambda x: (x[0], x[1]))

def num_visitors_per_poi(reg: Registry):
    visitors_per_poi = defaultdict(set)
    for v in reg.visitors.values():
        for vis in v.visits:
            visitors_per_poi[vis.poi_id].add(v.id)
    res = [(pid, len(s)) for pid, s in visitors_per_poi.items()]
    res.sort(key=lambda x: (-x[1], x[0]))
    return res

def num_pois_per_visitor(reg: Registry):
    res = []
    for v in reg.visitors.values():
        unique = len({vis.poi_id for vis in v.visits})
        res.append((v.id, unique, v.name))
    res.sort(key=lambda x: (-x[1], x[0], x[2]))
    return [(vid, cnt) for vid, cnt, _ in res]

def top_k_visitors_by_poi_count(reg: Registry, k: int):
    arr = []
    for v in reg.visitors.values():
        unique = len({vis.poi_id for vis in v.visits})
        arr.append((unique, v.id, v.name))
    arr.sort(key=lambda x: (-x[0], x[1], x[2]))
    return [(vid, name) for _, vid, name in arr[:k]]

def top_k_pois_by_visitor_count(reg: Registry, k: int):
    visitors_per_poi = defaultdict(set)
    for v in reg.visitors.values():
        for vis in v.visits:
            visitors_per_poi[vis.poi_id].add(v.id)
    arr = []
    for poi_id, s in visitors_per_poi.items():
        p = reg.pois.get(poi_id)
        if p:
            arr.append((len(s), p.id, p.name))
    arr.sort(key=lambda x: (-x[0], x[1], x[2]))
    return [(pid, name) for _, pid, name in arr[:k]]

def coverage_fairness(reg: Registry, m: int, t: int):
    res = []
    for v in reg.visitors.values():
        unique_pois = {vis.poi_id for vis in v.visits}
        types = {reg.pois[pid].type_name for pid in unique_pois if pid in reg.pois}
        if len(unique_pois) >= m and len(types) >= t:
            res.append((v.id, v.name, v.nationality, len(unique_pois), len(types)))
    res.sort(key=lambda x: (-x[3], x[0], x[1]))
    return res
