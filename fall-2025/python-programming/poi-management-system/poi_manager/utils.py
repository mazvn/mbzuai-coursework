from math import sqrt, isclose

GRID_MIN = 0
GRID_MAX = 1000
EPSILON = 1e-6

def clamp_coord(x: int) -> int:
    if not isinstance(x, int):
        raise ValueError("Coordinate must be int")
    if x < GRID_MIN or x > GRID_MAX:
        raise ValueError(f"Coordinate out of bounds [{GRID_MIN}, {GRID_MAX}]")
    return x

def dist(p1, p2):
    (x1, y1), (x2, y2) = p1, p2
    return sqrt((x1-x2)**2 + (y1-y2)**2)

def equal_float(a: float, b: float, eps: float = EPSILON) -> bool:
    return isclose(a, b, rel_tol=0.0, abs_tol=eps)
