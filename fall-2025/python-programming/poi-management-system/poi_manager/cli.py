from pprint import pprint
from .models import Registry
from .config_loader import load_config
from .storage import (
    list_pois_by_type, closest_pair, count_pois_per_type, pois_within_radius,
    k_nearest, boundary_exact, visits_for_visitor, num_visitors_per_poi,
    num_pois_per_visitor, top_k_visitors_by_poi_count, top_k_pois_by_visitor_count,
    coverage_fairness
)

MENU = '''
POI Management System
1) Load config (YAML)
2) Add type / attribute / rename
3) Add POI / Delete POI / Delete type
4) Add Visitor / Add Visit
5) POI queries (list by type, closest pair, counts, radius, k-nearest, boundary)
6) Visitor queries (visited-by, counts, top-k, coverage fairness)
0) Exit
> '''

def main(_argv=None):
    reg = Registry()
    while True:
        choice = input(MENU).strip()
        if choice == "0":
            print("Bye."); return 0
        elif choice == "1":
            path = input("YAML path: ").strip()
            try:
                load_config(reg, path); print("Loaded.")
            except Exception as e:
                print("Error:", e)
        elif choice == "2":
            sub = input("a) add type, b) add attr, c) rename attr, d) rename type: ").strip().lower()
            try:
                if sub == "a":
                    t = input("type name: ")
                    attrs = input("attributes (comma sep, optional): ").strip()
                    arr = [a.strip() for a in attrs.split(",") if a.strip()] if attrs else []
                    reg.add_type(t, arr)
                elif sub == "b":
                    t = input("type: "); a = input("attr: "); reg.add_attribute(t, a)
                elif sub == "c":
                    t = input("type: "); old = input("old attr: "); new = input("new attr: ")
                    reg.rename_attribute(t, old, new)
                elif sub == "d":
                    old = input("old type: "); new = input("new type: "); reg.rename_type(old, new)
                print("OK")
            except Exception as e:
                print("Error:", e)
        elif choice == "3":
            sub = input("a) add POI, b) delete POI, c) delete type: ").strip().lower()
            try:
                if sub == "a":
                    i = int(input("id: ")); n = input("name: "); t = input("type: ")
                    x = int(input("x [0..1000]: ")); y = int(input("y [0..1000]: "))
                    reg.add_poi(i, n, t, (x, y))
                elif sub == "b":
                    reg.delete_poi(int(input("id: ")))
                elif sub == "c":
                    reg.delete_type(input("type: "))
                print("OK")
            except Exception as e:
                print("Error:", e)
        elif choice == "4":
            sub = input("a) add visitor, b) add visit: ").strip().lower()
            try:
                if sub == "a":
                    i = int(input("id: ")); n = input("name: "); nat = input("nationality: ")
                    reg.add_visitor(i, n, nat)
                else:
                    vid = int(input("visitor id: ")); pid = int(input("poi id: ")); date = input("dd/mm/yyyy: ")
                    rr = input("rating (1..10 or blank): ").strip()
                    rating = int(rr) if rr else None
                    reg.add_visit(vid, pid, date, rating)
                print("OK")
            except Exception as e:
                print("Error:", e)
        elif choice == "5":
            sub = input("POI queries: a) list by type, b) closest pair, c) counts, d) within radius, e) k-nearest, f) boundary-exact: ").strip().lower()
            try:
                if sub == "a":
                    t = input("type: "); pprint(list_pois_by_type(reg, t))
                elif sub == "b":
                    print(closest_pair(reg))
                elif sub == "c":
                    print(count_pois_per_type(reg))
                elif sub == "d":
                    x=int(input("x: ")); y=int(input("y: ")); r=float(input("radius: "))
                    pprint(pois_within_radius(reg, (x,y), r))
                elif sub == "e":
                    x=int(input("x: ")); y=int(input("y: ")); k=int(input("k: "))
                    pprint(k_nearest(reg, (x,y), k))
                elif sub == "f":
                    x=int(input("x: ")); y=int(input("y: ")); r=float(input("radius: "))
                    pprint(boundary_exact(reg, (x,y), r))
            except Exception as e:
                print("Error:", e)
        elif choice == "6":
            sub = input("Visitor queries: a) visits for visitor, b) #visitors per POI, c) #POIs per visitor, d) top-k visitors, e) top-k POIs, f) coverage fairness: ").strip().lower()
            try:
                if sub == "a":
                    vid = int(input("visitor id: ")); pprint(visits_for_visitor(reg, vid))
                elif sub == "b":
                    pprint(num_visitors_per_poi(reg))
                elif sub == "c":
                    pprint(num_pois_per_visitor(reg))
                elif sub == "d":
                    k = int(input("k: ")); pprint(top_k_visitors_by_poi_count(reg, k))
                elif sub == "e":
                    k = int(input("k: ")); pprint(top_k_pois_by_visitor_count(reg, k))
                elif sub == "f":
                    m = int(input("min POIs m: ")); t = int(input("min types t: ")); pprint(coverage_fairness(reg, m, t))
            except Exception as e:
                print("Error:", e)
        else:
            print("Unknown choice")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
