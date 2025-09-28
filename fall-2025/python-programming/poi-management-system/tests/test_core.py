import unittest, os, math
from poi_manager.models import Registry
from poi_manager.storage import closest_pair, boundary_exact, k_nearest, num_pois_per_visitor, coverage_fairness
from poi_manager.config_loader import load_config

class TestCore(unittest.TestCase):
    def setUp(self):
        self.reg = Registry()
        # Types
        self.reg.add_type("museum", ["founded"])
        self.reg.add_type("park", ["area"])
        # POIs
        self.reg.add_poi(1, "A", "museum", (0, 0), {"founded": 2000})
        self.reg.add_poi(2, "B", "museum", (3, 4), {"founded": 1990})  # dist 5 to origin
        self.reg.add_poi(3, "C", "park", (0, 5), {"area": 10})         # dist 5 to origin; sqrt(10) to B
        # Visitors
        self.reg.add_visitor(1, "V1", "AA")
        self.reg.add_visitor(2, "V2", "BB")
        # V1 visits two POIs across two distinct types (A: museum, C: park)
        self.reg.add_visit(1, 1, "01/01/2025", 8)
        self.reg.add_visit(1, 3, "02/01/2025")
        # repeat a visit to check distinct counting
        self.reg.add_visit(1, 3, "03/01/2025")
        # V2 visits B
        self.reg.add_visit(2, 2, "01/01/2025")

    def test_min_pairwise_distance(self):
        pair = closest_pair(self.reg)
        self.assertIsNotNone(pair)
        # Closest pair should be B(3,4) and C(0,5) with distance sqrt(10)
        self.assertAlmostEqual(pair[2], math.sqrt(10), places=6)

    def test_boundary_exact(self):
        out = boundary_exact(self.reg, (0,0), 5.0)
        # B is exactly at radius 5; C also at 5 — we just assert B is present
        self.assertTrue(any(i[0]==2 for i in out))

    def test_k_nearest_ordering(self):
        out = k_nearest(self.reg, (0,0), 2)
        # Nearest two from origin are A (0) and then B or C (both 5); tie broken by id -> B
        self.assertEqual([x[0] for x in out], [1,2])

    def test_type_delete_constraint(self):
        with self.assertRaises(ValueError):
            self.reg.delete_type("museum")

    def test_distinct_counts_rules(self):
        counts = dict(num_pois_per_visitor(self.reg))
        # V1 visited A and C (C twice) -> distinct count = 2
        self.assertEqual(counts[1], 2)

    def test_config_loader_validation(self):
        reg2 = Registry()
        cfg = os.path.join(os.path.dirname(__file__), "..", "sample_config.yaml")
        load_config(reg2, cfg)
        self.assertIn("museum", reg2.types)

    def test_coverage_fairness(self):
        out = coverage_fairness(self.reg, m=2, t=2)
        # V1 visited >=2 distinct POIs across >=2 distinct types -> qualifies
        self.assertTrue(any(v[0]==1 for v in out))

if __name__ == "__main__":
    unittest.main()
