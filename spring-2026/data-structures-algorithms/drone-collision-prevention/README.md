# Drone Collision Prevention

**Task:** Drones are points in 2D or 3D. Find the closest pair fast, then the top-k closest pairs, then keep the top-k updated while drones move. Compare everything against brute force.

**What I did:** 2D uses the classic divide and conquer with the strip, O(n log n). For 3D I split by x the same way but handle the slab with a hash grid on (y, z). Top-k is a sweep over x-sorted points with a max-heap of size k, pruning on the x gap. The moving version keeps a 3D grid and only rechecks pairs around drones that moved. Everything is in one notebook with timings and plots.

It needs `pandas` and `matplotlib`. The first run generates the datasets (up to 10M points), which takes a while, so set `GENERATE_FILES = False` after that.
