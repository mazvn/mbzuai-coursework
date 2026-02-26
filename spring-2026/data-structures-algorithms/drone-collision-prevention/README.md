# drone collision prevention

**task:** drones are points in 2D/3D. find the closest pair fast (2D and 3D), then the top-k closest pairs, then keep the top-k updated while drones move. compare everything against brute force.

**what i did:** 2D is the classic divide & conquer with the strip (O(n log n)). for 3D i split by x the same way but handle the slab with a hash grid on (y, z). top-k is a sweep over x-sorted points with a max-heap of size k and pruning on the x gap. the moving part keeps a 3D grid and only rechecks pairs around drones that moved. all in one notebook with timings and plots.

needs `pandas` and `matplotlib`. the first run generates the datasets (up to 10M points, takes a while), set `GENERATE_FILES = False` after.
