# market basket analysis

**task:** take a grocery transactions dataset, clean it, give products random prices, and find which items are bought together the most (pairs and triples), with some plots.

**what i did:** one notebook going A to E. cleaning removes duplicate items in a basket and baskets with less than 2 items. pairs/triples are counted with `itertools.combinations` on sorted baskets into a `Counter`, then filtered by a min support count. plots are in `figures/`.

needs `groceries.csv` next to the notebook (course data, not included).
