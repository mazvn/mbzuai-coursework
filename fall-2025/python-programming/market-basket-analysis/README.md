# Market Basket Analysis

**Task:** Take a grocery transactions dataset, clean it, give products random prices and find which items are bought together the most (pairs and triples), with some plots.

**What I did:** One notebook going through parts A to E. Cleaning removes duplicate items in a basket and drops baskets with fewer than 2 items. Pairs and triples are counted with `itertools.combinations` on sorted baskets into a `Counter`, then filtered by a minimum support count. The plots are in `figures/`.

It needs `groceries.csv` next to the notebook. That's course data, so it's not included.
