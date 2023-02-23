#!/bin/python

import random
import sys

filename = "data.ds"
if len(sys.argv) >= 2:
    num_pairs = int(sys.argv[1])
else:
    num_pairs = 1000

with open(filename, "w") as f:
    pair_results = []
    for i in range(num_pairs):
        result = 0
        while result <= 0 or result >= 100 or n1 <= 0 or n2 <= 0 or n3 <= 0:
            n1 = round(random.uniform(0, 30), 6)
            n2 = round(random.uniform(0, 30), 6)
            n3 = round(random.uniform(0, 30), 6)
            result = (n1+n2+n3)
        pair_results.append((n1/100, n2/100, n3/100, result/100))

    f.write(" ".join(f"{n1:.6f},{n2:.6f},{n3:.6f}" for n1, n2, n3, _ in pair_results))
    f.write("\n")
    f.write(" ".join(f"{result:.6f}" for _, _, _, result in pair_results))
