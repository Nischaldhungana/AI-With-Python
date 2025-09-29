import numpy as np
import matplotlib.pyplot as plt

n_values = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]

for n in n_values:
    d1 = np.random.randint(1, 7, size=n)
    d2 = np.random.randint(1, 7, size=n)
    s = d1 + d2
    h, edges = np.histogram(s, range(2, 14))

    plt.figure(figsize=(7,4))
    plt.bar(edges[:-1], h / n, width=0.9, align='center')
    plt.xticks(edges[:-1])
    plt.xlabel("Sum of two dice")
    plt.ylabel("Relative frequency")
    plt.title(f"Sum of two dice (n = {n})")
    plt.ylim(0, 0.2)
    plt.tight_layout()
    plt.show()
