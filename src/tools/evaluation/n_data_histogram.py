import math
import numpy as np


phase = 30

with open('trained/weight_' + str(phase) + '.txt', 'r') as f:
    weights = [int(elem) for elem in f.read().splitlines()]

weights_without_zero = [w for w in weights if w != 0]

ADJ_IGNORE_N_APPEAR = 3

total = len(weights_without_zero)
rare_values = [w for w in weights_without_zero if w <= ADJ_IGNORE_N_APPEAR]
rare_counts = len(rare_values)
percent = (rare_counts / total * 100) if total else 0.0
print(f"Elements with occurrences <= {ADJ_IGNORE_N_APPEAR}: {rare_counts}/{total} ({percent:.2f}%)")


BAR_WIDTH = 100
INCLUDE_PERCENT = 80

import matplotlib.pyplot as plt

if not weights_without_zero:
    print("No non-zero weights to histogram.")
else:
    arr = np.array(weights_without_zero)

    # exclude the top (100 - INCLUDE_PERCENT)% -> keep the lower INCLUDE_PERCENT% of data
    low_pct = float(np.percentile(arr, 0))
    high_pct = float(np.percentile(arr, INCLUDE_PERCENT))

    # ensure first bin starts at 1 instead of 0
    start = max(1, math.floor(low_pct / BAR_WIDTH) * BAR_WIDTH)
    end = math.ceil((high_pct + 1) / BAR_WIDTH) * BAR_WIDTH
    bins = list(range(int(start), int(end) + BAR_WIDTH, BAR_WIDTH))

    # keep only values inside the lower INCLUDE_PERCENT% range for histogram calculation/plot
    filtered = arr[(arr >= start) & (arr <= end)]

    # counts per bin
    counts, bin_edges = np.histogram(filtered, bins=bins)

    # print text histogram (bars capped to 60 chars, scaled)
    # max_count = counts.max() if counts.size else 0
    # scale = max_count / 60 if max_count > 60 else 1
    # for i, c in enumerate(counts):
    #     low = int(bin_edges[i])
    #     high = int(bin_edges[i + 1]) - 1
    #     bar_len = int(c / scale) if scale else c
    #     bar = "█" * bar_len
    #     print(f"{low:>8} - {high:>8} : {c:>6} {bar}")

    # matplotlib plot
    plt.figure(figsize=(10, 6))
    plt.hist(filtered, bins=bins, edgecolor='black')
    plt.title(f"Weights histogram (lower {INCLUDE_PERCENT}% range: {int(start)} to {int(end)}, phase={phase})")
    plt.xlabel("Number of occurrences of a specific feature in the training data")
    plt.ylabel("Count")
    plt.xlim(start, end)
    plt.grid(axis="y", alpha=0.7)
    plt.tight_layout()

    # save and show
    # plt.savefig('trained/weights_histogram.png', dpi=150)
    plt.show()
