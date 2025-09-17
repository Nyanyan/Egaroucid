import math
import numpy as np
from collections import Counter
from tqdm import trange


ADJ_IGNORE_N_APPEAR = 3
percent_data = []

phases = []
for phase in range(12, 60):

    with open('trained/weight_' + str(phase) + '.txt', 'r') as f:
        weights = [int(elem) for elem in f.read().splitlines()]

    weights_without_zero = [w for w in weights if w != 0]
    total = len(weights_without_zero)
    rare_values = [w for w in weights_without_zero if w <= ADJ_IGNORE_N_APPEAR]
    rare_counts = len(rare_values)
    percent = (rare_counts / total * 100) if total else 0.0

    print('phase', phase, 'total', total, 'rare_counts', rare_counts, 'percent', percent)
    percent_data.append(percent)
    phases.append(phase)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(phases, percent_data, marker='o', linestyle='-')
plt.xlabel('Phase')
plt.ylabel('Percentage of rare features (%)')
plt.title(f'Rare feature percentage per phase (occurred <= {ADJ_IGNORE_N_APPEAR})')
plt.grid(True)
plt.tight_layout()
plt.savefig('trained/n_data_phase.png', dpi=300)
plt.close()
# plt.show()
