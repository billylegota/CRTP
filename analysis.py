# -*- coding: utf-8 -*-
"""analysis.py - Check the counterexample to make sure I'm not crazy.
"""

import matplotlib.pyplot as plt
import numpy as np

from crtp import CRTP, d


def main():
    # Counterexample.
    radius = 0.1
    houses = np.array([
        0.09406199,
        0.14327552,
        0.25960517,
        0.41831979,
        0.6052967,
        0.80002368,
        0.93620904
    ])

    # Determine the remaining span for all candidate towers.
    candidates = np.append(np.mod(houses + radius, 1), np.mod(houses - radius, 1))
    remaining = np.array([houses[d(houses, candidate) > radius] for candidate in candidates])
    span = np.array([np.ptp(np.mod(remaining[i] - candidates[i], 1)) for i in range(len(candidates))])

    tower = candidates[np.argmin(span)]

    for c, r, s in zip(candidates, remaining, span):
        print(c, r, s)

    print(tower)

    problem = CRTP(radius, houses)
    problem.plot(debug=True)
    plt.show()


if __name__ == '__main__':
    main()
