# -*- coding: utf-8 -*-
"""crtp.py - Circular Radio Tower Problem.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from itertools import chain
from lrtp import LRTP
from typing import Union


def d(x: Union[np.ndarray, float], y: Union[np.ndarray, float], modulus=1) -> Union[np.ndarray, float]:
    """Unsigned modular distance between x and y. If both x and y are arrays, they must have the same shape.
    :param x: The first value or array of values.
    :param y: The second value or array of values.
    :param modulus: The modulus of the distance metric. Defaults to 1.
    :return: The unsigned modular distance between x and y.
    """
    # See: https://stackoverflow.com/questions/6192825/c-calculating-the-distance-between-2-floats-modulo-12
    return np.minimum(
        np.mod(modulus + y - x, modulus),
        np.mod(modulus + x - y, modulus)
    )


class CRTP:
    def __init__(self, radius: float, houses: np.ndarray):
        """Create a new instance of the CRTP from a radius and sorted list of houses.
        :param radius: The radius of each tower (inclusive).
        :param houses: The positions of each house in order.
        """
        self.radius = radius
        self.houses = houses
        self._solution = None

    @property
    def solution(self):
        if self._solution is None:
            self._solve()
        return self._solution

    def _solve(self):
        """Solve this instance of the CRTP by O(n) reductions to the LRTP.
        """
        self._solution = None
        for offset in self.houses:
            linear = LRTP(self.radius, np.sort(np.mod(self.houses - offset, 1)))
            if self._solution is None or len(linear.solution) < len(self._solution):
                self._solution = np.mod(linear.solution + offset, 1)

    def plot(self, ax: plt.PolarAxes = None, show_houses: bool = True, show_solution: bool = True, debug: bool = True):
        if ax is None:
            fig = plt.figure()                      # type:plt.Figure
            ax = fig.add_subplot(111, polar=True)   # type:plt.PolarAxes

        ax.set_yticklabels([])

        if show_houses:
            ax.scatter(2 * np.pi * self.houses, np.ones_like(self.houses), marker='o', c='blue')

        if show_solution:
            for tower in self.solution:
                theta = 2 * np.pi * np.linspace(tower - self.radius, tower + self.radius, 100)
                ax.fill_between(theta, 1, 1.2, alpha=0.5, color='red')

            ax.scatter(2 * np.pi * self.solution, np.ones_like(self.solution), marker='x', c='red')

        if debug:
            # Find the tower location such that the remaining span is minimized.
            min_span = None
            min_candidate = None
            min_remaining = None
            for candidate in chain(np.mod(self.houses + self.radius, 1), np.mod(self.houses - self.radius, 1)):
                remaining = self.houses[d(self.houses, candidate) > self.radius]
                span = np.ptp(np.mod(remaining - candidate, 1))     # FIXME: This calculation is probably wrong.
                if min_span is None or span < min_span:
                    min_span = span
                    min_candidate = candidate
                    min_remaining = remaining

            theta = 2 * np.pi * np.linspace(
                np.min(np.mod(min_remaining - min_candidate, 1)) + min_candidate,
                np.max(np.mod(min_remaining - min_candidate, 1)) + min_candidate,
                100
            )
            ax.fill_between(theta, 0.6, 0.8, alpha=0.5, color='green')

            theta = 2 * np.pi * np.linspace(min_candidate - self.radius, min_candidate + self.radius, 100)
            ax.fill_between(theta, 0.6, 0.8, alpha=0.5, color='blue')

            # Find the number of towers and placement of those towers needed to cover the remaining ones.
            linear = LRTP(self.radius, min_remaining)
            proposed = np.sort(np.append(linear.solution, min_candidate))

            for tower in proposed:
                theta = 2 * np.pi * np.linspace(tower - self.radius, tower + self.radius, 100)
                ax.fill_between(theta, 0.8, 1, alpha=0.5, color='pink')

            ax.scatter([2 * np.pi * min_candidate], [1], marker='*', c='black', s=400)
            ax.scatter(2 * np.pi * proposed, np.ones_like(proposed), marker='X', c='pink')

            # # Perform a cut in either direction at the house with the minimum span.
            # left = LRTP(self.radius, np.sort(np.mod(self.houses - min_candidate, 1)))
            # right = LRTP(self.radius, np.sort(np.mod(min_candidate - self.houses, 1)))
            #
            # # FIXME: This is currently wrong. We are not cutting at the right place. The algorithm proposed says that we
            # #        should place a tower at the given candidate. This *should* just mean that the final answer is just
            # #        the number of towers needed to cover the remaining houses.
            #
            # if len(left.solution) <= len(right.solution):
            #     proposed = np.sort(np.mod(left.solution + min_candidate, 1))
            # else:
            #     proposed = np.sort(np.mod(min_candidate - right.solution, 1))
            #
            # for tower in proposed:
            #     theta = 2 * np.pi * np.linspace(tower - self.radius, tower + self.radius, 100)
            #     ax.fill_between(theta, 0.8, 1, alpha=0.5, color='pink')
            #
            # ax.scatter([2 * np.pi * min_candidate], [1], marker='*', c='black')
            # ax.scatter(2 * np.pi * proposed, np.ones_like(proposed), marker='X', c='pink')
            #
            # print(len(proposed), len(self.solution), len(proposed) == len(self.solution))

    @staticmethod
    def create(radius, count, check_gaps=True) -> CRTP:
        while True:
            houses = np.sort(np.random.random((count,)))

            if check_gaps:
                if np.any(np.diff(houses) >= 2 * radius):
                    continue
                if d(houses[-1], houses[0]) >= 2 * radius:
                    continue

            return CRTP(radius, houses)


def main():
    # fig = plt.figure()
    # for i in range(8):
    #     ax = fig.add_subplot(2, 4, i + 1, polar=True)
    #     problem = CRTP.create(0.1, 30)
    #     problem.plot(ax)
    #
    # plt.show()

    problem = CRTP.create(0.1, 30)
    problem.plot()
    plt.show()


if __name__ == '__main__':
    main()
