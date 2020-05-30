# -*- coding: utf-8 -*-
"""crtp.py - Circular Radio Tower Problem.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from lrtp import LRTP
from typing import Union


def d(x: Union[np.ndarray, float], y: Union[np.ndarray, float], modulus=1) -> Union[np.ndarray, float]:
    """Unsigned modular distance between x and y. If both x and y are arrays, they must have the same shape.
    :param x: The first value or array of values.
    :param y: The second value or array of values.
    :param modulus: The modulus of the distance metric. Defaults to 1.
    :return: The unsigned modular distance between x and y.
    """
    return np.minimum(
        np.mod(modulus + y - x, modulus),
        np.mod(modulus + x - y, modulus)
    )


class CRTP:
    def __init__(self, radius: float, houses: np.ndarray):
        """Create a new instance of the CRTP from a radius and sorted list of houses.
        :param radius: radius of each tower.
        :param houses: positions of each house in order.
        """
        self.radius = radius
        self.houses = houses
        self._solution = None

    @property
    def solution(self) -> np.ndarray:
        """Returns an optimal solution to this instance of the CRTP obtained via the O(n^2) algorithm.
        :return: an optimal solution to this instance of the CRTP.
        """
        if self._solution is None:
            self._solve()
        return self._solution

    def _solve(self):
        """Solve this instance of the CRTP by n reductions to the LRTP. Overall runtime O(n^2).
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
            ax.scatter(2 * np.pi * self.houses, np.ones_like(self.houses), marker='o', c='blue', label='House')

        if show_solution:
            for tower in self.solution:
                theta = 2 * np.pi * np.linspace(tower - self.radius, tower + self.radius, 100)
                ax.fill_between(theta, 1, 1.2, alpha=0.5, color='red', label='Tower Coverage')

            ax.scatter(2 * np.pi * self.solution, np.ones_like(self.solution), marker='x', c='red', label='Tower')

        # TODO: Separate the computation of the proposed solution from the plotting of the solution.
        if debug:
            # Find the tower position which has the property that the span of the remaining houses is minimized.
            min_span = None
            min_tower = None
            min_remaining = None
            for tower in np.append(np.mod(self.houses + self.radius, 1), np.mod(self.houses - self.radius, 1)):
                remaining = self.houses[d(self.houses, tower) > self.radius]
                span = np.ptp(np.mod(remaining - tower, 1))     # This only works when all gaps are less than 2r.
                if min_span is None or span < min_span:
                    min_span = span
                    min_tower = tower
                    min_remaining = remaining

            # Plot the span of the remaining towers for the optimal tower choice.
            theta = 2 * np.pi * np.linspace(
                np.min(np.mod(min_remaining - min_tower, 1)) + min_tower,
                np.max(np.mod(min_remaining - min_tower, 1)) + min_tower,
                100
            )
            ax.fill_between(theta, 0.6, 0.8, alpha=0.5, color='green', label='Optimal Remaining Span')

            # Plot the span of the optimal tower itself. This span along with the other should cover all houses.
            theta = 2 * np.pi * np.linspace(min_tower - self.radius, min_tower + self.radius, 100)
            ax.fill_between(theta, 0.6, 0.8, alpha=0.5, color='blue', label='Optimal Tower Coverage')
            ax.scatter([2 * np.pi * min_tower], [1], marker='x', c='blue', label='Optimal Tower')

            # Find the number of towers and placement of those towers needed to cover the remaining ones.
            linear = LRTP(self.radius, np.sort(np.mod(min_remaining - min_tower, 1)))
            proposed = np.sort(np.append(np.mod(linear.solution + min_tower, 1), min_tower))

            # Plot the proposed solution.
            for tower in proposed:
                theta = 2 * np.pi * np.linspace(tower - self.radius, tower + self.radius, 100)
                ax.fill_between(theta, 0.8, 1, alpha=0.5, color='pink', label='Proposed Tower Coverage')

            ax.scatter(2 * np.pi * proposed, np.ones_like(proposed), marker='x', c='pink', label='Proposed Tower')

            # Add legend.
            # TODO: Remove this once we get rid of the return below.
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.04, 1), loc='upper left')

            # Determine if the proposed solution is optimal or not.
            # TODO: Add a check to make sure we are actually covering instead of just checking the number of towers.
            return len(proposed) == len(self.solution)

        # Add legend.
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.04, 1), loc='upper left')

    @staticmethod
    def create(radius, count, check_gaps=True) -> CRTP:
        """Create an instance of the CRTP with given radius and number of houses. Optionally check for gaps of 2r.
        :param radius: the radius of each tower. Should be in (0, 0.5).
        :param count: the number of houses to place.
        :param check_gaps: reject generated instances with gaps of 2r or larger.
        :return: an instance of the CRTP with given radius and number of houses.
        """
        # TODO: Add checks to make sure it is actually possible to generate the number of houses requested.
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

    # Keep running until we find a counterexample (or at least supposed counterexample).
    while True:
        problem = CRTP.create(0.1, 7)
        if not problem.plot():
            print(problem.radius)
            print(problem.houses)
            plt.show()
            break
        else:
            plt.close()


if __name__ == '__main__':
    main()
