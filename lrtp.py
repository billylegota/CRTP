# -*- coding: utf-8 -*-
"""lrtp.py - Linear Radio Tower Problem.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


class LRTP:
    def __init__(self, radius: float, houses: np.ndarray):
        """Create a new instance of the LRTP from a radius and sorted list of houses.
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
        """Solve this instance of the LRTP using the O(n) greedy algorithm.
        """
        solution = []
        extent = None

        for house in self.houses:
            if extent is None or extent < house:
                solution.append(house + self.radius)
                extent = house + 2 * self.radius

        self._solution = np.array(solution)

    def plot(self, show_solution=True):
        fig = plt.figure()          # type:plt.Figure
        ax = fig.add_subplot(111)   # type:plt.Axes
        ax.get_yaxis().set_visible(False)

        ax.scatter(self.houses, np.zeros_like(self.houses), marker='o', c='blue')

        if show_solution:
            for tower in self.solution:
                x = np.linspace(tower - self.radius, tower + self.radius, 100)
                ax.fill_between(x, -1, 1, alpha=0.5, color='red')

            ax.scatter(self.solution, np.zeros_like(self.solution), marker='x', c='red')

    @staticmethod
    def create(radius, count) -> LRTP:
        houses = np.sort(np.random.random((count,)))
        return LRTP(radius, houses)


if __name__ == '__main__':
    problem = LRTP.create(0.1, 30)
    problem.plot()
    plt.show()
