import numpy as np
from scipy import stats
from scipy.integrate import quad
from scipy.optimize import root_scalar
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import statsmodels.api as sm
import math

def find_roots(
        b, particles, dens
    ):
    roots = []
    x0_min = min(particles)
    x0_max = max(particles)
    x0s = np.linspace(
        x0_min - 0.1*(x0_max-x0_min),
        x0_max + 0.1*(x0_max-x0_min),
        len(particles)
    )

    for x0 in x0s:
        result = root_scalar(
            f=lambda x: dens.evaluate(x)[0] - b,
            x0=x0,
            xtol=1e-10
        )

        if result.converged:
            root = round(result.root, 5)
            roots.append(root)

    roots = list(set(roots))
    roots.sort()
    return roots

def get_integrate_intervals(roots, dens, b):
    integrage_intervals = []
    start = None
    end = None
    for i in range(len(roots)-1):
        if start is None:
            if dens.evaluate((roots[i]+roots[i+1])/2)[0] > b:
                start = roots[i]
        else:
            if dens.evaluate((roots[i]+roots[i+1])/2)[0] < b:
                end = roots[i]
        if i == len(roots) - 2:
            if dens.evaluate((roots[i]+roots[i+1])/2)[0] > b:
                end = roots[i+1]
        if (start is not None) and (end is not None):
            integrage_intervals.append((start, end))
            start = None
            end = None

    return integrage_intervals

def calc_credible_interval_probability(integrate_intervals, dens):
    sum = 0
    for interval in integrate_intervals:
        _sum = quad(
            lambda x: dens.evaluate(x)[0],
            interval[0],
            interval[1]
        )
        sum += _sum[0]
    return sum

def optimized_ci_objective(b, alpha, particles, dens):
    roots = find_roots(b, particles, dens)
    print(roots)
    integrate_intervals = get_integrate_intervals(roots, dens, b)
    print(integrate_intervals)
    interval_probability = calc_credible_interval_probability(
        integrate_intervals,
        dens
    )

    print(f"b = {b}", f", interval probability = {interval_probability}")
    return alpha - interval_probability

def get_optimized_credible_intervals_from_particles(particles, alpha, band_width):
    dens = sm.nonparametric.KDEUnivariate(particles)
    dens.fit(bw=band_width)

    y = [dens.evaluate(_x)[0] for _x in particles]

    result = root_scalar(
        lambda x: optimized_ci_objective(x, alpha, particles, dens),
        bracket=(1e-5, max(y)),
        xtol=1e-4
    )

    roots = find_roots(result.root, particles, dens)
    intervals = get_integrate_intervals(roots, dens, result.root)

    return intervals

def standard_ci_objective(x, target, dens):
    probability = quad(
        lambda _x: dens.evaluate(_x)[0],
        -np.inf,
        x
    )
    return target - probability[0]

def get_standard_credible_intervals_from_particles(particles, alpha, band_width):
    dens = sm.nonparametric.KDEUnivariate(particles)
    dens.fit(bw=band_width)

    x0s = list(particles.copy())
    x0s.sort()
    for i, target in enumerate([(1-alpha)/2, alpha+(1-alpha)/2]):
        if i == 0:
            for x0 in x0s:
                result = root_scalar(
                    lambda x: standard_ci_objective(x, target, dens),
                    x0=x0,
                    xtol=1e-4
                )
                if result.converged:
                    start = result.root
                    break
        else:
            for x0 in reversed(x0s):
                result = root_scalar(
                    lambda x: standard_ci_objective(x, target, dens),
                    x0=x0,
                    xtol=1e-4
                )
                if result.converged:
                    end = result.root
                    break

    return (start, end)


if __name__ == "__main__":
    pass