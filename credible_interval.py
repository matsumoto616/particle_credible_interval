import numpy as np
from scipy import stats
from scipy.integrate import quad
from scipy.optimize import root_scalar
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import math

def draw_density_fig(
    particles: np.ndarray     
):
    """確率分布の粒子表現から、カーネル密度推定を行って密度関数を描写する。

    Args:
        particles (np.ndarray): 粒子（確率分布からのiidサンプリング結果）
    """
    return None

def kde(x, sample, band_width, kernel):
    n = len(sample)
    return np.sum([1/(n*band_width)*kernel((x-sample[i])/band_width) for i in range(n)])

def kde_derivative(x, sample, band_width, kernel_derivative):
    n = len(sample)
    return np.sum([1/(n*band_width**2)*kernel_derivative((x-sample[i])/band_width) for i in range(n)])

def kde_derivative2(x, sample, band_width, kernel_derivative2):
    n = len(sample)
    return np.sum([1/(n*band_width**3)*kernel_derivative2((x-sample[i])/band_width) for i in range(n)])

def normal_kernel(x):
    return 1/np.sqrt(2*np.pi)*np.exp(-x**2/2)

def normal_kernel_derivative(x):
    return -x/np.sqrt(2*np.pi)*np.exp(-x**2/2)

def normal_kernel_derivative2(x):
    return -1/np.sqrt(2*np.pi)*np.exp(-x**2/2)+x**2/np.sqrt(2*np.pi)*np.exp(-x**2/2)

def find_roots(
        b, particles, band_width, kernel, kernel_derivative, kernel_derivative2
    ):
    roots = []
    x0_min = min(particles) - band_width
    x0_max = max(particles) + band_width
    x0s = np.linspace(x0_min, x0_max, len(particles))

    for x0 in x0s:
        result = root_scalar(
            f=lambda x: kde(x, particles, band_width, kernel) - b,
            fprime=lambda x: kde_derivative(x, particles, band_width, kernel_derivative),
            fprime2=lambda x: kde_derivative2(x, particles, band_width, kernel_derivative2),
            x0=x0,
            xtol=1e-10
        )

        if result.converged:
            root = round(result.root, 5)
            roots.append(root)

    roots = list(set(roots))
    roots.sort()
    return roots

def get_integrate_intervals(roots, slopes_at_root):
    integrage_intervals = []
    for i in range(len(roots)):
        if i == 0:
            start = None
            end = None
            if slopes_at_root[i] < 0:
                print(roots)
                print(slopes_at_root)
                print(roots[i], slopes_at_root[i])
                raise Exception("maybe root findings has failed")
            else:
                start = roots[i]
        else:
            if start is not None:
                if slopes_at_root[i] > 0:
                    print(roots)
                    print(slopes_at_root)
                    print(roots[i], slopes_at_root[i])
                    raise Exception("maybe root findings has failed")
                else:
                    end = roots[i]
            else:
                if slopes_at_root[i] < 0:
                    print(roots)
                    print(slopes_at_root)
                    print(roots[i], slopes_at_root[i])
                    raise Exception("maybe root findings has failed")
                else:
                    start = roots[i]
        if (start is not None) and (end is not None):
            integrage_intervals.append((start, end))
            start = None
            end = None

    return integrage_intervals

def calc_credible_interval_probability(integrate_intervals, particles, band_width, kernel):
    sum = 0
    for interval in integrate_intervals:
        _sum = quad(
            lambda x: kde(x, particles, band_width, kernel),
            interval[0],
            interval[1]
        )
        sum += _sum[0]
    return sum

def objective(b, alpha, particles, band_width):
    roots = find_roots(
        b, particles, band_width, normal_kernel, normal_kernel_derivative, normal_kernel_derivative2
    )

    slopes_at_root = [
        kde_derivative(root, particles, band_width, normal_kernel_derivative)
        for root in roots
    ]

    integrate_intervals = get_integrate_intervals(roots, slopes_at_root)

    interval_probability = calc_credible_interval_probability(
        integrate_intervals,
        particles,
        band_width,
        normal_kernel
    )

    print(f"b = {b}", f", interval probability = {interval_probability}")
    return alpha - interval_probability

def get_credible_intervals_from_particles(particles, alpha, band_width):
    y = [kde(_x, particles, band_width, normal_kernel) for _x in particles]

    result = root_scalar(
        lambda x: objective(x, alpha, particles, band_width),
        bracket=(min(y), max(y)),
        xtol=1e-4
    )

    roots = find_roots(
        result.root, particles, band_width, normal_kernel, normal_kernel_derivative, normal_kernel_derivative2
    )

    slopes_at_root = [
        kde_derivative(root, particles, band_width, normal_kernel_derivative)
        for root in roots
    ]

    intervals = get_integrate_intervals(roots, slopes_at_root)

    return intervals

if __name__ == "__main__":
    particles = stats.norm.rvs(loc=50, scale=20, size=1000)
    band_width = np.sqrt(
        np.var(particles, ddof=1)*(len(particles)**(-1/5))**2
    )

    x = np.linspace(min(particles), max(particles), 1000)
    y = [kde(_x, particles, band_width, normal_kernel) for _x in x]

    fig = go.Figure(
        go.Scatter(
            x=x, y=y
        )
    )
    fig.write_html("test.html")