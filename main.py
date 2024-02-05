#%%
import numpy as np
import tqdm
from scipy import stats
from scipy.integrate import quad
from scipy.optimize import root_scalar
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from credible_interval import *
from scipy.misc import derivative
import statsmodels.api as sm

#%%
particles_a = stats.norm.rvs(loc=0, scale=0, size=100)
particles_b = stats.norm.rvs(loc=50, scale=5, size=100)
particles = np.append(particles_a, particles_b)

bw = 3
dens = sm.nonparametric.KDEUnivariate(particles)
dens.fit(bw=bw)

x = np.linspace(min(particles), max(particles), 1000)
y = [dens.evaluate(_x)[0] for _x in tqdm.tqdm(x)]

fig = go.Figure(
    go.Scatter(
        x=x, y=y
    )
)
fig.add_trace(
    go.Scatter(
        x=particles, y=np.zeros_like(particles),
        mode="markers"
    )
)
fig.show()

#%%
get_credible_intervals_from_particles(particles, 0.95, bw)

# %%
