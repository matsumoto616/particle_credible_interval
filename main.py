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
import statsmodels.api as sm

#%%
particles_a = stats.norm.rvs(loc=0, scale=1, size=100)
particles_b = stats.norm.rvs(loc=50, scale=5, size=100)
particles = np.append(particles_a, particles_b)

bw = "normal_reference"
bw = 5
dens = sm.nonparametric.KDEUnivariate(particles)
dens.fit(bw=bw)

x = np.linspace(
    min(particles) - 0.5*(max(particles)-min(particles)),
    max(particles) + 0.5*(max(particles)-min(particles)),
    1000
)
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
optimized_intervals = get_optimized_credible_intervals_from_particles(particles, 0.95, bw)
standard_interval = get_standard_credible_intervals_from_particles(particles, 0.95, bw)

#%%
fig = go.Figure(
    go.Scatter(
        x=x, y=y,
        name="kde"
    )
)
fig.add_trace(
    go.Scatter(
        x=particles, y=np.zeros_like(particles),
        mode="markers",
        name="particles"
    )
)

for interval in optimized_intervals:
    fig.add_vrect(
        x0=interval[0], x1=interval[1],
        fillcolor="LightSalmon", opacity=0.5,
        layer="below", line_width=0,
    )

# fig.add_vrect(
#     x0=standard_interval[0], x1=standard_interval[1],
#     fillcolor="LightSalmon", opacity=0.5,
#     layer="below", line_width=0,   
# )

fig.show()


# %%
