"""
Thermal Tank with Step Heat Disturbance

Author: Jesús Valera Echeverria
Description:
Lumped parameter energy balance model with convective
flow, heat transfer to ambient and step heat input.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ========================================================
# Parameters
# ========================================================

V = 10.0          # m3
F = 1.0           # m3/s
rho = 1000.0      # kg/m3
Cp = 4180.0       # J/(kg*K)

T_in = 350.0      # K
T_amb = 300.0     # K
UA = 15000.0      # W/K


# Step heat disturbance
def Q(t):
    return 0.0 if t < 20 else 200000.0  # W


# ========================================================
# Model
# ========================================================

def thermal_tank_ode(t, y):
    T = y[0]

    dTdt = (
        (F / V) * (T_in - T)
        + (UA / (rho * Cp * V)) * (T_amb - T)
        + Q(t) / (rho * Cp * V)
    )

    return [dTdt]


# ========================================================
# Simulation
# ========================================================

t_span = (0, 100)
t_eval = np.linspace(*t_span, 300)

y0 = [300.0]

sol = solve_ivp(thermal_tank_ode, t_span, y0, t_eval=t_eval)


# ========================================================
# Plot
# ========================================================

plt.plot(sol.t, sol.y[0])
plt.axvline(20, linestyle="--", label="Heat step")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (K)")
plt.title("Thermal Tank Step Response")
plt.grid()
plt.legend()
plt.show()
