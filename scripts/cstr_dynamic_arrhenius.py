"""
CSTR Dynamic Model with Arrhenius Kinetics

Author: Jesús Valera Echeverria
Description:
Dynamic mass and energy balance for a non-isothermal CSTR
with temperature-dependent reaction rate.

Demonstrates the coupling between reaction kinetics and
thermal dynamics under transient conditions.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ========================================================
# Parameters
# ========================================================

V = 10.0            # m3
F = 1.0             # m3/s

C_in = 2.0          # kg/m3
T_in = 350.0        # K

A = 1.0e7           # 1/s
Ea = 80000.0        # J/mol
R = 8.314           # J/(mol*K)

delta_H = -50000.0  # J/mol (exothermic)
rhoCp = 4.0e6       # J/(m3*K)

C0 = 0.5            # kg/m3
T0 = 350.0          # K


# ========================================================
# Model
# ========================================================

def cstr_ode(t, y):
    C, T = y

    # Arrhenius rate constant
    k = A * np.exp(-Ea / (R * T))

    # Mass balance
    dCdt = (F / V) * (C_in - C) - k * C

    # Energy balance
    dTdt = (F / V) * (T_in - T) + (-delta_H / rhoCp) * k * C

    return [dCdt, dTdt]


# ========================================================
# Simulation
# ========================================================

t_span = (0, 50)
t_eval = np.linspace(*t_span, 200)

y0 = [C0, T0]

sol = solve_ivp(cstr_ode, t_span, y0, t_eval=t_eval, method="BDF")


# ========================================================
# Plot results
# ========================================================

plt.figure()
plt.plot(sol.t, sol.y[0])
plt.xlabel("Time (s)")
plt.ylabel("Concentration (kg/m3)")
plt.title("CSTR Concentration Response")
plt.grid()
plt.show()

plt.figure()
plt.plot(sol.t, sol.y[1])
plt.xlabel("Time (s)")
plt.ylabel("Temperature (K)")
plt.title("CSTR Temperature Response")
plt.grid()
plt.show()
