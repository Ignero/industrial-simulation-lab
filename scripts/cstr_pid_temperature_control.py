"""
CSTR with PI Temperature Control

Author: Jesús Valera Echeverria

Non-isothermal CSTR with temperature control using
a PI controller acting on heat input.
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

C_in = 2.0        # mol/m3
T_in = 350.0      # K

A = 1.0e7         #1/s
Ea = 80000.0       # J/mol
R = 8.314          # J/(mol*K)
delta_H = -50000.0 # J/mol

Kp = 5e7
Ki = 2e6
T_set = 360.0       # K

# Step heat disturbance
def Q_dist(t):
    return 0.0 if t < 20 else 1_000_000.0  # W


# ========================================================
# Model
# ========================================================


def cstr_pi(t, y):
    C, T, I = y

    #Reaction
    k = A * np.exp(-Ea / (R * T))
    r = k * C

    #Mass balance
    dCdt = (F/V)*(C_in - C) - r

    #PI control
    error = T_set - T
    Q_unsat = Kp * error + Ki * I
    Q_control = np.clip(Q_unsat, -5e6, 9e7)

    if not np.isclose(Q_control, Q_unsat):
        dIdt = 0.0
    else:
        dIdt = error

    #Energy balance
    dTdt = (
        (F / V) * (T_in - T)
        + ((Q_control +Q_dist(t)) / (rho * Cp * V))
        +(-delta_H / (rho * Cp)) * r
    )

    return [dCdt, dTdt, dIdt]

y0 = [C_in, T_in, 0.0]

# ========================================================
# Simulation
# ========================================================

t_span = (0, 50)
t_eval = np.linspace(*t_span, 400)

y0 = [0.5, 350.0, 0.0] # [C0, T0, I0]

sol = solve_ivp(cstr_pi, t_span, y0, t_eval=t_eval, method ="BDF")



# ========================================================
# Plot
# ========================================================
plt.figure()
plt.plot(sol.t, sol.y[1], label="T(t)")
plt.axhline(T_set, linestyle="--", label="T_set")
plt.axvline(20, linestyle="--", label="Disturbance (t=20s)")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (K)")
plt.title("CSTR Temperature Control (PI)")
plt.grid()
plt.legend()
plt.show()
