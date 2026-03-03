"""
Closed-loop CSTR temperature control using PI controller.
Saves plots to figures/.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from cstr_pi import cstr_rhs
from pi import PIController


def main():
    os.makedirs("figures", exist_ok=True)

    # Parameters
    params = dict(
        V=10.0,
        F=1.0,
        rho=1000.0,
        Cp=4180.0,
        A=1.0e7,
        Ea=80000.0,
        R=8.314,
        delta_H=-50000.0,
    )

    # Controller
    T_set = 360.0
    pi = PIController(Kp=5e7, Ki=2e6, u_min=-5e6, u_max=9e7)

    # Inputs
    def C_in(t): return 2.0
    def T_in(t): return 350.0

    def Q_dist(t):
        return 0.0 if t < 20 else 200000.0

    # Augmented state: [C, T, I]
    def rhs_aug(t, y):
        C, T, I = y
        error = T_set - T
        Q_ctrl, dIdt = pi.compute(error, I)

        def Q_total(tt):  # keep signature Q(t)
            return Q_ctrl + Q_dist(tt)

        dCdt, dTdt = cstr_rhs(
            t, np.array([C, T]),
            **params, C_in=C_in, T_in=T_in, Q=Q_total
        )
        return [dCdt, dTdt, dIdt]

    y0 = [0.5, 350.0, 0.0]
    t_span = (0, 50)
    t_eval = np.linspace(*t_span, 400)

    sol = solve_ivp(rhs_aug, t_span, y0, t_eval=t_eval, method="BDF")

    t = sol.t
    C = sol.y[0]
    T = sol.y[1]

    plt.figure()
    plt.plot(t, T, label="T(t)")
    plt.axhline(T_set, linestyle="--", label="T_set")
    plt.axvline(20, linestyle="--", label="Disturbance (t=20s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (K)")
    plt.title("CSTR Temperature Control (PI)")
    plt.grid()
    plt.legend()
    plt.savefig("figures/closed_loop_PI_T.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()