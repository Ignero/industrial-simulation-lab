# Industrial Simulation Lab

Dynamic simulation models for industrial process systems implemented in Python.

This repository contains numerical models focused on classical process engineering
problems solved using modern computational tools. The objective is to bridge
mass and energy balance fundamentals with dynamic system analysis and industrial
digitalization concepts.

---

## Scope

The models included in this repository focus on:

- Dynamic mass balances
- Dynamic energy balances
- Non-isothermal reactors
- Arrhenius reaction kinetics
- Heat transfer (UA-based models)
- Transient response analysis
- Step disturbances and operational limits

---

## Current Models

### 1. Non-Isothermal CSTR with Arrhenius Kinetics
- Coupled mass and energy balance
- Temperature-dependent reaction rate
- Optional safety temperature event
- Transient simulation using SciPy `solve_ivp`

### 2. Thermal Tank with Heat Disturbance
- Lumped energy balance
- Convective and UA heat transfer
- Step heat input disturbance
- Transient temperature response

---

## Technical Stack

- Python 3
- NumPy
- SciPy
- Matplotlib

---

## Repository Structure
```
industrial-simulation-lab/
│
├── README.md
├── requirements.txt
├── scripts/
│ ├── cstr_dynamic_arrhenius.py
│ ├── thermal_tank_step_heat.py
│
└── figures/
```

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
python scripts/cstr_dynamic_arrhenius.py
```

## Maintainer

Jesús Valera Echeverria
Chemical Engineer | Process & Dynamic Modelling
Focus: Industrial Simulation & Dynamic Systems
