import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ============================================
# DOUBLE PENDULUM — PREDICTABILITY PROJECT
# ============================================

# Physical parameters
m1 = 1.0       # mass of pendulum 1 (kg)
m2 = 1.0       # mass of pendulum 2 (kg)
L1 = 1.0       # length of pendulum 1 (m)
L2 = 1.0       # length of pendulum 2 (m)
g = 9.81       # gravitational acceleration (m/s^2)


def double_pendulum(t, y):
    """
    Equations of motion for a double pendulum.

    y = [theta1, omega1, theta2, omega2]
    """

    theta1, omega1, theta2, omega2 = y

    delta = theta2 - theta1

    denominator1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta)**2

    denominator2 = (L2 / L1) * denominator1

    alpha1 = (
        m2 * L1 * omega1**2 * np.sin(delta) * np.cos(delta)
        + m2 * g * np.sin(theta2) * np.cos(delta)
        + m2 * L2 * omega2**2 * np.sin(delta)
        - (m1 + m2) * g * np.sin(theta1)
    ) / denominator1

    alpha2 = (
        -m2 * L2 * omega2**2 * np.sin(delta) * np.cos(delta)
        + (m1 + m2) * (
            g * np.sin(theta1) * np.cos(delta)
            - L1 * omega1**2 * np.sin(delta)
            - g * np.sin(theta2)
        )
    ) / denominator2

    return [omega1, alpha1, omega2, alpha2]


# ============================================
# INITIAL CONDITIONS
# ============================================

# System A
initial_A = [
    np.radians(30.000000),  # theta1
    0.0,                    # omega1
    np.radians(60.0),       # theta2
    0.0                     # omega2
]

# System B differs by only ONE MILLIONTH OF A DEGREE
initial_B = [
    np.radians(30.000001),
    0.0,
    np.radians(60.0),
    0.0
]


# ============================================
# RUN THE SIMULATIONS
# ============================================

t_start = 0
t_end = 20
num_points = 10000

t_eval = np.linspace(t_start, t_end, num_points)

solution_A = solve_ivp(
    double_pendulum,
    [t_start, t_end],
    initial_A,
    t_eval=t_eval,
    rtol=1e-10,
    atol=1e-12
)

solution_B = solve_ivp(
    double_pendulum,
    [t_start, t_end],
    initial_B,
    t_eval=t_eval,
    rtol=1e-10,
    atol=1e-12
)


# ============================================
# CALCULATE DIVERGENCE
# ============================================

theta1_A = solution_A.y[0]
theta2_A = solution_A.y[2]

theta1_B = solution_B.y[0]
theta2_B = solution_B.y[2]

# Angular separation between the two systems
difference = np.sqrt(
    (theta1_A - theta1_B)**2 +
    (theta2_A - theta2_B)**2
)


# ============================================
# PLOT 1 — TRAJECTORIES
# ============================================

plt.figure(figsize=(10, 5))

plt.plot(
    t_eval,
    np.degrees(theta1_A),
    label="System A",
    linewidth=1
)

plt.plot(
    t_eval,
    np.degrees(theta1_B),
    "--",
    label="System B",
    linewidth=1
)

plt.xlabel("Time (s)")
plt.ylabel("First pendulum angle (degrees)")
plt.title("Two Nearly Identical Double Pendulums")
plt.legend()
plt.grid(alpha=0.3)

plt.show()


# ============================================
# PLOT 2 — DIVERGENCE
# ============================================

plt.figure(figsize=(10, 5))

plt.semilogy(
    t_eval,
    difference,
    color="crimson"
)

plt.xlabel("Time (s)")
plt.ylabel("Angular separation")
plt.title("Growth of Uncertainty Between Two Trajectories")
plt.grid(alpha=0.3)

plt.show()
