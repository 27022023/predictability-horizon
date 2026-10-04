import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

# ============================================
# CREATE PLOTS FOLDER
# ============================================

os.makedirs("plots", exist_ok=True)

# ============================================
# DOUBLE PENDULUM PARAMETERS
# ============================================

m1 = 1.0
m2 = 1.0
L1 = 1.0
L2 = 1.0
g = 9.81


def double_pendulum(t, y):

    theta1, omega1, theta2, omega2 = y

    delta = theta2 - theta1

    denominator1 = (
        (m1 + m2) * L1
        - m2 * L1 * np.cos(delta) ** 2
    )

    denominator2 = (L2 / L1) * denominator1

    alpha1 = (
        m2 * L1 * omega1**2 * np.sin(delta) * np.cos(delta)
        + m2 * g * np.sin(theta2) * np.cos(delta)
        + m2 * L2 * omega2**2 * np.sin(delta)
        - (m1 + m2) * g * np.sin(theta1)
    ) / denominator1

    alpha2 = (
        -m2 * L2 * omega2**2 * np.sin(delta) * np.cos(delta)
        + (m1 + m2)
        * (
            g * np.sin(theta1) * np.cos(delta)
            - L1 * omega1**2 * np.sin(delta)
            - g * np.sin(theta2)
        )
    ) / denominator2

    return [omega1, alpha1, omega2, alpha2]


# ============================================
# INITIAL CONDITIONS
# ============================================

initial_A = [
    np.radians(30.000000),
    0.0,
    np.radians(60.0),
    0.0
]

initial_B = [
    np.radians(30.0001),
    0.0,
    np.radians(60.0),
    0.0
]

# ============================================
# SIMULATION SETTINGS
# ============================================

t_start = 0
t_end = 100
num_points = 10000

t_eval = np.linspace(
    t_start,
    t_end,
    num_points
)

# ============================================
# RUN SIMULATIONS
# ============================================

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
# DIVERGENCE
# ============================================

theta1_A = solution_A.y[0]
theta2_A = solution_A.y[2]

theta1_B = solution_B.y[0]
theta2_B = solution_B.y[2]

difference = np.sqrt(
    (theta1_A - theta1_B) ** 2
    + (theta2_A - theta2_B) ** 2
)

print("Maximum divergence:", np.max(difference))

# ============================================
# PREDICTABILITY HORIZON
# ============================================

threshold = 0.01

crossings = np.where(
    difference >= threshold
)[0]

if len(crossings) > 0:

    Tp = t_eval[crossings[0]]

    print(
        f"Predictability Horizon = {Tp:.3f} seconds"
    )

else:

    Tp = None

    print("Threshold not reached.")

# ============================================
# LYAPUNOV ESTIMATE
# ============================================

valid = difference > 0

log_diff = np.log(difference[valid])
times = t_eval[valid]

fit_end = min(1500, len(times))

coeffs = np.polyfit(
    times[:fit_end],
    log_diff[:fit_end],
    1
)

lambda_est = coeffs[0]

print(
    f"Estimated Lyapunov Exponent = {lambda_est:.6f}"
)

# ============================================
# TRAJECTORY PLOT
# ============================================

plt.figure(figsize=(10, 5))

plt.plot(
    t_eval,
    np.degrees(theta1_A),
    label="System A"
)

plt.plot(
    t_eval,
    np.degrees(theta1_B),
    "--",
    label="System B"
)

plt.title("Trajectory Comparison")
plt.xlabel("Time (s)")
plt.ylabel("Angle (degrees)")
plt.grid(alpha=0.3)
plt.legend()

plt.savefig("plots/trajectory.png")
plt.close()

# ============================================
# DIVERGENCE PLOT
# ============================================

plt.figure(figsize=(10, 5))

plt.semilogy(
    t_eval,
    difference,
    color="crimson"
)

plt.axhline(
    threshold,
    color="black",
    linestyle="--"
)

if Tp is not None:
    plt.axvline(
        Tp,
        color="blue",
        linestyle=":"
    )

plt.title("Growth of Uncertainty")
plt.xlabel("Time (s)")
plt.ylabel("Angular Separation")
plt.grid(alpha=0.3)

plt.savefig("plots/divergence.png")
plt.close()

print("Plots saved in /plots")
