import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

# ============================================
# CREATE PLOTS FOLDER
# ============================================

os.makedirs("plots", exist_ok=True)

# ============================================
# PHYSICAL PARAMETERS
# ============================================

m1 = 1.0
m2 = 1.0

L1 = 1.0
L2 = 1.0

g = 9.81

# ============================================
# DOUBLE PENDULUM EQUATIONS
# ============================================

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

initial_A = [
    np.radians(30.0),
    0.0,
    np.radians(60.0),
    0.0
]

initial_B = [
    np.radians(30.000001),
    0.0,
    np.radians(60.0),
    0.0
]

# ============================================
# SIMULATION SETTINGS
# ============================================

t_start = 0
t_end = 20
num_points = 10000

t_eval = np.linspace(
    t_start,
    t_end,
    num_points
)

# ============================================
# RUN A
# ============================================

solution_A = solve_ivp(
    double_pendulum,
    [t_start, t_end],
    initial_A,
    t_eval=t_eval,
    rtol=1e-10,
    atol=1e-12
)

# ============================================
# RUN B
# ============================================

solution_B = solve_ivp(
    double_pendulum,
    [t_start, t_end],
    initial_B,
    t_eval=t_eval,
    rtol=1e-10,
    atol=1e-12
)

# ============================================
# EXTRACT DATA
# ============================================

theta1_A = solution_A.y[0]
theta2_A = solution_A.y[2]

theta1_B = solution_B.y[0]
theta2_B = solution_B.y[2]

# ============================================
# DIVERGENCE
# ============================================

difference = np.sqrt(
    (theta1_A - theta1_B) ** 2
    +
    (theta2_A - theta2_B) ** 2
)

# ============================================
# PREDICTABILITY HORIZON
# ============================================

threshold = 0.1

crossings = np.where(
    difference >= threshold
)[0]

if len(crossings) > 0:

    Tp = t_eval[crossings[0]]

    print(
        f"Predictability Horizon = {Tp:.3f} s"
    )

else:

    Tp = None

    print(
        "Threshold not reached."
    )

# ============================================
# LYAPUNOV ESTIMATE
# ============================================

valid = difference > 0

times = t_eval[valid]
log_diff = np.log(difference[valid])

fit_end = 1500

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
# THEORETICAL HORIZON
# ============================================

if Tp is not None:

    delta0 = difference[0]

    Tp_theory = (
        1 / lambda_est
    ) * np.log(
        threshold / delta0
    )

    print(
        f"Theoretical Horizon = {Tp_theory:.3f} s"
    )

# ============================================
# PLOT 1
# ============================================

plt.figure(figsize=(10,5))

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

plt.xlabel("Time (s)")
plt.ylabel("Angle (degrees)")
plt.title(
    "Two Nearly Identical Double Pendulums"
)

plt.legend()
plt.grid(alpha=0.3)

plt.savefig(
    "plots/trajectory.png",
    dpi=300,
    bbox_inches="tight"
)

# ============================================
# PLOT 2
# ============================================

plt.figure(figsize=(10,5))

plt.semilogy(
    t_eval,
    difference,
    color="crimson",
    label="Separation"
)

plt.axhline(
    threshold,
    linestyle="--",
    color="black"
)

if Tp is not None:

    plt.axvline(
        Tp,
        linestyle=":",
        color="blue",
        label=f"Horizon = {Tp:.2f}s"
    )

plt.xlabel("Time (s)")
plt.ylabel("Angular Separation")

plt.title(
    "Predictability Horizon"
)

plt.legend()

plt.grid(alpha=0.3)

plt.savefig(
    "plots/divergence.png",
    dpi=300,
    bbox_inches="tight"
)

# ============================================
# PLOT 3
# ============================================

plt.figure(figsize=(10,5))

plt.plot(
    times[:fit_end],
    log_diff[:fit_end]
)

plt.xlabel("Time (s)")
plt.ylabel("log(Divergence)")

plt.title(
    "Lyapunov Exponent Estimation"
)

plt.grid(alpha=0.3)

plt.savefig(
    "plots/lyapunov.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
