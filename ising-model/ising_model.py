import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os

os.makedirs("figures", exist_ok=True)

# ============================================================
# SPIN LATTICE MODELING
# ============================================================

N = 20

# Random initialization of spins (+1 or -1)
M = np.random.choice([-1, 1], size=(N, N))

sns.heatmap(M, cmap=sns.color_palette(["#A1C9F4", "#F4A1C9"]))
plt.title("Spins: +1 in blue, -1 in pink")
plt.show()

# Manual modification of a given spin
M[5, 5] = -1

sns.heatmap(M, cmap=sns.color_palette(["#A1C9F4", "#F4A1C9"]))
plt.title("Modified spin configuration")
plt.savefig("figures/modified_spin_configuration.png")
plt.show()


# ============================================================
# ISING MODEL SIMULATION USING
# THE METROPOLIS-HASTINGS MCMC ALGORITHM
# ============================================================

# This function computes the energy change
# associated with flipping a spin
def delta_E(spins, i, j, J):

    s = spins[i, j]

    neighbors = (
        spins[(i + 1) % N, j]
        + spins[(i - 1) % N, j]
        + spins[i, (j + 1) % N]
        + spins[i, (j - 1) % N]
    )

    dE = 2 * J * s * neighbors

    return dE


def Metropolis(spins, T, J):

    for _ in range(N * N):

        # Random selection of a spin (2D lattice)
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)

        dE = delta_E(spins, i, j, J)

        # Metropolis acceptance criterion
        if dE <= 0 or np.random.rand() < np.exp(-dE / T):
            spins[i, j] *= -1


# Computes the total energy of the system
# while counting each spin pair only once
def E_total(spins, J):

    E = 0

    for i in range(N):
        for j in range(N):

            S = spins[i, j]

            neighbors = (
                spins[(i + 1) % N, j]
                + spins[i, (j + 1) % N]
            )

            E += -J * S * neighbors

    return E


# Computes the total magnetization of the system
def magnetization(spins):
    return np.sum(spins)


# ============================================================
# TEST SIMULATION
# ============================================================

M = np.random.choice([-1, 1], size=(N, N))

sns.heatmap(M, cmap=sns.color_palette(["#A1C9F4", "#F4A1C9"]))
plt.title("Initial spin configuration")
plt.savefig("figures/initial_spin_configuration.png")
plt.show()

# Physical parameters
J = 1      # Coupling constant between neighboring spins
T = 2      # Temperature
steps = 1000

energies = []
magnetizations = []

for _ in range(steps):

    Metropolis(M, T, J)

    E = E_total(M, J)
    Mag = magnetization(M)

    energies.append(E)
    magnetizations.append(Mag)


# Final spin configuration
sns.heatmap(M, cmap=sns.color_palette(["#A1C9F4", "#F4A1C9"]))
plt.title("Final spin configuration")
plt.savefig("figures/final_spin_configuration.png")
plt.show()

# Evolution of energy and magnetization
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(energies, color="orange")
plt.title("Total Energy")
plt.xlabel("Step")
plt.ylabel("Energy")

plt.subplot(1, 2, 2)
plt.plot(magnetizations, color="red")
plt.title("Magnetization")
plt.xlabel("Step")
plt.ylabel("Sum of Spins")

plt.tight_layout()
plt.savefig("figures/energy_magnetization_evolution.png")
plt.show()