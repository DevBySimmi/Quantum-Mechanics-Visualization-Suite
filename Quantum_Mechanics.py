import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from matplotlib.animation import FuncAnimation

plt.style.use('dark_background')

plt.rcParams['figure.facecolor'] = '#2d1b4e'
plt.rcParams['axes.facecolor'] = '#4a2c7a'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['text.color'] = 'white'
plt.rcParams['legend.facecolor'] = '#2d1b4e'
plt.rcParams['legend.edgecolor'] = 'white'

# ==================================================
# QUANTUM MECHANICS VISUALIZATION SUITE
# ==================================================

plt.style.use('ggplot')

# ==================================================
# FIGURE 1 : QUANTUM TUNNELING
# ==================================================

x = np.linspace(-5, 5, 1000)

V0 = 2.0
barrier_width = 1.0
E = 1.0

V = np.where(np.abs(x) <= barrier_width, V0, 0)

k = np.sqrt(E)
kappa = np.sqrt(V0 - E)

T = np.exp(-2 * kappa * 2 * barrier_width)

psi = np.sin(k * x)

inside = np.exp(-kappa * np.abs(x))

psi_tunnel = np.where(np.abs(x) <= barrier_width,
                      inside,
                      psi)

prob = psi_tunnel ** 2

fig1, ax = plt.subplots(1, 2, figsize=(14, 6))

ax[0].plot(x, V, color='black', linewidth=3,
           label='Potential V(x)')

ax[0].axhline(E,
              color='red',
              linestyle='--',
              label='Energy E = 1')

ax[0].fill_between(x, 0, V,
                   color='gray',
                   alpha=0.3)

ax[0].set_title(
    f'Potential Barrier (Transmission = {T:.3f})'
)

ax[0].set_xlabel('Position (x)')
ax[0].set_ylabel('Energy')
ax[0].legend()

ax[1].plot(x,
           psi_tunnel / 2,
           color='blue',
           linewidth=2,
           label='Re[ψ(x)]')

ax[1].plot(x,
           prob / 4,
           color='red',
           linewidth=2,
           label='|ψ(x)|²')

ax[1].plot(x,
           V / 5,
           '--',
           color='gray',
           label='V(x)/5')

ax[1].set_title('Tunneling Wavefunction')
ax[1].set_xlabel('Position (x)')
ax[1].set_ylabel('Amplitude')
ax[1].legend()

fig1.suptitle('Quantum Tunneling', fontsize=22)

# ==================================================
# FIGURE 2 : WAVE PACKET EVOLUTION (ANIMATED)
# ==================================================

fig2, ax2 = plt.subplots(figsize=(12, 6))

x_wave = np.linspace(-10, 10, 1200)

prob_line, = ax2.plot([], [],
                      color='red',
                      linewidth=3,
                      label='|ψ(x,t)|²')

real_line, = ax2.plot([], [],
                      '--',
                      color='blue',
                      label='Re[ψ(x,t)]')

imag_line, = ax2.plot([], [],
                      '--',
                      color='green',
                      label='Im[ψ(x,t)]')

title = ax2.set_title("Wave Packet Evolution")

ax2.set_xlim(-10, 10)
ax2.set_ylim(-0.8, 0.8)

ax2.set_xlabel("Position (x)")
ax2.set_ylabel("Amplitude")
ax2.legend()

def update(frame):

    t = frame / 20

    x0 = -3 + 2 * t
    sigma = 1.0
    k0 = 3

    envelope = np.exp(
        -(x_wave - x0) ** 2 /
        (2 * sigma ** 2)
    )

    real = envelope * np.cos(k0 * x_wave - 2 * t)
    imag = envelope * np.sin(k0 * x_wave - 2 * t)

    prob = envelope ** 2

    prob_line.set_data(x_wave, prob * 0.4)
    real_line.set_data(x_wave, real * 0.7)
    imag_line.set_data(x_wave, imag * 0.7)

    title.set_text(
        f"Wave Packet Evolution   |   Time = {t:.2f}"
    )

    return prob_line, real_line, imag_line

ani = FuncAnimation(
    fig2,
    update,
    frames=200,
    interval=50,
    blit=True
)

# ==================================================
# FIGURE 3 : QUANTUM HARMONIC OSCILLATOR
# ==================================================

x = np.linspace(-4, 4, 1200)

fig3, axes = plt.subplots(2, 2, figsize=(14, 8))

for n, ax in enumerate(axes.flat):

    Hn = hermite(n)

    psi = Hn(x) * np.exp(-x**2 / 2)

    psi /= np.max(np.abs(psi))

    probability = psi**2

    energy = n + 0.5

    turning = np.sqrt(2 * energy)

    ax.plot(x,
            psi * 0.75,
            color='blue',
            linewidth=2,
            label=f'ψ_{n}(x)')

    ax.plot(x,
            probability * 0.55,
            '--',
            color='red',
            linewidth=2,
            label=f'|ψ_{n}|²')

    ax.axhline(
        energy,
        color='green',
        linestyle=':',
        label=f'E_{n} = {energy}'
    )

    ax.axvline(
        turning,
        color='orange',
        linestyle='--',
        alpha=0.6
    )

    ax.axvline(
        -turning,
        color='orange',
        linestyle='--',
        alpha=0.6
    )

    ax.set_title(
        f'n = {n}, E = {energy}ħω'
    )

    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Amplitude')
    ax.legend()

fig3.suptitle(
    'Quantum Harmonic Oscillator',
    fontsize=22
)

# ==================================================
# SHOW ALL FIGURES
# ==================================================

plt.show()