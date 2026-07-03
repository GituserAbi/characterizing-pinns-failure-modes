"""
Visualize outputs.
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

def exact_u(Exact, x, t, nu, beta, rho, layers, N_f, L, source, u0_str, system, path):
    """Visualize exact solution."""
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)

    h = ax.imshow(Exact.T, interpolation='nearest', cmap='rainbow',
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    ax.set_xlabel('t', fontweight='bold', size=30)
    ax.set_ylabel('x', fontweight='bold', size=30)

    ax.tick_params(labelsize=15)

    plt.savefig(
        f"{path}/exactu_{system}_nu{nu}_beta{beta}_rho{rho}_Nf{N_f}_{layers}_L{L}_source{source}_{u0_str}.pdf",
        bbox_inches='tight',
        dpi=300
    )
    plt.close()

    return None

def u_diff(Exact, U_pred, x, t, nu, beta, rho, seed, layers, N_f, L, source, lr, u0_str, system, path, relative_error = False):
    """Visualize abs(u_pred - u_exact)."""

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)

    if relative_error:
        h = ax.imshow(np.abs(Exact.T - U_pred.T)/np.abs(Exact.T), interpolation='nearest', cmap='binary',
                    extent=[t.min(), t.max(), x.min(), x.max()],
                    origin='lower', aspect='auto')
    else:
        h = ax.imshow(np.abs(Exact.T - U_pred.T), interpolation='nearest', cmap='binary',
                    extent=[t.min(), t.max(), x.min(), x.max()],
                    origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    ax.set_xlabel('t', fontweight='bold', size=30)
    ax.set_ylabel('x', fontweight='bold', size=30)

    ax.tick_params(labelsize=15)

    plt.savefig(
        f"{path}/udiff_{system}_nu{nu}_beta{beta}_rho{rho}_Nf{N_f}_{layers}_L{L}_seed{seed}_source{source}_{u0_str}_lr{lr}.pdf",
        bbox_inches='tight',
        dpi=300
    )
    plt.close()

    return None

def u_predict(u_vals, U_pred, x, t, nu, beta, rho, seed, layers, N_f, L, source, lr, u0_str, system, path):
    """Visualize u_predicted."""

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)

    # colorbar for prediction: set min/max to ground truth solution.
    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto', vmin=u_vals.min(0), vmax=u_vals.max(0))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    ax.set_xlabel('t', fontweight='bold', size=30)
    ax.set_ylabel('x', fontweight='bold', size=30)

    ax.tick_params(labelsize=15)

    plt.savefig(
        f"{path}/upredicted_{system}_nu{nu}_beta{beta}_rho{rho}_Nf{N_f}_{layers}_L{L}_seed{seed}_source{source}_{u0_str}_lr{lr}.pdf",
        bbox_inches='tight',
        dpi=300
    )

    plt.close()
    return None


def save_loss_history(loss_history, filepath):
    """Save loss history dict to a pickle file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(loss_history, f)


def load_loss_history(filepath):
    """Load loss history dict from a pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def plot_loss(loss_history, path, filename, title='Loss history'):
    """Plot loss history saved during training."""
    if isinstance(loss_history, str):
        loss_history = load_loss_history(loss_history)

    fig, ax = plt.subplots(figsize=(9, 5))

    if 'segments' in loss_history:
        for idx, segment in enumerate(loss_history['segments']):
            ax.plot(segment['loss'], label=f'Segment {idx+1} total loss')
            ax.plot(segment['loss_u'], '--', label=f'Segment {idx+1} loss_u')
            ax.plot(segment['loss_b'], ':', label=f'Segment {idx+1} loss_b')
            ax.plot(segment['loss_f'], '-.', label=f'Segment {idx+1} loss_f')
    else:
        ax.plot(loss_history['loss'], label='Total loss')
        ax.plot(loss_history['loss_u'], label='Loss u')
        ax.plot(loss_history['loss_b'], label='Loss b')
        ax.plot(loss_history['loss_f'], label='Loss f')

    ax.set_yscale('log')
    ax.set_xlabel('Iteration', fontweight='bold', size=15)
    ax.set_ylabel('Loss', fontweight='bold', size=15)
    ax.set_title(title, fontweight='bold', size=18)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, which='both', ls='--', alpha=0.3)

    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, filename), bbox_inches='tight', dpi=300)
    plt.close()
    return None
