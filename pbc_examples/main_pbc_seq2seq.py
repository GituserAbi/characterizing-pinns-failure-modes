"""
Sequence-to-sequence (time-marching) learning for PINNs (paper §5.2,
Krishnapriyan et al. 2021).

Instead of training one PINN to predict the entire space-time domain at once,
the time horizon [0, T] is split into segments of width dt. A separate PINN
is trained per segment; the predicted solution at the end of segment k (i.e.
at t = (k+1)*dt) is used as the initial condition ("data") for segment k+1.
This is illustrated in Fig. 5 / Table 2 of the paper.

To keep the comparison fair (per the paper's footnote 4), the total number of
collocation points across all segments equals --N_f_total, split evenly
per-segment (N_f_total / n_segments collocation points per segment).

Example (reproduces Table 2 row nu=2, rho=5, dt=0.1):
    python main_pbc_seq2seq.py --system rd --nu 2 --rho 5 --dt 0.1 \
        --N_f_total 1000 --layers 50,50,50,50,1
"""

import argparse
import os
import copy
import numpy as np
import torch

from net_pbc import *
from systems_pbc import *
from utils import *
from visualize import *

def str2bool(v):
    """Proper boolean argparse type -- avoids the classic bug where
    `--visualize False` is treated as truthy because it's a non-empty string."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected, got: {v}')

################
# Arguments
################
parser = argparse.ArgumentParser(description='Sequence-to-sequence (time-marching) PINN training')

parser.add_argument('--system', type=str, default='convection', help='System to study.')
parser.add_argument('--seed', type=int, default=0, help='Random initialization.')
parser.add_argument('--N_f_total', type=int, default=1000,
                     help='TOTAL number of collocation points across the whole [0,T] domain '
                          '(split evenly across segments, matching the entire-state-space baseline).')
parser.add_argument('--optimizer_name', type=str, default='LBFGS', help='Optimizer of choice.')
parser.add_argument('--lr', type=float, default=1.0, help='Learning rate.')
parser.add_argument('--L', type=float, default=1.0, help='Multiplier on loss f.')

parser.add_argument('--xgrid', type=int, default=256, help='Number of points in the xgrid.')
parser.add_argument('--nt', type=int, default=100, help='Number of points in the FULL tgrid [0,T] (T=1).')
parser.add_argument('--dt', type=float, default=0.1, help='Width of each time segment (T=1 must be divisible by dt).')

parser.add_argument('--nu', type=float, default=0.0, help='nu (diffusion) coefficient.')
parser.add_argument('--rho', type=float, default=1.0, help='rho (reaction) coefficient.')
parser.add_argument('--beta', type=float, default=1.0, help='beta (convection) coefficient.')
parser.add_argument('--u0_str', default='sin(x)', help='str argument for initial condition (used only for segment 0).')
parser.add_argument('--source', default=0, type=float, help='Constant source/forcing term.')

parser.add_argument('--layers', type=str, default='50,50,50,50,1', help='Dimensions/layers of the NN, minus first layer.')
parser.add_argument('--net', type=str, default='DNN', help='The net architecture.')
parser.add_argument('--activation', default='tanh', help='Activation to use in the network.')
parser.add_argument('--loss_style', default='mean', help='Loss style (mean vs sum).')

parser.add_argument('--reinit_each_segment', action='store_true',
                     help='If set, re-initialize NN weights from scratch for every segment instead of '
                          'warm-starting from the previous segment (default: warm start, matching the '
                          'time-marching scheme described in the paper).')

parser.add_argument('--visualize', type=str2bool, default=False, help='Visualize the stitched full-domain solution (--visualize True/False).')
parser.add_argument('--save_model', type=str2bool, default=False, help='Save each per-segment model (--save_model True/False).')
parser.add_argument('--plot_loss', type=str2bool, default=False, help='Plot training loss from saved history without retraining.')
parser.add_argument('--loss_history_path', default=None, help='Path to a saved loss history pickle file.')

args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

orig_layers = args.layers
layers = [int(item) for item in args.layers.split(',')]

nu, beta, rho = args.nu, args.beta, args.rho
if args.system == 'diffusion':
    beta, rho = 0.0, 0.0
elif args.system == 'convection':
    nu, rho = 0.0, 0.0
elif args.system == 'rd':
    beta = 0.0
elif args.system == 'reaction':
    nu, beta = 0.0, 0.0
else:
    raise ValueError(f"Unknown system: {args.system}")

T_total = 1.0
n_segments = int(round(T_total / args.dt))
assert abs(n_segments * args.dt - T_total) < 1e-8, "T=1 must be (approximately) divisible by --dt"
N_f_per_segment = max(1, args.N_f_total // n_segments)

print(f"System: {args.system} | nu={nu}, beta={beta}, rho={rho}")
print(f"Time-marching: {n_segments} segments of width dt={args.dt}, "
      f"{N_f_per_segment} collocation points per segment ({args.N_f_total} total).")

############################
# Build the FULL fine-grained spatial/temporal grid and ground-truth solution,
# purely for evaluation/visualization and for sampling collocation points
# within each segment (NOT for training data beyond the initial condition --
# only the true initial condition u(x,0) is ever used as labeled data, exactly
# as in the paper: "the only data available here is from the PDE itself").
############################
x = np.linspace(0, 2 * np.pi, args.xgrid, endpoint=False).reshape(-1, 1)
t_full = np.linspace(0, T_total, args.nt).reshape(-1, 1)
X_full, T_full = np.meshgrid(x, t_full)
X_star_full = np.hstack((X_full.flatten()[:, None], T_full.flatten()[:, None]))

if 'convection' in args.system or 'diffusion' in args.system:
    u_vals_full = convection_diffusion(args.u0_str, nu, beta, args.source, args.xgrid, args.nt)
elif 'rd' in args.system:
    u_vals_full = reaction_diffusion_discrete_solution(args.u0_str, nu, rho, args.xgrid, args.nt)
elif 'reaction' in args.system:
    u_vals_full = reaction_solution(args.u0_str, rho, args.xgrid, args.nt)
else:
    raise ValueError("Unknown system")

u_star_full = u_vals_full.reshape(-1, 1)
Exact_full = u_star_full.reshape(len(t_full), len(x))

set_seed(args.seed)

############################
# Time-marching loop
############################
layers_with_input = copy.deepcopy(layers)
layers_with_input.insert(0, 2)  # input is (x, t)

# segment 0's IC is the true initial condition; later segments use the previous
# segment's PINN prediction at its final timestep as the "initial condition".
current_ic_x = x  # (xgrid, 1)
current_ic_u = Exact_full[0:1, :].T  # u(x, 0), shape (xgrid, 1)

u_pred_segments = []  # store predicted u(x,t) per segment, on a local fine grid
t_local_full = np.linspace(0, args.dt, max(2, args.nt // n_segments)).reshape(-1, 1)

prev_dnn_state = None
models = []

path = f"heatmap_results/{args.system}_seq2seq"
if args.visualize and not os.path.exists(path):
    os.makedirs(path)
save_path = "saved_models"
if args.save_model and not os.path.exists(save_path):
    os.makedirs(save_path)

for seg in range(n_segments):
    t0 = seg * args.dt
    t1 = (seg + 1) * args.dt
    print(f"\n=== Segment {seg + 1}/{n_segments}: t in [{t0:.4f}, {t1:.4f}] ===")

    # local time grid for this segment, normalized as [0, dt] internally to keep
    # the network's effective input range consistent across segments, but we
    # store/evaluate using the true global t = t0 + t_local.
    t_local = t_local_full  # (n_local_t, 1), values in [0, dt]

    X_local, T_local = np.meshgrid(x, t_local)
    X_star_local = np.hstack((X_local.flatten()[:, None], T_local.flatten()[:, None]))

    # interior-only collocation points (exclude t=0 plane and x=0 boundary line)
    t_local_noinitial = t_local[1:]
    x_local_noboundary = x[1:]
    X_nb, T_ni = np.meshgrid(x_local_noboundary, t_local_noinitial)
    X_star_noinitial_noboundary_local = np.hstack((X_nb.flatten()[:, None], T_ni.flatten()[:, None]))
    X_f_train = sample_random(X_star_noinitial_noboundary_local, N_f_per_segment)

    G = np.full(X_f_train.shape[0], float(args.source))

    # IC for this segment: (x, 0) -> current_ic_u
    X_u_train = np.hstack((current_ic_x, np.zeros_like(current_ic_x)))
    u_train = current_ic_u

    # periodic BC for this segment
    bc_lb = np.hstack((X_local[:, 0:1], T_local[:, 0:1]))
    x_bc_ub = np.array([2 * np.pi] * t_local.shape[0]).reshape(-1, 1)
    bc_ub = np.hstack((x_bc_ub, t_local))

    model = PhysicsInformedNN_pbc(
        args.system, X_u_train, u_train, X_f_train, bc_lb, bc_ub,
        copy.deepcopy(layers_with_input), G, nu, beta, rho,
        args.optimizer_name, args.lr, args.net, args.L, args.activation, args.loss_style
    )

    if prev_dnn_state is not None and not args.reinit_each_segment:
        # warm start from the previous segment's trained weights
        model.dnn.load_state_dict(prev_dnn_state)
        model.optimizer = choose_optimizer(args.optimizer_name, model.dnn.parameters(), args.lr)

    model.train()
    prev_dnn_state = copy.deepcopy(model.dnn.state_dict())
    models.append(model)

    # predict over the full local grid for this segment
    u_pred_local = model.predict(X_star_local).reshape(len(t_local), len(x))
    u_pred_segments.append(u_pred_local)

    # new IC for next segment = prediction at the END of this segment (t = dt)
    current_ic_u = u_pred_local[-1:, :].T  # (xgrid, 1)
    current_ic_x = x

    if args.save_model:
        torch.save(model, f"{save_path}/seq2seq_{args.system}_seg{seg}_dt{args.dt}"
                           f"_nu{nu}_beta{beta}_rho{rho}_seed{args.seed}.pt")

############################
# Stitch segments together onto the full evaluation grid for error reporting
############################
# Build stitched prediction on the same fine grid used for Exact_full (args.nt points over [0,1])
# by re-predicting each segment's model at the exact subset of t_full that falls in its range.
u_pred_stitched = np.zeros_like(Exact_full)
for seg in range(n_segments):
    t0 = seg * args.dt
    t1 = (seg + 1) * args.dt
    if seg == n_segments - 1:
        mask = (t_full.flatten() >= t0 - 1e-9) & (t_full.flatten() <= t1 + 1e-9)
    else:
        mask = (t_full.flatten() >= t0 - 1e-9) & (t_full.flatten() < t1 - 1e-9)
    t_sub = t_full[mask] - t0  # re-localize to [0, dt] for this segment's model
    if t_sub.shape[0] == 0:
        continue
    X_sub, T_sub = np.meshgrid(x, t_sub)
    X_star_sub = np.hstack((X_sub.flatten()[:, None], T_sub.flatten()[:, None]))
    u_pred_sub = models[seg].predict(X_star_sub).reshape(len(t_sub), len(x))
    u_pred_stitched[mask, :] = u_pred_sub

loss_history = {'segments': [m.history for m in models]}
loss_history_path = f"heatmap_results/{args.system}_seq2seq/loss_history_{args.system}_seq2seq_{args.seed}_{args.N_f_total}_{args.lr}_{args.L}_{args.u0_str}.pkl"
if args.plot_loss and args.loss_history_path is None:
    save_loss_history(loss_history, loss_history_path)
elif args.loss_history_path is not None:
    if args.plot_loss:
        plot_loss(args.loss_history_path, f"heatmap_results/{args.system}_seq2seq", f"loss_plot_{args.system}_seq2seq.pdf", title=f"Seq2seq loss history: {args.system}")
        print(f"Plot only mode: loss plot saved to heatmap_results/{args.system}_seq2seq/loss_plot_{args.system}_seq2seq.pdf")
        exit(0)

if args.plot_loss and args.loss_history_path is None:
    plot_loss(loss_history, f"heatmap_results/{args.system}_seq2seq", f"loss_plot_{args.system}_seq2seq.pdf", title=f"Seq2seq loss history: {args.system}")

u_pred_full = u_pred_stitched.reshape(-1, 1)

error_u_relative = np.linalg.norm(u_star_full - u_pred_full, 2) / np.linalg.norm(u_star_full, 2)
error_u_abs = np.mean(np.abs(u_star_full - u_pred_full))
error_u_linf = np.linalg.norm(u_star_full - u_pred_full, np.inf) / np.linalg.norm(u_star_full, np.inf)

print(f"\n=== Seq2seq (time-marching, dt={args.dt}) final results over full [0,1] domain ===")
print('Error u rel: %e' % error_u_relative)
print('Error u abs: %e' % error_u_abs)
print('Error u linf: %e' % error_u_linf)

if args.visualize:
    print(f"\nWriting visualization PDFs to: {os.path.abspath(path)}")
    os.makedirs(path, exist_ok=True)
    exact_u(Exact_full, x, t_full, nu, beta, rho, orig_layers, args.N_f_total, args.L, args.source, args.u0_str, args.system, path=path)
    u_diff(Exact_full, u_pred_stitched, x, t_full, nu, beta, rho, args.seed, orig_layers, args.N_f_total, args.L, args.source, args.lr, args.u0_str, args.system, path=path)
    u_predict(u_vals_full, u_pred_stitched, x, t_full, nu, beta, rho, args.seed, orig_layers, args.N_f_total, args.L, args.source, args.lr, args.u0_str, args.system, path=path)
    print(f"Done. Files in {path}: {os.listdir(path)}")
