"""
Curriculum regularization for PINNs (paper §5.1, Krishnapriyan et al. 2021).

Instead of training directly at the target (hard) PDE coefficient, this script
trains a sequence of PINNs starting from an easy coefficient and progressively
increasing toward the target value, re-using the weights from the previous
(easier) stage as the initialization for the next (harder) stage. This is the
"warm start" curriculum schedule shown in Fig. 4(a) of the paper.

Supports the same systems as main_pbc.py: 'convection', 'rd' (reaction-diffusion),
'reaction'. The coefficient that gets curriculum-scheduled is chosen automatically
based on the system (beta for convection, rho for reaction, nu for rd -- you can
override with --curriculum_param).

Example (reproduces Table 1, beta=30 case):
    python main_pbc_curriculum.py --system convection --beta 30 \
        --curriculum_start 1 --curriculum_steps 6 --N_f 100 --layers 50,50,50,50,1
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
parser = argparse.ArgumentParser(description='Curriculum regularization for PINNs')

parser.add_argument('--system', type=str, default='convection', help='System to study.')
parser.add_argument('--seed', type=int, default=0, help='Random initialization.')
parser.add_argument('--N_f', type=int, default=100, help='Number of collocation points to sample.')
parser.add_argument('--optimizer_name', type=str, default='LBFGS', help='Optimizer of choice.')
parser.add_argument('--lr', type=float, default=1.0, help='Learning rate.')
parser.add_argument('--L', type=float, default=1.0, help='Multiplier on loss f.')

parser.add_argument('--xgrid', type=int, default=256, help='Number of points in the xgrid.')
parser.add_argument('--nt', type=int, default=100, help='Number of points in the tgrid.')
parser.add_argument('--nu', type=float, default=1.0, help='nu (diffusion) coefficient -- target value.')
parser.add_argument('--rho', type=float, default=1.0, help='rho (reaction) coefficient -- target value.')
parser.add_argument('--beta', type=float, default=30.0, help='beta (convection) coefficient -- target value.')
parser.add_argument('--u0_str', default='sin(x)', help='str argument for initial condition.')
parser.add_argument('--source', default=0, type=float, help='Constant source/forcing term.')

parser.add_argument('--layers', type=str, default='50,50,50,50,1', help='Dimensions/layers of the NN, minus first layer.')
parser.add_argument('--net', type=str, default='DNN', help='The net architecture.')
parser.add_argument('--activation', default='tanh', help='Activation to use in the network.')
parser.add_argument('--loss_style', default='mean', help='Loss style (mean vs sum).')

# Curriculum-specific args
parser.add_argument('--curriculum_param', type=str, default=None,
                     help="Which coefficient to schedule: 'beta', 'rho', or 'nu'. "
                          "If not given, inferred from --system.")
parser.add_argument('--curriculum_start', type=float, default=1.0,
                     help='Starting (easy) value of the curriculum coefficient.')
parser.add_argument('--curriculum_steps', type=int, default=5,
                     help='Number of curriculum stages (linearly spaced from start to target, inclusive).')
parser.add_argument('--curriculum_schedule', type=str, default='linear', choices=['linear', 'log'],
                     help='Spacing of the curriculum stages between start and target value.')
parser.add_argument('--reset_optimizer_lr', type=float, default=None,
                     help='If set, use this LR for every curriculum stage instead of --lr (useful since '
                          'L-BFGS sometimes benefits from a smaller LR on later/harder stages).')

parser.add_argument('--visualize', type=str2bool, default=False, help='Visualize the final-stage solution (--visualize True/False).')
parser.add_argument('--save_model', type=str2bool, default=False, help='Save the final-stage model (--save_model True/False).')
parser.add_argument('--save_all_stages', type=str2bool, default=False, help='Save every curriculum-stage model (--save_all_stages True/False).')

args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# parse layers
orig_layers = args.layers
layers = [int(item) for item in args.layers.split(',')]

############################
# Figure out which coefficient is curriculum-scheduled, and the fixed others
############################
nu, beta, rho = args.nu, args.beta, args.rho

if args.system == 'diffusion':
    beta, rho = 0.0, 0.0
    default_param = 'nu'
elif args.system == 'convection':
    nu, rho = 0.0, 0.0
    default_param = 'beta'
elif args.system == 'rd':
    beta = 0.0
    default_param = 'rho'  # could also curriculum nu; user can override
elif args.system == 'reaction':
    nu, beta = 0.0, 0.0
    default_param = 'rho'
else:
    raise ValueError(f"Unknown system: {args.system}")

curriculum_param = args.curriculum_param if args.curriculum_param is not None else default_param
assert curriculum_param in ('beta', 'rho', 'nu'), "curriculum_param must be one of beta/rho/nu"

target_value = {'beta': beta, 'rho': rho, 'nu': nu}[curriculum_param]

if args.curriculum_schedule == 'linear':
    schedule = np.linspace(args.curriculum_start, target_value, args.curriculum_steps)
else:  # log-spaced, useful when start/target span orders of magnitude
    schedule = np.geomspace(max(args.curriculum_start, 1e-8), max(target_value, 1e-8), args.curriculum_steps)
    schedule[-1] = target_value  # ensure we hit the exact target

print(f"System: {args.system} | Curriculum parameter: {curriculum_param}")
print(f"Curriculum schedule ({args.curriculum_schedule}): {schedule}")

############################
# Process data (grid / domain are fixed across all curriculum stages)
############################
x = np.linspace(0, 2 * np.pi, args.xgrid, endpoint=False).reshape(-1, 1)
t = np.linspace(0, 1, args.nt).reshape(-1, 1)
X, T = np.meshgrid(x, t)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

t_noinitial = t[1:]
x_noboundary = x[1:]
X_noboundary, T_noinitial = np.meshgrid(x_noboundary, t_noinitial)
X_star_noinitial_noboundary = np.hstack((X_noboundary.flatten()[:, None], T_noinitial.flatten()[:, None]))

set_seed(args.seed)  # fix collocation sampling + initial weights across the whole curriculum run
X_f_train = sample_random(X_star_noinitial_noboundary, args.N_f)

xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
bc_lb = np.hstack((X[:, 0:1], T[:, 0:1]))
t_full = np.linspace(0, 1, args.nt).reshape(-1, 1)
x_bc_ub = np.array([2 * np.pi] * t_full.shape[0]).reshape(-1, 1)
bc_ub = np.hstack((x_bc_ub, t_full))

layers_with_input = copy.deepcopy(layers)
layers_with_input.insert(0, xx1.shape[-1])

############################
# Helper to compute the exact solution / G for a given coefficient triple
############################
def get_exact_solution(nu_, beta_, rho_):
    if 'convection' in args.system or 'diffusion' in args.system:
        u_vals = convection_diffusion(args.u0_str, nu_, beta_, args.source, args.xgrid, args.nt)
        G = np.full(X_f_train.shape[0], float(args.source))
    elif 'rd' in args.system:
        u_vals = reaction_diffusion_discrete_solution(args.u0_str, nu_, rho_, args.xgrid, args.nt)
        G = np.full(X_f_train.shape[0], float(args.source))
    elif 'reaction' in args.system:
        u_vals = reaction_solution(args.u0_str, rho_, args.xgrid, args.nt)
        G = np.full(X_f_train.shape[0], float(args.source))
    else:
        raise ValueError("Unknown system")
    return u_vals, G

############################
# Curriculum training loop
############################
prev_dnn_state = None
model = None
path = f"heatmap_results/{args.system}_curriculum"
if args.visualize and not os.path.exists(path):
    os.makedirs(path)
save_path = "saved_models"
if (args.save_model or args.save_all_stages) and not os.path.exists(save_path):
    os.makedirs(save_path)

for stage_idx, stage_value in enumerate(schedule):
    stage_nu, stage_beta, stage_rho = nu, beta, rho
    if curriculum_param == 'beta':
        stage_beta = float(stage_value)
    elif curriculum_param == 'rho':
        stage_rho = float(stage_value)
    elif curriculum_param == 'nu':
        stage_nu = float(stage_value)

    u_vals, G = get_exact_solution(stage_nu, stage_beta, stage_rho)
    u_star = u_vals.reshape(-1, 1)
    Exact = u_star.reshape(len(t_full), len(x))
    uu1 = Exact[0:1, :].T

    u_train = uu1
    X_u_train = xx1

    lr_this_stage = args.reset_optimizer_lr if args.reset_optimizer_lr is not None else args.lr

    print(f"\n=== Curriculum stage {stage_idx + 1}/{len(schedule)} | "
          f"{curriculum_param} = {stage_value:.4f} (target = {target_value}) ===")

    model = PhysicsInformedNN_pbc(
        args.system, X_u_train, u_train, X_f_train, bc_lb, bc_ub,
        copy.deepcopy(layers_with_input), G, stage_nu, stage_beta, stage_rho,
        args.optimizer_name, lr_this_stage, args.net, args.L, args.activation, args.loss_style
    )

    # Warm-start: load the previous stage's trained weights into the new DNN.
    if prev_dnn_state is not None:
        model.dnn.load_state_dict(prev_dnn_state)
        # re-create optimizer now that we've loaded warm-start weights
        model.optimizer = choose_optimizer(args.optimizer_name, model.dnn.parameters(), lr_this_stage)

    model.train()
    prev_dnn_state = copy.deepcopy(model.dnn.state_dict())

    u_pred = model.predict(X_star)
    rel_err = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    abs_err = np.mean(np.abs(u_star - u_pred))
    print(f"Stage {stage_idx + 1} -- relative error: {rel_err:.5e} | absolute error: {abs_err:.5e}")

    if args.save_all_stages:
        fname = (f"{save_path}/curriculum_{args.system}_{curriculum_param}{stage_value:.4f}"
                  f"_seed{args.seed}_Nf{args.N_f}.pt")
        torch.save(model, fname)

############################
# Final report (at target coefficient) + optional visualization/saving
############################
u_pred = model.predict(X_star)
u_vals, _ = get_exact_solution(nu, beta, rho)
u_star = u_vals.reshape(-1, 1)
Exact = u_star.reshape(len(t_full), len(x))

error_u_relative = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
error_u_abs = np.mean(np.abs(u_star - u_pred))
error_u_linf = np.linalg.norm(u_star - u_pred, np.inf) / np.linalg.norm(u_star, np.inf)

print(f"\n=== Final (target {curriculum_param}={target_value}) results ===")
print('Error u rel: %e' % error_u_relative)
print('Error u abs: %e' % error_u_abs)
print('Error u linf: %e' % error_u_linf)

if args.visualize:
    print(f"\nWriting visualization PDFs to: {os.path.abspath(path)}")
    os.makedirs(path, exist_ok=True)
    u_pred_grid = u_pred.reshape(len(t_full), len(x))
    exact_u(Exact, x, t_full, nu, beta, rho, orig_layers, args.N_f, args.L, args.source, args.u0_str, args.system, path=path)
    u_diff(Exact, u_pred_grid, x, t_full, nu, beta, rho, args.seed, orig_layers, args.N_f, args.L, args.source, args.lr, args.u0_str, args.system, path=path)
    u_predict(u_vals, u_pred_grid, x, t_full, nu, beta, rho, args.seed, orig_layers, args.N_f, args.L, args.source, args.lr, args.u0_str, args.system, path=path)
    print(f"Done. Files in {path}: {os.listdir(path)}")

if args.save_model:
    torch.save(model, f"{save_path}/curriculum_FINAL_{args.system}_u0{args.u0_str}_nu{nu}_beta{beta}_rho{rho}"
                       f"_Nf{args.N_f}_{args.layers}_L{args.L}_source{args.source}_seed{args.seed}.pt")
