#!/usr/bin/env python
"""
Train an acyclic REN controller for the system of 2 robots in a corridor or 12 robots swapping positions.
Author: Clara Galimberti (clara.galimberti@epfl.ch)
Usage:
python run.py                --sys_model     [SYS_MODEL]         \
                             --gpu           [USE_GPU]           \
Flags:
  --sys_model: select system where to design a controller. Available options: corridor, robots.
  --gpu: whether to use GPU.
"""
import numpy as np
import torch
import argparse

from src.models import SystemRobots, Controller, SystemRobotsDist
from src.plots import plot_trajectories, plot_traj_vs_time
from src.loss_functions import f_loss_states, f_loss_u, f_loss_ca, f_loss_obst
from src.utils import calculate_collisions, set_params


def main(sys_model):
    """
    :param sys_model: System where to design a controller. Select from: "corridor", "robots".
    :param gpu: Whether to use GPU during training.
    :return:
    """

    ny = 2  # output dimension of single ren = input dimension of single ren from other ren
    nx = 4  # number of states single agent = dimension of single exogenous input
    n_agents = 4
    torch.manual_seed(1)

    M1 = torch.hstack((torch.zeros((4, 8)), torch.eye(4), torch.zeros((4, 12))))
    M2 = torch.hstack((torch.zeros((2, 2)), torch.eye(2), torch.zeros((2, 20))))
    M3 = torch.hstack((torch.zeros((2, 6)), torch.eye(2), torch.zeros((2, 16))))
    M4 = torch.hstack((torch.zeros((4, 12)), torch.eye(4), torch.zeros((4, 8))))
    M5 = torch.hstack((torch.zeros((2, 4)), torch.eye(2), torch.zeros((2, 18))))
    M6 = torch.hstack((torch.eye(2), torch.zeros((2, 22))))
    M7 = torch.hstack((torch.zeros((4, 16)), torch.eye(4), torch.zeros((4, 4))))
    M8 = torch.hstack((torch.zeros((2, 2)), torch.eye(2), torch.zeros((2, 20))))
    M9 = torch.hstack((torch.zeros((2, 6)), torch.eye(2), torch.zeros((2, 16))))
    M10 = torch.hstack((torch.zeros((4, 20)), torch.eye(4)))
    M11 = torch.hstack(( torch.eye(2), torch.zeros((2, 22))))
    M12 = torch.hstack((torch.zeros((2, 4)), torch.eye(2), torch.zeros((2, 18))))

    M = torch.vstack((M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12))

    Muy = M[:, 0:8]
    Mud = M[:, 7:-1]

    N = 4

    # Muy = torch.vstack((torch.hstack((o2, i2)), o4, torch.hstack((i2, o2)), o4))
    # Mud = torch.vstack((torch.zeros((2, 8)),
    #  torch.hstack((torch.eye(4), o4)), torch.zeros((2, 8)),
    #  torch.hstack((o4, torch.eye(4)))))

    # # # # # # # # Parameters and hyperparameters # # # # # # # #
    if sys_model == "corridor" or sys_model == "robots" or sys_model == "distributedREN":
        params = set_params(sys_model)
        min_dist, t_end, n_agents, x0, xbar, xbarspring, linear, learning_rate, epochs, Q, \
            alpha_u, alpha_ca, alpha_obst, n_xi, l, n_traj, std_ini = params
    else:
        raise ValueError("System model not implemented.")

    if sys_model == "distributedREN":
        sys = SystemRobotsDist(xbarspring, xbar, linear)
    else:
        sys = SystemRobots(xbar, linear)

    # # # # # # # # Define models # # # # # # # #

    ctl = Controller(sys.f, N, Muy, Mud, np.array([ny + nx, ny + nx, ny + nx, ny + nx]), np.array([ny, ny, ny, ny]),
                     n_xi, l)
    # # # # # # # # Define optimizer and parameters # # # # # # # #
    optimizer = torch.optim.Adam(ctl.parameters(), lr=learning_rate)
    # # # # # # # # Training # # # # # # # #
    print("------------ Begin training ------------")
    print("Problem: " + sys_model + " -- t_end: %i" % t_end + " -- lr: %.2e" % learning_rate +
          " -- epochs: %i" % epochs + " -- n_traj: %i" % n_traj + " -- std_ini: %.2f" % std_ini)
    print(" -- alpha_u: %.1f" % alpha_u + " -- alpha_ca: %i" % alpha_ca + " -- alpha_obst: %.1e" % alpha_obst)
    print("--------- --------- ---------  ---------")
    for epoch in range(epochs):
        gamma = []
        optimizer.zero_grad()
        loss_x, loss_u, loss_ca, loss_obst = 0, 0, 0, 0
        if epoch == 300 and sys_model == 'corridor':
            std_ini = 0.5
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']
        for kk in range(n_traj):
            w_in = torch.zeros(t_end + 1, sys.n)
            w_in[0, :] = (x0.detach() - sys.xbar) + std_ini * torch.randn(x0.shape)
            x = sys.xbar
            u = torch.zeros(sys.m)
            xi = torch.zeros((sum(n_xi)))
            omega = (x, u)
            for t in range(t_end):
                x, _ = sys(t, x, u, w_in[t, :])
                u, xi, omega, gamma = ctl(t, u, x, xi, omega)
                loss_x = loss_x + f_loss_states(t, x, sys, Q)
                loss_u = loss_u + alpha_u * f_loss_u(t, u)
                loss_ca = loss_ca + alpha_ca * f_loss_ca(x, sys, min_dist)
                if alpha_obst != 0:
                    loss_obst = loss_obst + alpha_obst * f_loss_obst(x)
        loss = loss_x + loss_u + loss_ca + loss_obst
        print("Epoch: %i --- Loss: %.4f ---||--- Loss x: %.2f --- " % (epoch, loss / t_end, loss_x) +
              "Loss u: %.2f --- Loss ca: %.2f --- Loss obst: %.2f" % (loss_u, loss_ca, loss_obst))
        print(gamma)
        loss.backward(retain_graph=True)
        optimizer.step()
    # # # # # # # # Save trained model # # # # # # # #
    torch.save(ctl.psi_u.state_dict(), "trained_models/" + sys_model + "_tmp.pt")
    # # # # # # # # Print & plot results # # # # # # # #
    x_log = torch.zeros((t_end, sys.n))
    u_log = torch.zeros((t_end, sys.m))
    w_in = torch.zeros((t_end + 1, sys.n))
    w_in[0, :] = (x0.detach() - sys.xbar)
    u = torch.zeros(sys.m)
    x = sys.xbar
    xi = torch.zeros((sum(n_xi)))
    omega = (x, u)
    for t in range(t_end):
        x, _ = sys(t, x, u, w_in[t, :])
        u, xi, omega, gamma = ctl(t, u, x, xi, omega)
        x_log[t, :] = x.detach()
        u_log[t, :] = u.detach()
    plot_traj_vs_time(t_end, sys.n_agents, x_log, u_log)
    # Number of collisions
    n_coll = calculate_collisions(x_log, sys, min_dist)
    print("Number of collisions after training: %d" % n_coll)
    # Extended time
    t_ext = t_end * 4
    x_log = torch.zeros(t_ext, sys.n)
    u_log = torch.zeros(t_ext, sys.m)
    w_in = torch.zeros(t_ext + 1, sys.n)
    w_in[0, :] = (x0.detach() - sys.xbar)
    u = torch.zeros(sys.m)
    x = sys.xbar
    xi = torch.zeros((sum(n_xi)))
    omega = (x, u)
    for t in range(t_ext):
        x, _ = sys(t, x, u, w_in[t, :])
        u, xi, omega, gamma = ctl(t, u, x, xi, omega)
        x_log[t, :] = x.detach()
        u_log[t, :] = u.detach()
    plot_trajectories(x_log, xbar, sys.n_agents, text="CL - after training - extended t", T=t_end, obst=alpha_obst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sys_model', type=str, default='distributedREN')
    args = parser.parse_args()
    # Run main
    main(args.sys_model)
