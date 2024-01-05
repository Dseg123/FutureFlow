import osqp
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

#x has form x0, x1, x2


def optimal_traj(targ_x, targ_y, x0, vx0, y0, vy0, num_steps, dt, max_u, force_weight = 0, winds = None):
    if winds is None:
        x_winds = [0 for i in range(num_steps + 1)]
        y_winds = [0 for i in range(num_steps + 1)]
    else:
        x_winds = winds[:, 0]
        y_winds = winds[:, 1]

    #x has form x0, x1, x2... xN, y0, y1, y2 ... yN, vx0, vx1, vx2... vxN, vy0, vy1, ... vyN, ux0, ux1... uxN, uy0, uy1 .. uyN

    #objective is minimize (x0 - targ_x)^2 + (x1 - tx)^2... + (xN - tx)^2 + lambda * (u1^2 + u2^2... + uN^2)
    
    N = num_steps + 1

    P = np.zeros((6*N, 6*N))
    q = np.zeros(6*N)
    for i in range(2*N):
        P[i, i] = 1
        P[i + 4*N, i + 4*N] = force_weight
    P = 2 * P

    for i in range(N):
        q[i] = -2 * targ_x
    for i in range(N):
        q[N + i] = -2 * targ_y
    
    #x1 = x0 + v0 * dt --> x0 - x1 + dt * v0 = 0
    A_pos = np.zeros((2*(N - 1), 6 * N))
    l_pos = np.zeros(2*(N - 1))
    u_pos = np.zeros(2*(N - 1))
    for i in range(N - 1):
        A_pos[i, i] = 1
        A_pos[i, i + 1] = -1
        A_pos[i, i + 2*N] = dt
        l_pos[i] = 0
        u_pos[i] = 0
    for i in range(N - 1):
        A_pos[N - 1 + i, N + i] = 1
        A_pos[N - 1 + i, N + i + 1] = -1
        A_pos[N - 1 + i, N + i + 2*N] = dt
        l_pos[N - 1 + i] = 0
        u_pos[N - 1 + i] = 0

    #v1 = v0 + (u0 + f0) * dt --> v0 - v1 + dt * u0 = -dt * f0
    A_vel = np.zeros((2*(N - 1), 6 * N))
    l_vel = np.zeros(2*(N - 1))
    u_vel = np.zeros(2*(N - 1))
    for i in range(N - 1):
        A_vel[i, i + 2*N] = 1
        A_vel[i, i + 2*N + 1] = -1
        A_vel[i, i + 4*N] = dt
        l_vel[i] = -dt * x_winds[i]
        u_vel[i] = -dt * x_winds[i]
    for i in range(N - 1):
        A_vel[i + N - 1, i + 3*N] = 1
        A_vel[i + N - 1, i + 3*N + 1] = -1
        A_vel[i + N - 1, i + 5*N] = dt
        l_vel[i + N - 1] = -dt * y_winds[i]
        u_vel[i + N - 1] = -dt * y_winds[i]
    
    #x0 = x0, v0 = v0
    A_bound = np.zeros((4, 6*N))
    l_bound = np.zeros(4)
    u_bound = np.zeros(4)

    A_bound[0, 0] = 1
    A_bound[1, 2*N] = 1
    A_bound[2, N] = 1
    A_bound[3, 3*N] = 1
    l_bound[0] = x0
    u_bound[0] = x0
    l_bound[1] = vx0
    u_bound[1] = vx0
    l_bound[2] = y0
    u_bound[2] = y0
    l_bound[3] = vy0
    u_bound[3] = vy0


    #-max_u < u0 < max_u
    A_lim = np.zeros((2*N, 6*N))
    l_lim = np.zeros(2*N)
    u_lim = np.zeros(2*N)

    for i in range(2*N):
        A_lim[i, i + 4*N] = 1
        l_lim[i] = -max_u
        u_lim[i] = max_u
    
    A = np.concatenate((A_pos, A_vel, A_bound, A_lim), axis=0)
    l = np.concatenate((l_pos, l_vel, l_bound, l_lim))
    u = np.concatenate((u_pos, u_vel, u_bound, u_lim))

    A = sparse.csc_matrix(A)
    P = sparse.csc_matrix(P)

    prob = osqp.OSQP()
    prob.setup(P, q, A, l, u, alpha=1.0)

    res = prob.solve()

    return res.x

if __name__ == "__main__":
    targ_x = 0
    targ_y = 0
    x0 = 1
    vx0 = 0
    y0 = 1
    vy0 = 0
    num_steps = 100
    dt = 0.1
    max_u = 2
    L = 0.2
    x_winds = [2*np.sin(i / 10) for i in range(num_steps+1)]
    y_winds = [2*np.cos(i / 10) for i in range(num_steps+1)]
    winds = np.array([x_winds, y_winds]).T
    print(winds)
    print(winds.shape)

    traj = optimal_traj(targ_x, targ_y, x0, vx0, y0, vy0, num_steps, dt, max_u, winds=winds)
    fig, axs = plt.subplots(6)
    N = num_steps+1
    axs[0].plot(traj[:N])
    axs[1].plot(traj[N:2*N])
    axs[2].plot(traj[2*N:3*N])
    axs[3].plot(traj[3*N:4*N])
    axs[4].plot(traj[4*N:5*N])
    axs[5].plot(traj[5*N:])
    plt.show()




