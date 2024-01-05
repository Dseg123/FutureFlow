import osqp
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

#x has form x0, x1, x2


def optimal_traj(targ_x, x0, v0, num_steps, dt, max_u, force_weight = 0, winds = None):
    if winds is None:
        winds = [0 for i in range(num_steps + 1)]

    #x has form x0, x1, x2... xN, v0, v1, v2... vN, u0, u1... uN

    #objective is minimize (x0 - targ_x)^2 + (x1 - tx)^2... + (xN - tx)^2 + lambda * (u1^2 + u2^2... + uN^2)
    
    N = num_steps + 1

    P = np.zeros((3*N, 3*N))
    q = np.zeros(3*N)
    for i in range(N):
        P[i, i] = 1
        P[i + 2*N, i + 2*N] = force_weight
    P = 2 * P

    for i in range(N):
        q[i] = -2 * targ_x
    
    #x1 = x0 + v0 * dt --> x0 - x1 + dt * v0 = 0
    A_pos = np.zeros((N - 1, 3 * N))
    l_pos = np.zeros(N - 1)
    u_pos = np.zeros(N - 1)
    for i in range(N - 1):
        A_pos[i, i] = 1
        A_pos[i, i + 1] = -1
        A_pos[i, i + N] = dt
        l_pos[i] = 0
        u_pos[i] = 0

    #v1 = v0 + (u0 + f0) * dt --> v0 - v1 + dt * u0 = -dt * f0
    A_vel = np.zeros((N - 1, 3 * N))
    l_vel = np.zeros(N - 1)
    u_vel = np.zeros(N - 1)
    for i in range(N - 1):
        A_vel[i, i + N] = 1
        A_vel[i, i + N + 1] = -1
        A_vel[i, i + 2*N] = dt
        l_vel[i] = -dt * winds[i]
        u_vel[i] = -dt * winds[i]
    
    #x0 = x0, v0 = v0
    A_bound = np.zeros((2, 3*N))
    l_bound = np.zeros(2)
    u_bound = np.zeros(2)

    A_bound[0, 0] = 1
    A_bound[1, N] = 1
    l_bound[0] = x0
    u_bound[0] = x0
    l_bound[1] = v0
    u_bound[1] = v0


    #-max_u < u0 < max_u
    A_lim = np.zeros((N, 3*N))
    l_lim = np.zeros(N)
    u_lim = np.zeros(N)

    for i in range(N):
        A_lim[i, i + 2*N] = 1
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
    x0 = 1
    v0 = 1
    num_steps = 100
    dt = 0.1
    max_u = 2
    L = 0.2
    winds = [2*np.sin(i / 10) for i in range(num_steps+1)]

    traj = optimal_traj(targ_x, x0, v0, num_steps, dt, max_u, winds=winds)
    fig, axs = plt.subplots(3)
    axs[0].plot(traj[:num_steps+1])
    axs[1].plot(traj[num_steps+1:2*num_steps+1])
    axs[2].plot(traj[2*num_steps+1:])
    plt.show()




