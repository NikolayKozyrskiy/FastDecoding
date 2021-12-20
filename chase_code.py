from scipy.stats import norm
import copy
import numpy as np
import itertools as itt
from mpi4py import MPI

def get_H(m):
    k = 2**m - 1 - m
    n = 2**m - 1
    H = np.zeros((n, n-k))
    for i, comb in enumerate(itt.product([0, 1], repeat=n-k)):
        if np.sum(comb) == 0:
            continue
        H[i-1, :] = np.array(comb)
    return H.astype(int)
    
def bpsk(x):
    y = np.where(x == 1, -np.ones_like(x), x)
    y = np.where(y == 0, np.ones_like(y), y)
    return y

def get_LLR(x):
    llr = np.ones_like(x)
    llr[0] = -0.5
    llr[-1] = -0.5
    return llr

def get_score(hard_decision, llr):
    return np.sum(hard_decision * np.abs(llr))

def get_H_row_idx(syndrome):
    idx = ''
    for i in syndrome:
        idx += str(i)
    idx = int(idx, base=2) - 1
    return idx

def process_combinations(llr, a, H, mpisize, mpirank):
    hard_decision = llr < 0
    sorting_idxs = np.argsort(np.abs(llr))[:a]
    scores = []
    hd_list = []
    vals_list = list(itt.product([0, 1], repeat=a))
    vals_per_proc = 2**a // mpisize
    residuals = 2**a % mpisize
    if mpirank < residuals:
        vals_per_proc += 1
        start_idx = mpirank * vals_per_proc
        stop_idx = (mpirank + 1) * vals_per_proc
    else:
        start_idx = mpirank * vals_per_proc + residuals
        stop_idx = (mpirank + 1) * vals_per_proc + residuals
    vals_list = vals_list[start_idx:stop_idx]   
    for vals in vals_list:
        new_hard_decision = copy.deepcopy(hard_decision)
        new_hard_decision[sorting_idxs] = vals
        # new_hard_decision[sorting_idxs] = np.asarray(vals)
        syndrome = (new_hard_decision @ H) % 2
        H_row_idx = get_H_row_idx(syndrome)
        if H_row_idx != -1:
            new_hard_decision[H_row_idx] = (new_hard_decision[H_row_idx] + 1) % 2
        scores.append(get_score(new_hard_decision, llr))
        hd_list.append(new_hard_decision)
    return scores, hd_list

def chase_code(a, H, mpisize, mpirank):
    code_word = bpsk(np.zeros(H.shape[0]))
    llr = get_LLR(code_word)
    start_time = MPI.Wtime()
    scores_part, hd_list_part = process_combinations(llr, a, H, mpisize, mpirank)
    stop_time = MPI.Wtime()
    scores = []
    scores = comm.gather(scores_part, root=0)
    hd_list = comm.gather(hd_list_part, root=0)
    gathered_time = comm.gather(stop_time - start_time, root=0)
    if mpirank == 0:
        scores = [i for batch in scores for i in batch]
        hd_list = [i for batch in hd_list for i in batch]
        return hd_list[np.argmin(scores)], gathered_time
    else:
        return None



comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

m = 14
a = 9
n = 2**m - 1
H = get_H(m)
N = 100

times = []
for _ in range(N):
    if rank == 0:
        res, time = chase_code(a, H, size, rank)
        times.append(np.max(time))
    else:
        chase_code(a, H, size, rank)

if rank == 0:
    with open(f'{size}.txt', "a") as f:
        f.write(str(np.mean(times)) + '\n')
