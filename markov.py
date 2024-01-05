import numpy as np

def get_tpm(data_arr, max_state = 5, num_states = 100):
    tpm = np.zeros((data_arr.shape[1], num_states, num_states))
    state_size = 2 * max_state / num_states

    for i in range(data_arr.shape[0] - 1):
        print(i)
        for j in range(data_arr.shape[1]):
            first = data_arr[i, j]
            sec = data_arr[i + 1, j]

            first_state = (first + max_state) // (state_size)
            first_state = int(min(max(first_state, 0), num_states - 1))
            sec_state = (sec + max_state) // (state_size)
            sec_state = int(min(max(sec_state, 0), num_states - 1))

            tpm[j, first_state, sec_state] += 1

    for i in range(tpm.shape[0]):
        for j in range(tpm.shape[1]):
            mean = tpm[i, j, :].sum()
            for k in range(tpm.shape[1]):
                tpm[i, j, k] = tpm[i, j, k]/mean
    
    return tpm
            


