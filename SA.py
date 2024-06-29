import numpy as np


def SA_TSP(data, places, initial_temperature=1, end_temperature=1e-5):

    def decrease_temperature(init_temperature, iteration):
        return init_temperature / (iteration + 1)

    def transition_probability(dE, T):
        return np.exp(-dE / T)

    def calculate_energy(route):
        return data.route_distance(route)

    def generate_state(seq):
        # seq - previous state (route)
        n = len(seq)
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        while abs(i - j) < 2:
            j = np.random.randint(0, n)
        if i > j:
            i, j = j, i
        rs = seq[i:j]
        rs.reverse()
        return seq[:i] + rs + seq[j:]

    # main SA procedure
    state = places
    #np.random.shuffle(state)
    energy = calculate_energy(state)
    T = initial_temperature

    i = 0
    states = []
    while T > end_temperature:
        states.append(state)
        state_candidate = generate_state(state)
        candidate_energy = calculate_energy(state_candidate)
        if candidate_energy < energy:
            energy = candidate_energy
            state = state_candidate
        else:
            if transition_probability(candidate_energy - energy, T) >= np.random.random():
                energy = candidate_energy
                state = state_candidate
        T = decrease_temperature(initial_temperature, i)
        i += 1

    return states