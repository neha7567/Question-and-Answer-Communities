import numpy as np
from copy import copy
import scipy.stats as stats


# given answer_to user dict, update knowledge #

def get_k_leaving_array_disappointed_and_update_knowledge_gain(k_array_after_answers_received, answer_to_user, delta):
    k_array_copy = copy(k_array_after_answers_received)
    indices_leaving_disappointed, indices_gain_k, leaving_array = [], [], []
    for i in answer_to_user.keys():
        if answer_to_user[i] == 0:
            leaving_array.append(i)
            indices_leaving_disappointed.append(np.argwhere(k_array_copy == i)[0][0])
        else:
            indices_gain_k.append(np.argwhere(k_array_copy == i)[0][0])
    k_array_copy[indices_gain_k] += delta
    k_array_copy = np.delete(k_array_copy, indices_leaving_disappointed, None)
    return np.array(leaving_array), k_array_copy


# randomly leaving users
def get_k_leaving_indices_random(k_array_after_unanswered_leave, theta, max_iteration):
    x = 0
    for m in range(max_iteration):
        x += stats.binom.rvs(len(k_array_after_unanswered_leave), theta)
    leaving_pop = int(x / max_iteration)

    indices_leaving = stats.randint.rvs(low=0, high=len(k_array_after_unanswered_leave), size=leaving_pop)
    leaving_arr = k_array_after_unanswered_leave[indices_leaving]
    final_array = np.delete(k_array_after_unanswered_leave, indices_leaving, None)
    return leaving_arr, final_array


def update_dist_parameters(k_array):
    variance = np.var(k_array)
    mean_value = np.mean(k_array)
    parameter_alpha = (mean_value ** 2 * (1 - mean_value) / variance) - mean_value
    parameter_beta = parameter_alpha * (1 - mean_value) / mean_value
    # parameters = stats.beta.fit(k_array)
    # return parameters[:-2]
    return parameter_alpha, parameter_beta