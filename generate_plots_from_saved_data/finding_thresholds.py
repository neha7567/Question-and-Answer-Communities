import numpy as np
import main_file_with_parameters as main_file
import utility


def convert_array_to_cdf_function(array):
    hist, bin_edges = np.histogram(array, bins=np.linspace(0, 1, num=100), density=True)
    cdf = np.cumsum(hist * np.diff(bin_edges))
    return np.insert(cdf, 0, 0), bin_edges


def convert_array_to_pdf_function(array):
    pdf, bin_edges = np.histogram(array, bins=np.linspace(0, 1, num=100), density=True)
    return np.insert(pdf, 0, 0), bin_edges


def equilibrium_equations_core_periphery(k_array, reward_a, reward_q, c_q, beta, alpha,
                                         x):  # for computing the asking threshold for askers
    # in core periphery, x[0]/alpha > x[1]. Lets keep asking in difficulty level and answering threshold in knowledge units.

    exp_answers_total = beta * k_array[(k_array >= x[0] / alpha)].size / k_array.size
    exp_questions = k_array[(k_array <= x[0] / alpha)].size / k_array.size
    return [reward_q * exp_answers_total ** .5 - c_q * exp_questions ** .5,
            (reward_a * exp_questions - (utility.cost_answering(x[1]) *
                                         beta * k_array[(k_array >= x[1])].size / k_array.size))]


def equilibrium_equations_non_core_periphery(k_array, reward_a, reward_q, c_q, beta, alpha,
                                             x):  # for computing the asking threshold for askers
    exp_answers_total = beta * k_array[(k_array >= x[1])].size / k_array.size
    exp_questions = k_array[(k_array <= x[0] / alpha)].size / k_array.size
    return [reward_q * exp_answers_total ** .5 - c_q * exp_questions ** .5,
            (reward_a * exp_questions - (utility.cost_answering(x[1])
                                         * beta * k_array[(k_array >= x[1])].size / k_array.size))]


def answering_threshold_equation(reward_a, ask_thresh, alpha, beta, k_array, y):
    # after questions are asked, the actual answering threshold
    cdf, bins = convert_array_to_cdf_function(k_array)
    cdf_at_y = cdf[np.argmax(bins > y)]
    prop_asking = cdf[np.argmax(bins > ask_thresh / alpha)]
    return (reward_a * prop_asking) - (utility.cost_answering(y) * beta * (1 - cdf_at_y))


def find_asking_threshold(k_array, reward_a, reward_q, c_q, alpha, beta):
    d_arr = np.linspace(0, alpha, num=100)
    cdf_arr, bins_arr = convert_array_to_cdf_function(k_array)
    util, ans_thresh = np.zeros(len(d_arr)), np.zeros(len(d_arr))
    for i in range(len(d_arr)):
        diff = d_arr[i]
        if answering_threshold_equation(reward_a, diff, alpha, beta, k_array, 0) >= 0:
            ans_thresh[i] = 0
        else:
            k_range = np.linspace(0, 1, num=100)
            ans_util = np.zeros(len(k_range))
            for j in range(len(k_range)):
                ans_util[j] = answering_threshold_equation(reward_a, diff, alpha, beta, k_array, k_range[j])
            ans_thresh[i] = k_range[np.argmax(ans_util > 0)]

        if cdf_arr[np.argmax(bins_arr > diff / alpha)] > 0 :
            prop_answers_on_d_star = max(0, beta * (1 - cdf_arr[np.argmax(bins_arr > max(diff, ans_thresh[i]))]) \
                                 / cdf_arr[np.argmax(bins_arr > diff / alpha)])
        else:
            prop_answers_on_d_star = 0

        util[i] = (reward_q * prop_answers_on_d_star ** .5) - c_q

    if util[util < 0] == []:
        asking_threshold = alpha
        answering_threshold = ans_thresh[-1]
    else:
        index_needed = np.argmax(util < 0)
        asking_threshold = d_arr[index_needed]
        answering_threshold = ans_thresh[index_needed]

    # print('asking_threshold =', str(asking_threshold / alpha), 'answering_threshold =', str(answering_threshold))
    return asking_threshold, answering_threshold
