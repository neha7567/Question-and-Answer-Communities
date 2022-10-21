import numpy as np
import scipy.stats as stats
import main_file_with_parameters as main_file
import utility
import finding_thresholds
from copy import copy

from termcolor import colored


def utility_asking_question_of_k(k, alpha, beta, asking_threshold, answering_threshold, k_array):
    if alpha * k < asking_threshold:  # asking threshold is difficulty level
        expected_num_answers = (beta *
                                k_array[(k_array >= max(alpha * k, answering_threshold))].size
                                / k_array[(k_array <= asking_threshold / alpha)].size) ** .5
        return expected_num_answers
    else:
        return 0


def utility_answering_question_of_k(answering_threshold, difficulty_array, beta, k_array, reward_ans, k):
    util = 0
    number_questions_to_pick = min(beta, len(difficulty_array.keys()))

    if number_questions_to_pick > 0:
        if k >= answering_threshold:
            choose_questions = stats.randint.rvs(0, len(difficulty_array.keys()), size=number_questions_to_pick)
            chosen_users = np.array(list(difficulty_array.keys()))[choose_questions]  # chosen askers knowledge
            for i in chosen_users:
                k_star_for_i_question = max(difficulty_array[i], answering_threshold)
                if k > k_star_for_i_question:
                    x = (reward_ans * len(difficulty_array) / (beta * k_array[k_array > k_star_for_i_question].size)) - \
                                utility.cost_answering(k)
                        #print(colored('X is negative, check why', 'red'))
                    util += max(0, x)

    return util


def get_joining_array(k_array_current, k_join_array_last_p, r_a, r_q, c_q, alpha, beta, outside_users_k_array):
    k_joining_array = []
    k_array = np.append(copy(k_array_current), copy(k_join_array_last_p))

    ask_threshold, ans_threshold = finding_thresholds.find_asking_threshold(k_array, r_a, r_q, c_q, alpha,
                                                                            beta)
    difficulty_arr = utility.generate_questions_hardness(k_array, ask_threshold, alpha)

    for i in outside_users_k_array:
        u_ask = max(r_q * utility_asking_question_of_k(i, alpha, beta, ask_threshold, ans_threshold, k_array) - c_q, 0)
        u_ans = utility_answering_question_of_k(ans_threshold, difficulty_arr, beta, k_array, r_a, i)
        if u_ask + u_ans > main_file.init_outside_option:
            k_joining_array.append(i)
            # print("asking_thres=", ask_threshold/alpha, "ans_thres=", ans_threshold,
            #     'u_ask=', u_ask, 'u_ans=', u_ans, 'for k =', i)
    print("joining_thresholds=", ask_threshold / alpha, ans_threshold, "mean_of_k_array=", np.mean(k_array),
          "variance of k_array=", np.var(k_array), "num_joined=", len(k_joining_array))
    return np.array(k_joining_array)