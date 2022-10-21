import pickle as pkl
import numpy as np


def get_final_list_dicts_and_save(result_dicts, r_q, c_q_vector, num_iter, T, filename):
    x_axis_time = range(T)

    num_questions_final, total_num_answers_final, num_unanswered_questions_final, question_threshold_final, \
    answer_threshold_final, alpha_final, beta_final, k_array_final, mean_final, num_users_final, \
    answer_to_user_final, leaving_k_list_disappointed_final, joining_array_list_final, \
    leaving_k_list_random_final = result_dicts

    proportion_unanswered_dict_in_dict = \
        compute_proportion_unanswered_questions(num_questions_final, num_unanswered_questions_final,
                                                c_q_vector, r_q, num_iter, x_axis_time)

    num_left = compute_num_from_array(leaving_k_list_disappointed_final, num_iter, c_q_vector, r_q)
    joining_arrays_modified = remove_first_entry_each_dict(joining_array_list_final, num_iter, c_q_vector, r_q)
    num_joined = compute_num_from_array(joining_arrays_modified, num_iter, c_q_vector, r_q)
    num_answerers, proportion_answerers = compute_num_answerers(answer_threshold_final, k_array_final, T,
                                                                num_iter, c_q_vector, r_q)

    list_of_final_dicts = [num_questions_final, total_num_answers_final, num_unanswered_questions_final,
                           question_threshold_final, answer_threshold_final, alpha_final, beta_final,
                           k_array_final, mean_final, num_users_final, answer_to_user_final,
                           leaving_k_list_disappointed_final, joining_arrays_modified, leaving_k_list_random_final,
                           proportion_unanswered_dict_in_dict, num_left, num_joined, num_answerers,
                           proportion_answerers]
    pkl.dump(list_of_final_dicts, open(filename, "wb"))
    return list_of_final_dicts


def remove_first_entry_each_dict(dict_result, num_iteration, cost_ask_vec, reward_ask_vec):
    final = {}
    for i in cost_ask_vec:
        for j in reward_ask_vec:
            for num in range(num_iteration):
                final[(i, j, num)] = dict_result[(i, j, num)][1:]
    return final


def compute_num_answerers(threshold_dict, k_array_dict, total_time,
                          samples, cost_ask_vec, reward_ask_vec):
    num_answerers, proportion_answerers = {}, {}
    for i in cost_ask_vec:
        for k in reward_ask_vec:
            for num in range(samples):
                num_answerers[(i, k, num)] = np.array([
                    k_array_dict[(i, k, num)][j][k_array_dict[(i, k, num)][j] >= threshold_dict[(i, k, num)][j]].size
                    for j in range(total_time)])
                proportion_answerers[(i, k, num)] = np.array([
                    k_array_dict[(i, k, num)][j][k_array_dict[(i, k, num)][j] >=
                                                 threshold_dict[(i, k, num)][j]].size / k_array_dict[(i, k, num)][
                        j].size
                    for j in range(total_time)])

    return num_answerers, proportion_answerers


def compute_num_from_array(dict_result, total_iter, cost_ask_vec, reward_ask_vec):
    result = {}
    for i in cost_ask_vec:
        for j in reward_ask_vec:
            for num in range(total_iter):
                result[i, j, num] = [len(k) for k in dict_result[(i, j, num)]]
    return result


def compute_long_run_averages(dict_result, total_iter, cost_ask_vec, reward_ask_vec, time_averaged):
    result = {(i, j, num): 0 for i in cost_ask_vec for j in reward_ask_vec for num in range(total_iter)}
    for i in cost_ask_vec:
        for j in reward_ask_vec:
            for num in range(total_iter):
                result[(i, j, num)] = [i, j, num, np.mean(np.array(dict_result[(i, j, num)][-time_averaged:]))]
    return result


def compute_proportion_unanswered_questions(total_questions_dict, unanswered_questions_dict, cost_ask_vec,
                                            reward_ask_vec, iterations_total, x_axis_time):
    proportion_unanswered = {}
    for i in cost_ask_vec:
        for k in reward_ask_vec:
            for num in range(iterations_total):
                proportion_unanswered[(i, k, num)] = np.array([
                    unanswered_questions_dict[(i, k, num)][j] /
                    (total_questions_dict[(i, k, num)][
                         j] + 1e-5) for j in x_axis_time])
    return proportion_unanswered


def required_qty_for_c_q_plots_from_saved_data(file, num_iter, c_q_vector, time_avg, r_q_vector):
    [num_questions_final, total_num_answers_final, num_unanswered_questions_final,
     question_threshold_final, answer_threshold_final, alpha_final, beta_final,
     k_array_final, mean_final, num_users_final, answer_to_user_final,
     leaving_k_list_disappointed_final, joining_arrays_modified, leaving_k_list_random_final,
     proportion_unanswered_dict_in_dict, num_left, num_joined, num_answerers,
     proportion_answerers] = pkl.load(open(file, 'rb'))

    num_users_with_c_q = compute_long_run_averages(num_users_final, num_iter, c_q_vector, r_q_vector,
                                                   time_averaged=time_avg)
    num_joined_with_c_q = compute_long_run_averages(num_joined, num_iter, c_q_vector, r_q_vector,
                                                    time_averaged=time_avg)
    num_left_disappointed_with_c_q = compute_long_run_averages(num_left, num_iter, c_q_vector, r_q_vector,
                                                               time_averaged=time_avg)
    num_questions_with_c_q = compute_long_run_averages(num_questions_final, num_iter, c_q_vector,
                                                       r_q_vector, time_averaged=time_avg)
    num_answers_with_c_q = compute_long_run_averages(total_num_answers_final, num_iter, c_q_vector,
                                                     r_q_vector, time_averaged=time_avg)
    num_unanswered_questions_with_c_q = compute_long_run_averages(num_unanswered_questions_final, num_iter,
                                                                  c_q_vector, r_q_vector,
                                                                  time_averaged=time_avg)
    num_answerers_with_c_q = compute_long_run_averages(num_answerers, num_iter, c_q_vector, r_q_vector,
                                                       time_averaged=time_avg)
    prop_answerers_with_c_q = compute_long_run_averages(proportion_answerers, num_iter, c_q_vector,
                                                        r_q_vector, time_averaged=time_avg)
    prop_unanswered_with_c_q = compute_long_run_averages(proportion_unanswered_dict_in_dict, num_iter,
                                                         c_q_vector, r_q_vector, time_averaged=time_avg)
    mean_knowledge_with_c_q = compute_long_run_averages(mean_final, num_iter, c_q_vector, r_q_vector,
                                                        time_averaged=time_avg)
    question_threshold_with_c_q = compute_long_run_averages(question_threshold_final, num_iter, c_q_vector,
                                                            r_q_vector, time_averaged=time_avg)
    answer_threshold_with_c_q = compute_long_run_averages(answer_threshold_final, num_iter, c_q_vector,
                                                          r_q_vector, time_averaged=time_avg)

    return [num_users_with_c_q, num_joined_with_c_q, num_left_disappointed_with_c_q, num_questions_with_c_q,
            num_answers_with_c_q, num_answerers_with_c_q, prop_answerers_with_c_q, prop_unanswered_with_c_q,
            mean_knowledge_with_c_q, question_threshold_with_c_q, answer_threshold_with_c_q,
            num_unanswered_questions_with_c_q]


def get_proportions_after_data_is_saved(num_dict, den_dict, c_q_vec, r_q_vec, samples):
    result = {}
    for i in c_q_vec:
        for j in r_q_vec:
            for num in range(samples):
                if den_dict[(i, j, num)][3] > 0:
                    result[(i, j, num)] = [i, j, num,
                                           num_dict[(i, j, num)][3] / den_dict[(i, j, num)][3]]
                else:
                    result[(i, j, num)] = [i, j, num, 0]

    return result
