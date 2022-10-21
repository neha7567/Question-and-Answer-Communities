import numpy as np
import multiprocessing as mp
import pickle as pkl


init_cost_ans_max = 1.01
init_prob_leaving = 0.20
init_outside_option = 1
init_T = 250
init_delta = 1e-3
init_max_iter_for_internal_sim = 20
init_num_iter = 10
num_outside, difficulty_parameter, cognitive_parameter_vector = 1000, 1.3, 4
reward_q_vec, c_q_vector = [1.5], np.linspace(.5, 2, num=15)  # np.linspace(.8, 3, num=2)  # [.3, .4, .5]
reward = 2
time_avg, x_axis_time = init_T - 50, list(range(init_T))
initial_dist_parameters = [(1, 1), (1, 2), (2, 1)]
result_dist = {}


def store_results_in_file(result_file, num):
    output_data = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
    for i in result_file: #list of len(c_q_vector), each has 14 dicts
        for dict_number in range(len(i)):
            output_data[dict_number].update(i[dict_number])

    # should be list of 14 dictionaries with keys c_q, r_q, num_iter
    aggregating_fn.get_final_list_dicts_and_save(output_data, reward_q_vec, c_q_vector,
                                                 init_num_iter, init_T, 'qa_forums_parallel/aggregated_output_'
                                                                        'for_dist{}.pkl'.format(num))


if __name__ == "__main__":
    import running_simulation
    import modify_results_for_plots_and_save as aggregating_fn
    for j in range(len(initial_dist_parameters)):
        pool = mp.Pool(mp.cpu_count())
        final_results_dict_list, file_log_list = [], []
        result_dist[j] = pool.starmap_async(running_simulation.get_all_sample_paths_with_c_q,
                                            [(reward, reward_q_vec[0], c_q, num_outside, difficulty_parameter,
                                              cognitive_parameter_vector, init_T, initial_dist_parameters[j]) for c_q in
                                             c_q_vector]).get()  # list of 16 dictionaries
        pool.close()
        print(len(result_dist[j]))
        store_results_in_file(result_dist[j], j)
