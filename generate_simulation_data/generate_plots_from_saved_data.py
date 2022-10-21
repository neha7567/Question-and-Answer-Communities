import pickle as pkl
import multiprocessing as mp
import main_file_with_parameters as main_file
import modify_results_for_plots_and_save as create_required_qty
import plots_to_generate as gen_plots

"""num_files = 3
final_results_dict_list = ['Q_A_forums_code_for_paper/qa_forums_parallel/aggregated_output_for_dist{}.pkl'.format(num)
                           for num in range(num_files)]"""
final_results_dict_list = ['C://Users//Neha Sharma//Google Drive//'
                           'Numerical_study_QA//Results_simulation//aggregated_output_for_dist2.pkl']
num_files = 1
dicts_needed_list = []
for file in final_results_dict_list:
    dicts_needed_list.append(create_required_qty.required_qty_for_c_q_plots_from_saved_data(file,
                                                                                            main_file.init_num_iter,
                                                                                            main_file.c_q_vector,
                                                                                            main_file.time_avg,
                                                                                            main_file.reward_q_vec))
if __name__ == '__main__':
    """pool = mp.Pool(mp.cpu_count())
    pool.starmap_async(gen_plots.generate_plots_with_c_q_r_q,
                       [(dicts_needed_list[num], main_file.c_q_vector, main_file.reward_q_vec, str(num)) for num in
                        range(num_files)]).get()
    pool.close()"""

    pool = mp.Pool(mp.cpu_count())
    pool.starmap_async(gen_plots.generate_plots_with_c_q_constant_r_q,
                       [(dicts_needed_list[num], main_file.c_q_vector,
                         main_file.reward_q_vec, str(num)) for num in
                        range(num_files)]).get()
    pool.close()

    """pool = mp.Pool(mp.cpu_count())
    pool.starmap_async(gen_plots.generate_plots_with_time,
                       [(pkl.load(open(final_results_dict_list[num], 'rb')),
                         [main_file.c_q_vector[0], main_file.c_q_vector[7]],
                         main_file.reward_q_vec, main_file.init_num_iter
                         , main_file.init_T, str(num)) for num in
                        range(num_files)]).get()
    pool.close()"""
