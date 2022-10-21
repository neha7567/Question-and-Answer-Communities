import creating_figures as plotting_function
import main_file_with_parameters as main_file
import modify_results_for_plots_and_save as needed_fns


def generate_plots_with_c_q_r_q(output_data, c_q_vector_reqd, r_q_vector_reqd, suffix):
    num_users_with_c_q, num_joined_with_c_q, num_left_disappointed_with_c_q, num_questions_with_c_q, \
    num_answers_with_c_q, num_answerers_with_c_q, prop_answerers_with_c_q, prop_unanswered_with_c_q, \
    mean_k_with_c_q, question_threshold_with_c_q, \
    answer_threshold_with_c_q, num_unanswered_questions_with_c_q = output_data

    plotting_function.plot_with_c_q([num_users_with_c_q], r_q_vector_reqd, c_q_vector_reqd,
                                    r'cost of asking, $c_q$', r'Number of users, $N^{EQ}$',
                                    ['num_users'], [r'Number of users, $N^{EQ}$'], 'num_users_with_c_q' + suffix)
    plotting_function.plot_with_c_q([num_joined_with_c_q], r_q_vector_reqd, c_q_vector_reqd,
                                    r' cost of asking, $c_q$',
                                    r'Number joining, $N_{join}^{EQ}$',
                                    ['num_joined'], [r'Number joining, $N_{join}^{EQ}$'], 'num_joined_with_c_q' + suffix)
    plotting_function.plot_with_c_q([num_left_disappointed_with_c_q], r_q_vector_reqd, c_q_vector_reqd,
                                    r' cost of asking, $c_q$',
                                    r'Number leaving, $N_{leave}^{EQ}$', ['num_left'],
                                    [r'Number leaving, $N_{leave}^{EQ}$'], 'num_unanswered_with_c_q' + suffix)
    plotting_function.plot_with_c_q([num_questions_with_c_q], r_q_vector_reqd, c_q_vector_reqd,
                                    r' cost of asking, $c_q$',
                                    r'Questions, $F^{EQ}(\mathcal{D}^{-1}(d^*))N^{EQ}$',
                                    ['total_questions'], [r'Questions, $F^{EQ}(\mathcal{D}^{-1}(d^*))N^{EQ}$'],
                                    'total_questions_with_c_q' + suffix)
    plotting_function.plot_with_c_q([num_answers_with_c_q], r_q_vector_reqd, c_q_vector_reqd,
                                    r' cost of asking, $c_q$',
                                    r'Answers, $\mathbb{E}_{F^{EQ}}[\sum_{k} y^{EQ}(k) | d^*]$',
                                    ['total_answers'], [r'Answers, $\mathbb{E}_{F^{EQ}}[\sum_{k} y^{EQ}(k) | d^*]$'],
                                    'total_answers_with_c_q' + suffix)
    plotting_function.plot_with_c_q([num_answerers_with_c_q], r_q_vector_reqd, c_q_vector_reqd,
                                    r' cost of asking, $c_q$',
                                    r'Answerers, $(1-F^{EQ}(\tilde{k}))N^{EQ}$',
                                    ['num_answering'], [r'Answerers, $(1-F^{EQ}(\tilde{k}))N^{EQ}$'],
                                    'num_answering_with_c_q' + suffix)
    plotting_function.plot_with_c_q([prop_answerers_with_c_q], r_q_vector_reqd, c_q_vector_reqd,
                                    r' cost of asking, $c_q$',
                                    r'Fraction Answerers, $(1-F^{EQ}(\tilde{k}))$', ['frac_answering'],
                                    [r'Fraction Answerers, $(1-F^{EQ}(\tilde{k}))$'],
                                    'proportion_answering_with_c_q' + suffix)
    plotting_function.plot_with_c_q([prop_unanswered_with_c_q], r_q_vector_reqd, c_q_vector_reqd,
                                    r' cost of asking, $c_q$',
                                    'Fraction Unanswered', ['frac_unanswered'], ['Fraction Unanswered'],
                                    'proportion_unanswered_with_c_q' + suffix)
    plotting_function.plot_with_c_q([num_unanswered_questions_with_c_q], r_q_vector_reqd, c_q_vector_reqd,
                                    r' cost of asking, $c_q$',
                                    'Questions Unanswered', ['num_unanswered'], ['Questions Unanswered'],
                                    'Num_unanswered_with_c_q' + suffix)
    plotting_function.plot_with_c_q([mean_k_with_c_q], r_q_vector_reqd, c_q_vector_reqd,
                                    r' cost of asking, $c_q$',
                                    'Mean Knowledge', ['mean_k'], ['Mean Knowledge'], 'mean_k_with_c_q' + suffix)
    plotting_function.plot_with_c_q([question_threshold_with_c_q], r_q_vector_reqd, c_q_vector_reqd,
                                    r' cost of asking, $c_q$',
                                    r'Asking threshold, $d^{*}$', ['asking_threshold_k'],
                                    [r'Asking threshold, $d^{*}$'],
                                    'asking_threshold_with_c_q' + suffix)
    plotting_function.plot_with_c_q([answer_threshold_with_c_q], r_q_vector_reqd, c_q_vector_reqd,
                                    r' cost of asking, $c_q$',
                                    r'Answering threshold, $\tilde{k}$', ['answering_thres'],
                                    [r'Answering threshold, $\tilde{k}$'],
                                    'answering_threshold_with_c_q' + suffix)


def generate_plots_with_c_q_constant_r_q(output_data, c_q_vector_reqd, r_q_vector_reqd, suffix):
    num_users_with_c_q, num_joined_with_c_q, num_left_disappointed_with_c_q, num_questions_with_c_q, \
    num_answers_with_c_q, num_answerers_with_c_q, prop_answerers_with_c_q, prop_unanswered_with_c_q, \
    mean_k_with_c_q, question_threshold_with_c_q, \
    answer_threshold_with_c_q, num_unanswered_questions_with_c_q = output_data

    proportion_asking = needed_fns.get_proportions_after_data_is_saved(num_questions_with_c_q, num_users_with_c_q,
                                                                       c_q_vector_reqd,
                                                                       r_q_vector_reqd, main_file.init_num_iter)
    answer_per_q = needed_fns.get_proportions_after_data_is_saved(num_answers_with_c_q,
                                                                  num_questions_with_c_q, c_q_vector_reqd,
                                                                  r_q_vector_reqd, main_file.init_num_iter)

    plotting_function.plot_with_c_q([num_joined_with_c_q, num_left_disappointed_with_c_q], r_q_vector_reqd,
                                    c_q_vector_reqd,
                                    r' cost of asking, $c_q$',
                                    'Number of users', ['Joining', 'Leaving'],
                                    [r'Number joining, $N_{join}^{EQ}$', r'Number leaving, $N_{leave}^{EQ}$'],
                                    'joining_leaving_with_c_q' + suffix)
    plotting_function.plot_with_c_q([num_questions_with_c_q, num_answers_with_c_q, num_answerers_with_c_q],
                                    r_q_vector_reqd,
                                    c_q_vector_reqd,
                                    r' cost of asking, $c_q$',
                                    'Participation', ['Questions', 'Answers', 'Answerers'],
                                    [r'Expected questions, $F^{EQ}(\mathcal{D}^{-1}(d^*))N^{EQ}$',
                                     r'Expected answers, $\mathbb{E}_{F^{EQ}}[\sum_{k} y^{EQ}(k) | d^*]$',
                                     r'Expected answerers, $(1-F^{EQ}(\tilde{k}))N^{EQ}$'],
                                    'participation_with_c_q' + suffix)
    plotting_function.plot_with_c_q([question_threshold_with_c_q, answer_threshold_with_c_q],
                                    r_q_vector_reqd, c_q_vector_reqd,
                                    r' cost of asking, $c_q$',
                                    'Equilibrium thresholds',
                                    ['asking_threshold_k', 'answering_thres'],
                                    [r'Asking threshold, $\mathcal{D}^{-1}(d^*)$', r'Answering threshold, $\tilde{k}$'],
                                    'thresholds_with_c_q' + suffix)
    plotting_function.plot_with_c_q([proportion_asking, prop_answerers_with_c_q],
                                    r_q_vector_reqd,
                                    c_q_vector_reqd,
                                    r' cost of asking, $c_q$',
                                    'Fraction of Users', ['asking', 'answering'],
                                    [r'Fraction asking, $F^{EQ}(\mathcal{D}^{-1}(d^*))$',
                                     r'Fraction answering, $(1-F^{EQ}(\tilde{k}))$'],
                                    'proportion_participation_with_c_q' + suffix)
    plotting_function.plot_with_c_q([answer_per_q], r_q_vector_reqd, c_q_vector_reqd,
                                    r' cost of asking, $c_q$',
                                    r'Answers per question', ['answers_per_q'], [r'Expected answers per question'],
                                    'answers_per_question_with_c_q' + suffix)


def generate_plots_with_time(output_data, c_q_vector_for_plots, r_q_vector_reqd, num_iter, T, suffix):
    [num_questions_final, total_num_answers_final, num_unanswered_questions_final,
     question_threshold_final, answer_threshold_final, alpha_final, beta_final,
     k_array_final, mean_final, num_users_final, answer_to_user_final,
     leaving_k_list_disappointed_final, joining_arrays_modified, leaving_k_list_random_final,
     proportion_unanswered_dict_in_dict, num_left, num_joined, num_answerers,
     proportion_answerers] = output_data

    x_axis_time = list(range(T))
    plotting_function.get_plots_of_two_scalars(alpha_final, beta_final, c_q_vector_for_plots, r_q_vector_reqd,
                                               x_axis_time, r'time (t)', 'parameters', r'$\alpha$', r'$\beta$',
                                               'alpha_beta_with_time' + suffix)
    plotting_function.get_plots_of_two_scalars(question_threshold_final, answer_threshold_final,
                                               c_q_vector_for_plots, r_q_vector_reqd,
                                               x_axis_time, r'time (t)', 'Thresholds',
                                               r'$d^*$ ', r'$k^*$', 'asking_answering_threshold_with_time' + suffix)
    plotting_function.get_plots_of_a_scalar(mean_final, c_q_vector_for_plots, r_q_vector_reqd, x_axis_time,
                                            r'time (t)', 'Mean knowledge', 'mean_k_with_time' + suffix)
    plotting_function.get_plots_of_a_scalar(total_num_answers_final, c_q_vector_for_plots, r_q_vector_reqd, x_axis_time,
                                            r'time (t)', r'Answers, $\mathbb{E}_{F^{t}}[\sum_{k} y^{t}(k) | d^*]$',
                                            'total_answers_with_time' + suffix)
    plotting_function.get_plots_of_a_scalar(num_questions_final, c_q_vector_for_plots, r_q_vector_reqd,
                                            x_axis_time, r'time (t)',
                                            r'Questions, $F^{t}(\mathcal{D}^{-1}(d^*))N^{t}$',
                                            'total_questions_with_time' + suffix)  # number of users is after joining
    plotting_function.get_plots_of_a_scalar(num_users_final, c_q_vector_for_plots, r_q_vector_reqd,
                                            x_axis_time, r'time (t)', r'Number of users, $N^{t}$',
                                            'num_users_with_time' + suffix)  # number of users is afer joining
    plotting_function.get_plots_of_a_scalar(num_joined, c_q_vector_for_plots, r_q_vector_reqd,
                                            x_axis_time, r'time (t)', r'Number Joining, $N_{join}^t$',
                                            'num_joining_with_time' + suffix)
    plotting_function.get_plots_of_a_scalar(num_left, c_q_vector_for_plots, r_q_vector_reqd,
                                            x_axis_time, r'time (t)', r'Number Leaving, $N_{leave}^t$',
                                            "num_leaving_with_time")
    plotting_function.get_plots_of_a_scalar(proportion_unanswered_dict_in_dict, c_q_vector_for_plots, r_q_vector_reqd,
                                            x_axis_time,
                                            r'time (t)', 'Fraction unanswered',
                                            'proportion_unanswered_with_time' + suffix)

    plotting_function.plot_array_cdf_with_time(leaving_k_list_disappointed_final, c_q_vector_for_plots, r_q_vector_reqd,
                                               num_iter, T, r'knowledge (k)', r'Leaving users, $F_{leave}^t(k)$',
                                               'CDF leaving users' + suffix)
    plotting_function.plot_array_cdf_with_time(joining_arrays_modified, c_q_vector_for_plots, r_q_vector_reqd,
                                               num_iter, T, r'knowledge (k)', r'Joining users, $F_{join}^t(k)$',
                                               'CDF joining users' + suffix)
    plotting_function.plot_array_cdf_with_time(k_array_final, c_q_vector_for_plots, r_q_vector_reqd,
                                               num_iter, T, r'knowledge (k)', r'Community users, $F^t(k)$',
                                               'CDF knowledge' + suffix)

    # get knowledge array in stationary eq ############################
    plotting_function.plot_array_cdf_in_eq(k_array_final, c_q_vector_for_plots, r_q_vector_reqd,
                                           num_iter, r'knowledge (k)', r'Community users, $F^{EQ}(k)$',
                                           'CDF_k_in_eq' + suffix)

    # get leaving array in stationary eq ############################
    plotting_function.plot_array_cdf_in_eq(leaving_k_list_disappointed_final, c_q_vector_for_plots, r_q_vector_reqd,
                                           num_iter, r'knowledge (k)', r'Leaving users, $F_{leave}^{EQ}(k)$',
                                           'CDF_k_leaving_in_eq' + suffix)

    # get joining array in stationary eq ############################
    plotting_function.plot_array_cdf_in_eq(joining_arrays_modified, c_q_vector_for_plots, r_q_vector_reqd,
                                           num_iter, r'knowledge (k)', r'Joining users, $F_{join}^{EQ}(k)$',
                                           'CDF_k_joining_in_eq' + suffix)

    plotting_function.plot_array_pdf_with_time(leaving_k_list_disappointed_final, c_q_vector_for_plots,
                                               r_q_vector_reqd,
                                               num_iter, T, r'knowledge (k)', r'Leaving users, $f_{leave}^t(k)$',
                                               'PDF leaving users' + suffix)
    plotting_function.plot_array_pdf_with_time(joining_arrays_modified, c_q_vector_for_plots,
                                               r_q_vector_reqd,
                                               num_iter, T, r'knowledge (k)', r'Joining users, $f_{join}^t(k)$',
                                               'PDF joining users' + suffix)
    plotting_function.plot_array_pdf_with_time(k_array_final, c_q_vector_for_plots,
                                               r_q_vector_reqd,
                                               num_iter, T, r'knowledge (k)', r'Community users, $f^t(k)$',
                                               'PDF knowledge' + suffix)

    # get knowledge array in stationary eq ############################
    plotting_function.plot_array_pdf_in_eq(k_array_final, c_q_vector_for_plots, r_q_vector_reqd,
                                           num_iter, r'knowledge (k)', r'Community users, $f^{EQ}(k)$',
                                           'PDF_k_in_eq' + suffix)

    # get leaving array in stationary eq ############################
    plotting_function.plot_array_pdf_in_eq(leaving_k_list_disappointed_final, c_q_vector_for_plots, r_q_vector_reqd,
                                           num_iter, r'knowledge (k)', r'Leaving users, $f_{leave}^{EQ}(k)$',
                                           'PDF_k_leaving_in_eq' + suffix)

    # get joining array in stationary eq ############################
    plotting_function.plot_array_pdf_in_eq(joining_arrays_modified, c_q_vector_for_plots, r_q_vector_reqd,
                                           num_iter, r'knowledge (k)', r'Joining users, $f_{join}^{EQ}(k)$',
                                           'PDF_k_joining_in_eq' + suffix)
