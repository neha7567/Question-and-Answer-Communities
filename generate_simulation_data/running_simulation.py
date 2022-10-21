import numpy as np
import scipy.stats as stats
import main_file_with_parameters as main_file
from copy import copy
import utility
import finding_thresholds
import finding_joining_array
import knowledge_update_functions
import find_answers_to_every_question

delta = main_file.init_delta
max_iteration = main_file.init_max_iter_for_internal_sim
num_iter = main_file.init_num_iter


def simulate_forum_for_given_reward(reward_ans, reward_ques, cost_q, t_users, alpha_hardness,
                                    num_chosen, t_max, distribution_parameters):
    prob_l, init_dist_parameters = main_file.init_prob_leaving, copy(distribution_parameters)

    # initialize the system ############################################################

    init_num_users = stats.binom.rvs(t_users, .1, random_state=2)
    outside_k_array = utility.generate_initial_knowledge_users(t_users, init_dist_parameters, random_state_input=12)
    knowledge_array = np.linspace(0, 1, init_num_users)

    # result lists initializer #########################################################

    num_questions_list, total_num_answers_list, num_unanswered_questions_list, \
    question_threshold_list, answer_threshold_list, alpha_dist_list, beta_dist_list, \
    knowledge_array_list, mean_k_list, num_users_list, answer_to_user_list, \
    leaving_k_list_disappointed, leaving_k_list_random, joining_array_list = [], [], [], [], \
                                                                             [], [], [], [], [], \
                                                                             [], [], [], [], []
    joining_array_list.append(outside_k_array)
    knowledge_array_list.append(knowledge_array)

    t = 0
    while t < t_max:
        t += 1

        # joining decision of users and new knowledge array ##############
        joining_array = finding_joining_array.get_joining_array(knowledge_array, joining_array_list[-1],
                                                                reward_ans, reward_ques, cost_q,
                                                                alpha_hardness, num_chosen, outside_k_array)

        joining_array_list.append(joining_array)

        knowledge_array = np.append(knowledge_array, joining_array)  # update knowledge array after joining
        print('mean_after_joining=', np.mean(knowledge_array), 'variance=', np.var(knowledge_array))
        # storing_initial_values ############################################################
        asking_threshold, answering_threshold_init = finding_thresholds.find_asking_threshold(knowledge_array,
                                                                                              reward_ans, reward_ques,
                                                                                              cost_q, alpha_hardness,
                                                                                              num_chosen)
        print('thresholds_after_joining=', asking_threshold, answering_threshold_init)
        # generate difficulty questions
        difficulty_questions = utility.generate_questions_hardness(knowledge_array, asking_threshold, alpha_hardness)

        # now on observing questions users find their answering threshold
        answers_to_each_user, answering_threshold = \
            find_answers_to_every_question.generate_number_answers_and_ans_threshold(difficulty_questions,
                                                                                     asking_threshold, alpha_hardness,
                                                                                     reward_ans, num_chosen,
                                                                                     knowledge_array)
        # store values #########
        num_questions_list.append(len(difficulty_questions.keys()))
        question_threshold_list.append(asking_threshold / alpha_hardness)
        answer_threshold_list.append(answering_threshold), alpha_dist_list.append(distribution_parameters[0])
        beta_dist_list.append(distribution_parameters[1]), mean_k_list.append(np.mean(knowledge_array))
        num_users = len(knowledge_array)
        num_users_list.append(num_users), answer_to_user_list.append(answers_to_each_user)  # store this dictionary

        # unanswered leave and answered gain knowledge ##################################
        leaving_arr, knowledge_array = \
            knowledge_update_functions.get_k_leaving_array_disappointed_and_update_knowledge_gain(knowledge_array,
                                                                                                  answers_to_each_user,
                                                                                                  delta)
        random_leaving_arr, knowledge_array = \
            knowledge_update_functions.get_k_leaving_indices_random(knowledge_array, prob_l, max_iteration)

        knowledge_array_list.append(knowledge_array)

        leaving_k_list_disappointed.append(leaving_arr)  # store this value
        leaving_k_list_random.append(random_leaving_arr)
        total_num_answers_list.append(sum(answers_to_each_user.values()))
        num_unanswered_questions_list.append(sum(value < 1 for value in answers_to_each_user.values()))

        distribution_parameters = knowledge_update_functions.update_dist_parameters(
            knowledge_array)  # this gets stored at start of loop

        print("for t=", t, "num_joined=", len(joining_array), "num_left_unanswered=", len(leaving_arr),
              'num_left_random=', len(random_leaving_arr), "users_in_system=", num_users)

    return num_questions_list, total_num_answers_list, num_unanswered_questions_list, question_threshold_list, \
        answer_threshold_list, alpha_dist_list, beta_dist_list, knowledge_array_list, mean_k_list, num_users_list, \
        answer_to_user_list, leaving_k_list_disappointed, joining_array_list, leaving_k_list_random


def get_all_sample_paths_with_c_q(reward_ans, reward_ques, cost_q, t_users, alpha_hardness,
                                  num_chosen, t_max, distribution_parameters):
    num = 0
    num_questions_dict, total_num_answers_dict, num_unanswered_questions_dict, question_threshold_dict, \
        answer_threshold_dict, alpha_dist_dict, beta_dist_dict, knowledge_array_dict, mean_k_dict, num_users_dict, \
        answer_to_user_dict, leaving_k_list_disappointed_dict, joining_array_dict, leaving_k_list_random_dict = \
        {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

    while num < num_iter:
        num_questions_dict[(cost_q, reward_ques, num)], total_num_answers_dict[(cost_q, reward_ques, num)], \
            num_unanswered_questions_dict[(cost_q, reward_ques, num)], \
            question_threshold_dict[(cost_q, reward_ques, num)], answer_threshold_dict[(cost_q, reward_ques, num)], \
            alpha_dist_dict[(cost_q, reward_ques, num)], beta_dist_dict[(cost_q, reward_ques, num)], \
            knowledge_array_dict[(cost_q, reward_ques, num)], mean_k_dict[(cost_q, reward_ques, num)], \
            num_users_dict[(cost_q, reward_ques, num)], answer_to_user_dict[(cost_q, reward_ques, num)], \
            leaving_k_list_disappointed_dict[(cost_q, reward_ques, num)], \
            joining_array_dict[(cost_q, reward_ques, num)], leaving_k_list_random_dict[(cost_q, reward_ques, num)] = \
            simulate_forum_for_given_reward(reward_ans, reward_ques, cost_q, t_users, alpha_hardness,
                                            num_chosen, t_max, distribution_parameters)
        num += 1

    return [num_questions_dict, total_num_answers_dict, num_unanswered_questions_dict, question_threshold_dict,
            answer_threshold_dict, alpha_dist_dict, beta_dist_dict, knowledge_array_dict, mean_k_dict, num_users_dict,
            answer_to_user_dict, leaving_k_list_disappointed_dict, joining_array_dict, leaving_k_list_random_dict]


def get_all_sample_paths_with_c_q_parallel(reward_ans, reward_ques, cost_q, t_users, alpha_hardness,
                                           num_chosen, t_max, distribution_parameters):
    num = 0
    num_questions_dict, total_num_answers_dict, num_unanswered_questions_dict, question_threshold_dict, \
        answer_threshold_dict, alpha_dist_dict, beta_dist_dict, knowledge_array_dict, mean_k_dict, num_users_dict, \
        answer_to_user_dict, leaving_k_list_disappointed_dict, joining_array_dict, leaving_k_list_random_dict = \
        {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

    while num < num_iter:
        num_questions_dict[(cost_q, reward_ques, num)], total_num_answers_dict[(cost_q, reward_ques, num)], \
            num_unanswered_questions_dict[(cost_q, reward_ques, num)], \
            question_threshold_dict[(cost_q, reward_ques, num)], answer_threshold_dict[(cost_q, reward_ques, num)], \
            alpha_dist_dict[(cost_q, reward_ques, num)], beta_dist_dict[(cost_q, reward_ques, num)], \
            knowledge_array_dict[(cost_q, reward_ques, num)], mean_k_dict[(cost_q, reward_ques, num)], \
            num_users_dict[(cost_q, reward_ques, num)], answer_to_user_dict[(cost_q, reward_ques, num)], \
            leaving_k_list_disappointed_dict[(cost_q, reward_ques, num)], \
            joining_array_dict[(cost_q, reward_ques, num)], leaving_k_list_random_dict[(cost_q, reward_ques, num)] = \
            simulate_forum_for_given_reward(reward_ans, reward_ques, cost_q, t_users, alpha_hardness,
                                            num_chosen, t_max, distribution_parameters)
        num += 1

    return [cost_q, reward_ques, num_questions_dict, total_num_answers_dict, num_unanswered_questions_dict,
            question_threshold_dict, answer_threshold_dict, alpha_dist_dict, beta_dist_dict, knowledge_array_dict,
            mean_k_dict, num_users_dict, answer_to_user_dict, leaving_k_list_disappointed_dict, joining_array_dict,
            leaving_k_list_random_dict]
