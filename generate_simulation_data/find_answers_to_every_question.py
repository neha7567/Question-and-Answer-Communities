import numpy as np
import finding_thresholds
import scipy.stats as stats


def generate_number_answers_and_ans_threshold(difficulty_questions, ask_threshold, alpha, reward_a, beta, k_array):
    q_array = np.array(list(difficulty_questions.keys()))  # knowledge level of askers

    if finding_thresholds.answering_threshold_equation(reward_a, ask_threshold, alpha, beta, k_array, 0) >= 0:
        threshold_answers = 0
    else:
        k_range = np.linspace(0, 1, num=300)
        ans_util = np.zeros(len(k_range))
        for j in range(len(k_range)):
            ans_util[j] = finding_thresholds.answering_threshold_equation(reward_a, ask_threshold, alpha, beta,
                                                                          k_array, k_range[j])
        threshold_answers = k_range[np.argmax(ans_util > 0)]

    answering_arr = k_array[k_array >= threshold_answers]

    answers_to_user = {i: 0 for i in difficulty_questions.keys()}  # answer to user of knowledge i
    for k in answering_arr:
        # picks beta questions that he can solve
        if len(q_array) > 1:
            indices_picked = stats.randint.rvs(0, len(q_array), size=min(beta, len(q_array)))
            for index_q in indices_picked:
                if difficulty_questions[q_array[index_q]] <= k:
                    answers_to_user[q_array[index_q]] += 1

    return answers_to_user, threshold_answers
