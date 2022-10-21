import scipy.stats as stats
import main_file_with_parameters as main_file


def generate_initial_knowledge_users(n, dist_parameters, random_state_input):
    k_array = stats.beta.rvs(a=dist_parameters[0],
                             b=dist_parameters[1],
                             size=n, random_state=random_state_input)
    return k_array


def generate_questions_hardness(k_array, hardness_thresh, alpha):
    difficulty_questions = {i: alpha * i for i in k_array if
                            i <= hardness_thresh / alpha}  # k is key and difficulty level is value
    return difficulty_questions


def cost_answering(y, max_cost=main_file.init_cost_ans_max):
    return max_cost - y


