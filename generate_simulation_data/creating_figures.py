import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import gaussian_kde

from finding_thresholds import convert_array_to_cdf_function, convert_array_to_pdf_function

plt.rcParams["ps.usedistiller"] = 'xpdf'
plt.rcParams['axes.linewidth'] = 5
plt.rcParams.update({'axes.titlesize': 'small'})
styles = ['k-.', 'ro-', 'g:', 'b:', 'g--']
colors = ['red', 'blue', 'green', 'm']
markers = ["X", 'D', 'o']


def setup_sns():
    sns.set_palette(sns.color_palette(colors))
    sns.set(font_scale=2.5)


def save_figs(figname):
    my_file_eps = figname + '.eps'
    my_file_png = figname + '.png'
    plt.savefig('figs/{}'.format(my_file_png), bbox_inches="tight")
    plt.savefig('figs/{}'.format(my_file_eps), bbox_inches="tight")
    # plt.show()
    plt.close()


def subplot_label_settings(x, y):
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    # plt.xlabel('%s' % x, fontsize=20)
    # plt.ylabel('%s' % y, fontsize=20)


def get_exploded_data_frames_from_ts_dic_entries(dict_input, time_frame, y):  # time_frame is list of all times
    data = pd.Series(dict_input).rename_axis(["c_q", "r_q", "iter"]).reset_index(name=y)
    num_rows = data.shape[0]
    data['time'] = pd.Series(np.array([time_frame for i in range(num_rows)]).tolist())
    return data.explode(['time', y]).explode([y])


def find_error_vector(column, df_result):
    df_result['mean_value'] = df_result.groupby(by=['c_q', 'r_q'])[column].transform('mean')
    df_result['min_error'] = - df_result.groupby(by=['c_q', 'r_q'])[column].transform('min') + df_result['mean_value']
    df_result['max_error'] = df_result.groupby(by=['c_q', 'r_q'])[column].transform('max') - df_result['mean_value']
    df_result = df_result.drop_duplicates(subset=["c_q", "r_q", "mean_value", "min_error", "max_error"])
    return df_result


def plot_with_c_q(dict_res, reward_vec, cost_vec, x_lab, y_lab, col_lab, legend_text, fig_name):
    plt.rcParams["figure.figsize"] = (20, 10)
    fig, ax = plt.subplots(1, 1)
    n = len(dict_res)
    # generate df for a single dict
    for i in range(n):
        data = pd.DataFrame.from_dict(dict_res[i], orient='index', columns=["c_q", "r_q", "iter", col_lab[i]])
        df = data[(data["r_q"].isin(reward_vec)) & (data["c_q"].isin(cost_vec))]
        df = find_error_vector(col_lab[i], df)

        ax.errorbar(df["c_q"].to_numpy(), df['mean_value'].to_numpy(),
                    yerr=df[['min_error', 'max_error']].to_numpy().transpose(),
                    barsabove=True, capsize=6, elinewidth=3, capthick=5, ecolor='r', fmt='p')
        ax.scatter(df["c_q"].to_numpy(), df['mean_value'].to_numpy(), s=100, c='k', marker=markers[i],
                   label=legend_text[i])
        ax.tick_params(axis='both', which='both', labelsize=30, width=5, length=10)
        #handles, labels = ax.get_legend_handles_labels()
        #handles = [h[0] for h in handles]
        # use them in the legend
        #ax.legend(handles, labels, loc='best', prop={'size': 25})
        """df['mean_value'] = df.groupby(by=['c_q', 'r_q'])[col_lab[i]].transform('mean')
        df.plot(kind='scatter', x='c_q', y=col_lab[i], marker="o", ax=axs, legend=False)
        if n > 1:
            df.plot(kind='scatter', x='c_q', y='mean_value', marker=markers[i],
                    label=legend_text[i], ax=axs)
        else:
            df.plot(kind='scatter', x='c_q', y='mean_value', marker=markers[i], legend=False, ax=axs)"""
    plt.xlabel('%s' % x_lab, fontsize=50)
    plt.ylabel('%s' % y_lab, fontsize=50)
    # xticks size modification
    """if n > 1:
        plt.legend(loc='best', prop={'size': 30})"""
    plt.legend(loc='best', frameon=False, handletextpad=0.1, prop={'size': 35})
    save_figs(fig_name)


def avoid_sns_formatting(ax):
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)


def get_plots_of_two_scalars(dict_1, dict_2, cost_ask_vec, reward_ask_vec, x_axis_time,
                             xlab, y_lab, plot_lab_1, plot_lab_2, fig_name):
    df_1 = get_exploded_data_frames_from_ts_dic_entries(dict_1, x_axis_time, plot_lab_1)
    df_2 = get_exploded_data_frames_from_ts_dic_entries(dict_2, x_axis_time, plot_lab_2)
    data = pd.merge(df_1, df_2, how='inner', on=['c_q', 'r_q', 'iter', 'time'])
    data = pd.melt(data, id_vars=['c_q', 'r_q', "iter", 'time'], value_vars=[plot_lab_1, plot_lab_2])

    m, n = len(reward_ask_vec), len(cost_ask_vec)
    fig, axs = plt.subplots(nrows=m, ncols=n, sharex='all', sharey='all', figsize=(20, 10))
    fig.add_subplot(111, frameon=False)
    fig.supxlabel(xlab, fontsize=30)
    fig.supylabel(y_lab, fontsize=30)

    if m > 1:
        for k in range(m):
            for i in range(n):
                df = data[(data["r_q"] == reward_ask_vec[k]) & (data["c_q"] == cost_ask_vec[i])]
                unique_vars = pd.unique(df.variable)
                for hue_num in range(len(unique_vars)):
                    axs[k, i].scatter(x='time', y="value", data=df[df.variable == unique_vars[hue_num]],
                                      marker=markers[hue_num], label='%s' % unique_vars[hue_num])
                    axs[k, i].set_title(r'$r_q = $ %s, $c_q = $ %s' % (reward_ask_vec[k], round(cost_ask_vec[i], 2)))
                    avoid_sns_formatting(axs[i, k])
        handles, labels = axs[0, 0].get_legend_handles_labels()
    else:
        k = 0
        for i in range(n):
            df = data[(data["r_q"] == reward_ask_vec[k]) & (data["c_q"] == cost_ask_vec[i])]
            unique_vars = pd.unique(df.variable)
            for hue_num in range(len(unique_vars)):
                axs[i].scatter(x='time', y="value", data=df[df.variable == unique_vars[hue_num]],
                               marker=markers[hue_num], label='%s' % unique_vars[hue_num])
                axs[i].set_title(r'$r_q = $ %s, $c_q = $ %s' % (reward_ask_vec[k], round(cost_ask_vec[i], 2)))
                avoid_sns_formatting(axs[i])
        handles, labels = axs[0].get_legend_handles_labels()
    subplot_label_settings(xlab, y_lab)
    plt.legend(handles, labels, loc='best', prop={'size': 30})
    # remove_legends_subplots(axs, m, n)
    save_figs(fig_name)


def remove_legends_subplots(ax, num_r, num_c):
    if num_r > 1:
        for i in range(num_r):
            for j in range(num_c):
                ax[i, j].get_legend().remove()
    else:
        for j in range(num_c):
            ax[j].get_legend().remove()


def get_plots_of_a_scalar(dict_1, cost_ask_vec, reward_ask_vec, x_axis_time, xlab, y_lab, fig_name):
    df_1 = get_exploded_data_frames_from_ts_dic_entries(dict_1, x_axis_time, y_lab)
    df_1 = df_1[df_1['c_q'].isin(cost_ask_vec)]

    m, n = len(reward_ask_vec), len(cost_ask_vec)
    fig, axs = plt.subplots(nrows=m, ncols=n, sharex='all', sharey='all', figsize=(15, 7))
    fig.add_subplot(111, frameon=False)
    fig.supxlabel(xlab, fontsize=20)
    fig.supylabel(y_lab, fontsize=20)

    for k in range(m):
        data = df_1[df_1["r_q"] == reward_ask_vec[k]]
        setup_sns()
        for num in range(len(cost_ask_vec)):
            axs[k].scatter(x='time', y=y_lab, data=data[data.c_q == cost_ask_vec[num]],
                           marker=markers[num],
                           label='%s' % cost_ask_vec[num])
        axs[k].set_title(r'$R_q = $ %s' % (reward_ask_vec[k]))
        avoid_sns_formatting(axs[k])

    handles, labels = axs[0].get_legend_handles_labels()
    subplot_label_settings(xlab, y_lab)
    plt.legend(handles, labels, loc='best', prop={'size': 20}, title=r'$c_q = $')
    # remove_legends_subplots(axs, 1, m)
    save_figs(fig_name)


def plot_array_cdf_with_time(result_dict, cost_ask_vec, reward_ask_vec,
                             num_iter, max_time, xlab, y_lab, fig_name):
    m, n, number_times = len(reward_ask_vec), len(cost_ask_vec), 3
    step_size = int(max_time / number_times)

    fig, axs = plt.subplots(nrows=n, ncols=m, sharex='all', sharey='all', figsize=(15, 7))
    fig.add_subplot(111, frameon=False)
    fig.supxlabel(xlab, fontsize=20)
    fig.supylabel(y_lab, fontsize=20)
    setup_sns()
    k = 0
    for i in range(n):
        list_for_avgs = []
        num_time = 0
        for time in range(0, max_time, step_size):
            for num in range(num_iter):
                cdf, bin_edges = convert_array_to_cdf_function(result_dict[(cost_ask_vec[i],
                                                                            reward_ask_vec[k], num)][time])
                list_for_avgs.append(cdf)

            axs[i].plot(bin_edges, np.mean(np.array(list_for_avgs), axis=0), styles[num_time],
                        linewidth=4,
                        label='%s' % time)
            num_time += 1
        axs[i].set_title(r'$c_q = $%s, $r_q = $%s' % (round(cost_ask_vec[i], 2), reward_ask_vec[k]))
        avoid_sns_formatting(axs[i])

    handles, labels = axs[0].get_legend_handles_labels()
    subplot_label_settings(xlab, y_lab)
    plt.legend(handles, labels, loc='best', prop={'size': 20}, title=r'time, $t = $')
    save_figs(fig_name)


def plot_array_cdf_in_eq(result_dict, cost_ask_vec, reward_ask_vec, num_iter, xlab, y_lab, fig_name):
    m, n = len(reward_ask_vec), len(cost_ask_vec)

    fig, axs = plt.subplots(figsize=(20, 10))
    fig.add_subplot(111, frameon=False)
    fig.supxlabel(xlab, fontsize=30)
    fig.supylabel(y_lab, fontsize=30)
    setup_sns()
    time_length = len(result_dict[(cost_ask_vec[0], reward_ask_vec[0], 0)])
    for k in range(m):
        for i in range(n):
            list_for_avg = []
            for num in range(num_iter):
                for time in range(max(0, time_length - 100), time_length):
                    cdf, bin_edges = convert_array_to_cdf_function(
                        result_dict[(cost_ask_vec[i], reward_ask_vec[k], num)][time])
                    list_for_avg.append(cdf)
            axs.plot(bin_edges, np.mean(np.array(list_for_avg), axis=0), styles[i], linewidth=4,
                     label='%s' % round(cost_ask_vec[i], 2))

        # axs[k].set_title(r'$R_q = $%s' % (reward_ask_vec[k]))
        avoid_sns_formatting(axs)
    subplot_label_settings(xlab, y_lab)
    plt.legend(loc='best', prop={'size': 30}, title=r'$c_q = $')
    save_figs(fig_name)


def plot_array_pdf_with_time(result_dict, cost_ask_vec, reward_ask_vec,
                             num_iter, max_time, xlab, y_lab, fig_name):
    m, n, number_times = len(reward_ask_vec), len(cost_ask_vec), 3
    step_size = int(max_time / number_times)

    fig, axs = plt.subplots(nrows=n, ncols=m, sharex='all', sharey='all', figsize=(20, 10))
    fig.add_subplot(111, frameon=False)
    fig.supxlabel(xlab, fontsize=30)
    fig.supylabel(y_lab, fontsize=30)
    k = 0
    for i in range(n):
        list_for_avgs = []
        num_time = 0
        for time in range(0, max_time, step_size):
            for num in range(num_iter):
                pdf, bin_edges = convert_array_to_pdf_function(result_dict[(cost_ask_vec[i],
                                                                            reward_ask_vec[k], num)][time])
                list_for_avgs.append(pdf)
            axs[i].plot(bin_edges, np.mean(np.array(list_for_avgs), axis=0), styles[num_time],
                        linewidth=4,
                        label='%s' % time)
            num_time += 1
        setup_sns()
        axs[i].set_title(r'$c_q = $%s, $r_q = $%s' % (round(cost_ask_vec[i], 2), reward_ask_vec[k]))
        avoid_sns_formatting(axs[i])

    handles, labels = axs[0].get_legend_handles_labels()
    subplot_label_settings(xlab, y_lab)
    plt.legend(handles, labels, loc='best', prop={'size': 30}, title=r'time, $t = $')
    save_figs(fig_name)


def plot_array_pdf_in_eq(result_dict, cost_ask_vec, reward_ask_vec, num_iter, xlab, y_lab, fig_name):
    m, n = len(reward_ask_vec), len(cost_ask_vec)

    # fig, axs = plt.subplots(nrows=m, ncols=n, sharex='all', sharey='all', figsize=(15, 7))
    fig, axs = plt.subplots(figsize=(20, 10))
    fig.add_subplot(111, frameon=False)
    fig.supxlabel(xlab, fontsize=30)
    fig.supylabel(y_lab, fontsize=30)

    time_length = len(result_dict[(cost_ask_vec[0], reward_ask_vec[0], 0)])
    for k in range(m):
        for i in range(n):
            list_for_avg = []
            for num in range(num_iter):
                for time in range(max(0, time_length - 100), time_length):
                    pdf, bin_edges = convert_array_to_pdf_function(
                        result_dict[(cost_ask_vec[i], reward_ask_vec[k], num)][time])
                    list_for_avg.append(pdf)

            setup_sns()
            axs.plot(bin_edges, np.mean(np.array(list_for_avg), axis=0), styles[i], linewidth=4,
                     label='%s' % round(cost_ask_vec[i], 2))

        # axs[k].set_title(r'$R_q = $%s' % (reward_ask_vec[k]))
        avoid_sns_formatting(axs)
    subplot_label_settings(xlab, y_lab)
    plt.legend(loc='best', prop={'size': 30}, title=r'$c_q = $')
    save_figs(fig_name)
