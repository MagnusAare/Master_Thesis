import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from Functions_thesis import *


def plot_subplots_lagrangian(c_spo, c_lr, c_test, LT, figsize=(20, 10)):
    c_list = ['C0', 'C1', 'C2', 'C3', 'C4']

    fig, axs = plt.subplots(2, 3, figsize=figsize)

    titles = ["C-vector", "$\\lambda - \\mu$", "$\\mu$",
              "$\\lambda$", "Zero vector", "Pw"]
    data_ranges = [
        (0, len(c_spo)),
        (0, LT),
        (LT, 2*LT),
        (2*LT, 3*LT),
        (3*LT, 4*LT),
        (4*LT, 5*LT)
    ]

    # Lines and their respective labels
    lines_labels = [
        (c_spo, "SPO", c_list[0]),
        (c_lr, "Lin. Reg.", c_list[1]),
        (c_test, "Target", c_list[2])
    ]

    for i, ax in enumerate(axs.flatten()):
        start, end = data_ranges[i]
        ax.set_title(titles[i])

        for data, label, color in lines_labels:
            if i == 5:  # Special case for "PW" plot, different data processing
                ax.plot(c_spo[start:end] / (-1 * c_spo[LT:2*LT]),
                        label=label, color=c_list[0])
                ax.plot(c_lr[start:end] / (-1 * c_lr[LT:2*LT]),
                        label=label, color=c_list[1])
                ax.plot(c_test[start:end] / (-1 * c_test[LT:2*LT]),
                        label=label, color=c_list[2])
            else:
                ax.plot(data[start:end], label=label, color=color)

        if i == 1 or i == 4:  # For "Lambda - mu" and "Zero" plots
            ax.plot(np.zeros(LT), linestyle="--",
                    label="Theoretical Target", color=c_list[3])
        elif i == 5:  # For "PW" plot
            ax.plot(np.zeros(LT), linestyle='--',
                    label="Feasibility Limit", color=c_list[4])

        if i != 0:  # Add x-ticks to all plots except the first one
            ax.set_xticks(np.arange(0, LT, 2))

        ax.grid(alpha=0.4)

    # Extract all lines and labels from the subplots
    lines, labels = [], []
    for ax in axs.flatten():
        for line in ax.get_lines():
            if line.get_label() not in labels:
                lines.append(line)
                labels.append(line.get_label())

    fig.legend(lines, labels, loc="upper center", bbox_to_anchor=(
        0.5, 0.1), fancybox=True, ncol=len(c_list))
    plt.show()


def plot_subplots_lagrangian2(results_list, names_list, LT, figsize=(20, 10)):
    c_list = [f'C{i}' for i in range(len(results_list))]

    fig, axs = plt.subplots(2, 3, figsize=figsize)

    titles = ["C-vector", "$\\lambda - \\mu$", "$\\mu$",
              "$\\lambda$", "Zero vector", "Pw"]
    data_ranges = [
        (0, len(results_list[0])),
        (0, LT),
        (LT, 2*LT),
        (2*LT, 3*LT),
        (3*LT, 4*LT),
        (4*LT, 5*LT)
    ]

    for i, ax in enumerate(axs.flatten()):
        start, end = data_ranges[i]
        ax.set_title(titles[i])

        for j, (data, label) in enumerate(zip(results_list, names_list)):
            if i == 5:  # Special case for "PW" plot, different data processing
                ax.plot(results_list[0][start:end] / (-1 * results_list[0][LT:2*LT]),
                        label=names_list[0], color=c_list[0])
                ax.plot(results_list[1][start:end] / (-1 * results_list[1][LT:2*LT]),
                        label=names_list[1], color=c_list[1])
                ax.plot(results_list[2][start:end] / (-1 * results_list[2][LT:2*LT]),
                        label=names_list[2], color=c_list[2])
            else:
                ax.plot(data[start:end], label=label, color=c_list[j])

        if i == 1 or i == 4:  # For "Lambda - mu" and "Zero" plots
            ax.plot(np.zeros(LT), linestyle="--",
                    label="Theoretical Target", color='grey')
        elif i == 5:  # For "PW" plot
            ax.plot(np.zeros(LT), linestyle='--',
                    label="Feasibility Limit", color='black')

        if i != 0:  # Add x-ticks to all plots except the first one
            ax.set_xticks(np.arange(0, LT, 2))

        ax.grid(alpha=0.4)

    # Extract all lines and labels from the subplots
    lines, labels = [], []
    for ax in axs.flatten():
        for line in ax.get_lines():
            if line.get_label() not in labels:
                lines.append(line)
                labels.append(line.get_label())

    fig.legend(lines, labels, loc="upper center", bbox_to_anchor=(
        0.5, 0.1), fancybox=True, ncol=len(results_list))
    plt.show()


def plot_subplots_dual(c_spo, c_lr, c_test, LT, figsize=(20, 10)):

    c_list = ['C0', 'C1', 'C2', 'C3', 'C4']

    fig, axs = plt.subplots(2, 3, figsize=figsize)

    titles = ["C-vector", "Pw", "P^Ch/P^Dis",
              "$P^Ch/P^Dis", "SOC", "SOC_init"]

    data_ranges = [
        (0, len(c_spo)),
        (0, LT+1),
        (LT+1, 2*LT+1),
        (2*LT+1, 3*LT+1),
        (3*LT+1, 4*LT+1),
        (4*LT+1, 5*LT+1)
    ]

    lines_labels = [
        (c_spo, "SPO", c_list[0]),
        (c_lr, "Lin. Reg.", c_list[1]),
        (c_test, "Target", c_list[2])
    ]

    for i, ax in enumerate(axs.flatten()):
        start, end = data_ranges[i]
        ax.set_title(titles[i])

        for data, label, color in lines_labels:
            ax.plot(data[start:end], label=label, color=color)
        if i == 1 or i == 4:  # For "Lambda - mu" and "Zero" plots
            ax.plot(np.zeros(LT), linestyle="--",
                    label="Feasibility limit", color=c_list[4])

        if i != 0:  # Add x-ticks to all plots except the first one
            ax.set_xticks(np.arange(0, LT, 2))

        ax.grid(alpha=0.4)

    # Extract all lines and labels from the subplots
    lines, labels = [], []
    for ax in axs.flatten():
        for line in ax.get_lines():
            if line.get_label() not in labels:
                lines.append(line)
                labels.append(line.get_label())

    fig.legend(lines, labels, loc="upper center", bbox_to_anchor=(
        0.5, 0.1), fancybox=True, ncol=len(c_list))
    plt.show()


def plot_subplots_dual2(results_list, names_list, LT, figsize=(20, 10)):
    c_list = [f'C{i}' for i in range(len(results_list))]

    fig, axs = plt.subplots(2, 3, figsize=figsize)

    titles = ["C-vector", "Pw", "P^Ch/P^Dis",
              "$P^Ch/P^Dis$", "SOC", "SOC_init"]

    data_ranges = [
        (0, len(results_list[0])),
        (0, LT+1),
        (LT+1, 2*LT+1),
        (2*LT+1, 3*LT+1),
        (3*LT+1, 4*LT+1),
        (4*LT+1, 5*LT+1)
    ]

    for i, ax in enumerate(axs.flatten()):
        start, end = data_ranges[i]
        ax.set_title(titles[i])

        for j, (data, label) in enumerate(zip(results_list, names_list)):
            ax.plot(data[start:end], label=label, color=c_list[j])

        if i == 1 or i == 4:  # For "Lambda - mu" and "Zero" plots
            ax.plot(np.zeros(LT), linestyle="--",
                    label="Feasibility limit", color='grey')

        if i != 0:  # Add x-ticks to all plots except the first one
            ax.set_xticks(np.arange(0, LT, 2))

        ax.grid(alpha=0.4)

    # Extract all lines and labels from the subplots
    lines, labels = [], []
    for ax in axs.flatten():
        for line in ax.get_lines():
            if line.get_label() not in labels:
                lines.append(line)
                labels.append(line.get_label())

    fig.legend(lines, labels, loc="upper center", bbox_to_anchor=(
        0.5, 0.1), fancybox=True, ncol=len(results_list))
    plt.show()


def plot_subplots_equality(c_spo, c_lr, c_test, LT, figsize=(20, 10)):
    c_list = ['C0', 'C1', 'C2', 'C3', 'C4']

    fig, axs = plt.subplots(2, 3, figsize=figsize)

    titles = ["C-vector", "$\\lambda$", "$-\\lambda$",
              "$-\\lambda * PW$", "0", "PW"]
    data_ranges = [
        (0, len(c_spo)),
        (0, LT),
        (LT, 2*LT),
        (2*LT, 3*LT),
        (3*LT, 4*LT),
        (2*LT, 3*LT)]

    # Lines and their respective labels
    lines_labels = [
        (c_spo, "SPO", c_list[0]),
        (c_lr, "Lin. Reg.", c_list[1]),
        (c_test, "Target", c_list[2])
    ]

    for i, ax in enumerate(axs.flatten()):
        start, end = data_ranges[i]
        ax.set_title(titles[i])

        for data, label, color in lines_labels:
            if i == 5:  # Special case for "PW" plot, different data processing
                ax.plot(c_spo[start:end] / (c_spo[LT:2*LT]),
                        label=label, color=c_list[0])
                ax.plot(c_lr[start:end] / (c_lr[LT:2*LT]),
                        label=label, color=c_list[1])
                ax.plot(c_test[start:end] / (c_test[LT:2*LT]),
                        label=label, color=c_list[2])
            else:
                ax.plot(data[start:end], label=label, color=color)

        if i == 4:  # For "Lambda - mu" and "Zero" plots
            ax.plot(np.zeros(LT), linestyle="--",
                    label="Theoretical Target", color=c_list[3])
        elif i == 5:  # For "PW" plot
            ax.plot(np.zeros(LT), linestyle='--',
                    label="Feasibility Limit", color=c_list[4])

        if i != 0:  # Add x-ticks to all plots except the first one
            ax.set_xticks(np.arange(0, LT, 2))

        ax.grid(alpha=0.4)

    # Extract all lines and labels from the subplots
    lines, labels = [], []
    for ax in axs.flatten():
        for line in ax.get_lines():
            if line.get_label() not in labels:
                lines.append(line)
                labels.append(line.get_label())

    fig.legend(lines, labels, loc="upper center", bbox_to_anchor=(
        0.5, 0.1), fancybox=True, ncol=len(c_list))
    plt.show()


def plot_subplots_equality2(results_list, names_list, LT, figsize=(20, 10)):
    c_list = [f'C{i}' for i in range(len(results_list))]

    fig, axs = plt.subplots(2, 3, figsize=figsize)

    titles = ["C-vector", "$\\lambda$", "$-\\lambda$",
              "$-\\lambda * PW$", "0", "PW"]
    data_ranges = [
        (0, len(results_list[0])),
        (0, LT),
        (LT, 2*LT),
        (2*LT, 3*LT),
        (3*LT, 4*LT),
        (2*LT, 3*LT)]

    for i, ax in enumerate(axs.flatten()):
        start, end = data_ranges[i]
        ax.set_title(titles[i])

        for j, (data, label) in enumerate(zip(results_list, names_list)):
            if i == 5:  # Special case for "PW" plot
                ax.plot(results_list[0][start:end] / results_list[0][LT:2*LT],
                        label=names_list[0], color=c_list[0])
                ax.plot(results_list[1][start:end] / results_list[1][LT:2*LT],
                        label=names_list[1], color=c_list[1])
                ax.plot(results_list[2][start:end] / results_list[2][LT:2*LT],
                        label=names_list[2], color=c_list[2])
            else:
                ax.plot(data[start:end], label=label, color=c_list[j])

        if i == 4:  # For "Lambda - mu" and "Zero" plots
            ax.plot(np.zeros(LT), linestyle="--",
                    label="Theoretical Target", color='grey')
        elif i == 5:  # For "PW" plot
            ax.plot(np.zeros(LT), linestyle='--',
                    label="Feasibility Limit", color='black')

        if i != 0:  # Add x-ticks to all plots except the first one
            ax.set_xticks(np.arange(0, LT, 2))

        ax.grid(alpha=0.4)

    # Extract all lines and labels from the subplots
    lines, labels = [], []
    for ax in axs.flatten():
        for line in ax.get_lines():
            if line.get_label() not in labels:
                lines.append(line)
                labels.append(line.get_label())

    fig.legend(lines, labels, loc="upper center", bbox_to_anchor=(
        0.5, 0.1), fancybox=True, ncol=len(results_list))
    plt.show()


def plot_cluster(df, n_clusters, cluster_target, cluster_center, LT, save_name=None):

    c_list = [f'C{i}' for i in range(n_clusters)]
    plt.figure(figsize=(12, 8))
    for idx in range(n_clusters):
        cluster_df = df[df['Cluster'] == idx]
        plt.plot(cluster_center[idx], color=c_list[idx], label=f"Cluster {idx+1}", linewidth=3, path_effects=[
                 pe.Stroke(linewidth=5, foreground='black'), pe.Normal()])
        for id in cluster_df.ID.unique():
            plt.plot(cluster_df[cluster_df['ID'] == id]
                     [cluster_target].values, color=c_list[idx], alpha=0.3, zorder=0)
    plt.legend()
    plt.grid(alpha=0.4)
    plt.xlabel("Hours")
    plt.ylabel("Value [-]")
    if save_name is not None:
        plt.savefig(f'{save_name}.png', dpi=300,
                    bbox_inches='tight', facecolor='white')
    plt.show()


def plot_cluster_multitarget(df, n_clusters, cluster_targets, cluster_center, LT):

    c_list = [f'C{i}' for i in range(n_clusters)]

    # Combined plot for each target
    for jdx, target in enumerate(cluster_targets):
        plt.figure(figsize=(12, 8))

        for idx in range(n_clusters):
            cluster_df = df[df['Cluster'] == idx]
            cluster_mean = cluster_center[idx].flatten()[jdx*LT:(jdx+1)*LT]

            plt.plot(cluster_mean, color=c_list[idx], label=f"Cluster {idx+1} - {target}",
                     linewidth=3, path_effects=[pe.Stroke(linewidth=5, foreground='black'), pe.Normal()])

            for id in cluster_df.ID.unique():
                plt.plot(cluster_df[cluster_df['ID'] == id][target].values,
                         color=c_list[idx], alpha=0.3, zorder=0)

        plt.legend()
        plt.grid(alpha=0.4)
        plt.xlabel("Hours")
        plt.ylabel(f"{target} [-]")
        plt.title(f"Clustered {target}")
        # if Name is not None:
        #     plt.savefig(f'{Name}.png', dpi=300,
        #                 bbox_inches='tight', facecolor='white')
        plt.show()
