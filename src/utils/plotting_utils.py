"""
plotting_utils.py

Contains the function to plot and save figures for
- Score evolution
- Kappa
- Recall
- LLM-based metric
"""

import os
import matplotlib.pyplot as plt

def plot_and_save_figures(score_list, kappa_grade_list, recall_grade_list, llm_based_grade_list, figures_dir):
    """
    Plots each of the 4 metrics over time and saves them as .png in figures_dir.
    """

    # 1) Our custom score
    plt.plot(score_list, marker='o')
    plt.title('Scores Over Time')
    plt.xlabel('Prompt Number')
    plt.ylabel('Score')
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, "score_grade_evolution.png"))
    plt.show()

    # 2) Kappa
    plt.plot(kappa_grade_list, marker='o')
    plt.title('Kappa Score Over Time')
    plt.xlabel('Prompt Number')
    plt.ylabel('Kappa')
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, "kappa_grade_evolution.png"))
    plt.show()

    # 3) Recall
    plt.plot(recall_grade_list, marker='o')
    plt.title('Recall Over Time')
    plt.xlabel('Prompt Number')
    plt.ylabel('Recall')
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, "recall_grade_evolution.png"))
    plt.show()

    # 4) LLM-based
    plt.plot(llm_based_grade_list, marker='o')
    plt.title('LLM-Based Score Over Time')
    plt.xlabel('Prompt Number')
    plt.ylabel('LLM-Based Score')
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, "llm_based_grade_evolution.png"))
    plt.show()
