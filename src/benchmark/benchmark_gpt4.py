"""
benchmark_gpt4.py

Benchmarks the GPT-4 model (OpenAI) for the radiotherapy summarization task.
"""

import os
import time
import csv
import openai

from ..utils.prompt_utils import random_prompt_creation
from ..utils.scoring_utils import define_scores_dict, answers_to_score_kappa
from ..utils.evaluation_utils import (
    scores_properties,
    compute_metric,
    LLM_evaluator,
    right_number_treatment,
    right_free_answer,
    kappa_cohen_index,
    keywords_in_output,
    keywords_not_in_output,
    recall,
    question_to_keywords,
    keywords
)
from ..utils.plotting_utils import plot_and_save_figures

score_list = []
recall_grade_list = []
kappa_grade_list = []
llm_based_grade_list = []

def main(prompt, scores, QA, answers, prompt_filename, output_filename, csv_writer, prompt_number):
    openai.api_key = os.getenv("OPENAI_API_KEY", "")

    start_time = time.time()
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7,
            n=1
        )
        output_text = response.choices[0].message['content']
        elapsed_time = time.time() - start_time

        with open(output_filename, "w", encoding='utf-8') as file:
            try:
                keywords_dict, symptoms, daily_activities = scores_properties()
                grade = compute_metric(scores, keywords_dict, output_text, daily_activities, symptoms)
                llm_grade = LLM_evaluator(QA, output_text)

                number_treat = right_number_treatment(output_text, answers)
                free_ans = right_free_answer(output_text, answers)
                score_dict = define_scores_dict()
                answers_score = answers_to_score_kappa(answers, score_dict)
                kw_to_check = keywords_in_output(answers_score, question_to_keywords)
                not_kw_to_check = keywords_not_in_output(keywords, kw_to_check)

                kappa = kappa_cohen_index(output_text, kw_to_check, not_kw_to_check, number_treat, free_ans)
                recall_grade_ = recall(kw_to_check, number_treat, free_ans, output_text)

                score_list.append(grade)
                recall_grade_list.append(recall_grade_)
                kappa_grade_list.append(kappa)
                llm_based_grade_list.append(llm_grade)

                csv_writer.writerow([prompt_number, grade, kappa, recall_grade_, llm_grade])
                file.write(output_text)

            except UnicodeDecodeError:
                file.write(output_text)

        print(f"Time taken: {elapsed_time:.2f} seconds")

    except Exception as e:
        print("Unexpected error with GPT-4 API:", e)


if __name__ == "__main__":
    base_dir = "saved_files/Benchmark_GPT-4"
    prompt_dir = os.path.join(base_dir, "prompt")
    output_dir = os.path.join(base_dir, "output")
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(prompt_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    csv_filename = os.path.join(base_dir, "benchmark.csv")

    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Prompt number', 'Grade Score', 'Kappa Score', 'Recall Score', 'LLM Based Score'])

        X = 5
        for i in range(X):
            print(f"Prompt number {i+1} / {X}")
            prompt, scores, QA, answers = random_prompt_creation()

            prompt_filename = os.path.join(prompt_dir, f"prompt_GPT4_{i}.txt")
            with open(prompt_filename, "w", encoding='utf-8') as file:
                file.write(prompt)
                file.write(str(scores))

            output_filename = os.path.join(output_dir, f"output_GPT4_{i}.txt")
            main(prompt, scores, QA, answers, prompt_filename, output_filename, csv_writer, i+1)

    mean_grades = sum(score_list)/len(score_list) if score_list else 0
    mean_kappa = sum(kappa_grade_list)/len(kappa_grade_list) if kappa_grade_list else 0
    mean_recall_ = sum(recall_grade_list)/len(recall_grade_list) if recall_grade_list else 0
    mean_llm_ = sum(llm_based_grade_list)/len(llm_based_grade_list) if llm_based_grade_list else 0

    print(f"The mean grade: {mean_grades:.3f}")
    print(f"The mean kappa: {mean_kappa:.3f}")
    print(f"The mean recall: {mean_recall_:.3f}")
    print(f"The mean LLM-based metric: {mean_llm_:.3f}")

    plot_and_save_figures(score_list, kappa_grade_list, recall_grade_list, llm_based_grade_list, figures_dir)
