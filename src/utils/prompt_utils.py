"""
prompt_utils.py

Functions to create random prompts and answers for the radiotherapy
follow-up questionnaire.
"""

import random
import re
from typing import List, Tuple, Dict

from .scoring_utils import define_scores_dict, answers_to_score

def define_keywords():
    """
    Define the keywords for each question based on the one provided in the questionnaire.
    """
    keywords = [
        "",
        "fatigue",
        "fatigue",
        "flatulence",
        "diarrhea",
        "abdominal abdomen pain",
        "abdominal abdomen pain",
        "abdominal abdomen pain",
        "urination pain",
        "urge urges urinate",
        "urge urges urinate",
        "frequent frequency urination",
        "urination urine frequency",
        "urine color coloration",
        "leakage",
        "leakage",
        "skin burns",
        ""
    ]
    return keywords

def random_prompt_creation() -> Tuple[str, Dict[int, float], str, List[str]]:
    """
    Generates a random set of answers, formats them, creates a prompt for summarization,
    and calculates initial scores based on these answers.

    Returns:
        Tuple containing:
            - prompt (str): The formatted prompt for summarization.
            - scores (Dict[int, float]): Dictionary mapping question numbers to scores.
            - QA (str): The question+answer text as a single string.
            - answers_no_tag (List[str]): List of answers (strings) without "A#: " tags.
    """
    questions = [
        "Q1. How many radiation treatments have you had? It’s okay if you don’t know.",
        "Q2. In the last 7 days, what was the SEVERITY of your FATIGUE, TIREDNESS, OR LACK OF ENERGY at its WORST?",
        "Q3. In the last 7 days, how much did FATIGUE, TIREDNESS, OR LACK OF ENERGY INTERFERE with your usual activities?",
        "Q4. In the last 7 days, did you have any INCREASED PASSING OF GAS (FLATULENCE)?",
        "Q5. In the last 7 days, how OFTEN did you have LOOSE OR WATERY STOOLS (DIARRHEA)?",
        "Q6. In the last 7 days, how OFTEN did you have PAIN IN THE ABDOMEN (BELLY AREA)?",
        "Q7. In the last 7 days, what was the SEVERITY of your PAIN IN THE ABDOMEN (BELLY AREA) at its WORST?",
        "Q8. In the last 7 days, how much did PAIN IN THE ABDOMEN (BELLY AREA) INTERFERE with your activities?",
        "Q9. In the last 7 days, what was the SEVERITY of your PAIN OR BURNING WITH URINATION at its WORST?",
        "Q10. In the last 7 days, how OFTEN did you feel an URGE TO URINATE ALL OF A SUDDEN?",
        "Q11. In the last 7 days, how much did SUDDEN URGES TO URINATE INTERFERE with your daily activities?",
        "Q12. In the last 7 days, were there times when you had to URINATE FREQUENTLY?",
        "Q13. In the last 7 days, how much did FREQUENT URINATION INTERFERE with your daily activities?",
        "Q14. In the last 7 days, did you have any URINE COLOR CHANGE?",
        "Q15. In the last 7 days, how OFTEN did you have LOSS OF CONTROL OF URINE (LEAKAGE)?",
        "Q16. In the last 7 days, how much did LOSS OF CONTROL OF URINE (LEAKAGE) INTERFERE with your usual activities?",
        "Q17. In the last 7 days, what was the SEVERITY of your SKIN BURNS FROM RADIATION at their WORST?",
        "Q18. Finally, do you have any other symptoms that you wish to report?"
    ]

    # possible answers for A2 -> A18
    possible_answers = [
        ["None", "Mild", "Moderate", "Severe", "Very Severe"],      # A2
        ["Not at all", "A little bit", "Somewhat", "Quite a bit"],  # A3
        ["Yes", "No"],                                              # A4
        ["Never", "Rarely", "Occasionally", "Frequently", "Almost constantly"],  # A5
        ["Never", "Rarely", "Occasionally", "Frequently", "Almost Constantly"],  # A6
        ["None", "Mild", "Moderate", "Severe", "Very Severe"],      # A7
        ["Not at all", "A little bit", "Somewhat", "Quite a bit", "Very much"],  # A8
        ["None", "Mild", "Moderate", "Severe", "Very Severe"],      # A9
        ["Never", "Rarely", "Occasionally", "Frequently", "Almost Constantly"],  # A10
        ["Not at all", "A little bit", "Somewhat", "Quite a bit", "Very much"],  # A11
        ["Never", "Rarely", "Occasionally", "Frequently", "Almost Constantly"],  # A12
        ["Not at all", "A little bit", "Somewhat", "Quite a bit", "Very much"],  # A13
        ["Yes", "No"],                                              # A14
        ["Never", "Rarely", "Occasionally", "Frequently", "Almost Constantly"],  # A15
        ["Not at all", "A little bit", "Somewhat", "Quite a bit", "Very much"],  # A16
        ["None", "Mild", "Moderate", "Severe", "Very Severe", "Not Applicable"],# A17
        ["No additional symptoms", "Experiencing occasional headaches", "Slight nausea and dizziness"] # A18
    ]

    # A1 random integer between 1 and 38
    nb_treatments = str(random.randint(1, 38))
    answers_no_tag = [nb_treatments]
    answers = [f"A1. {nb_treatments}"]

    # Format the rest of the answers
    formatted_answers = []
    formatted_answers_no_tag = []
    for i, ans_opts in enumerate(possible_answers, start=2):
        random_answer = random.choice(ans_opts)
        formatted_ans_no_tag = f"{random_answer}"
        formatted_ans = f"A{i}. {random_answer}"

        formatted_answers_no_tag.append(formatted_ans_no_tag)
        formatted_answers.append(formatted_ans)

    answers += formatted_answers
    answers_no_tag += formatted_answers_no_tag

    # Example text
    example = (
        "Example:\n\n"
        "The patient has received 24 radiation treatments and reports several significant symptoms, "
        "including frequent urination that interferes 'somewhat' with daily activities, and so forth..."
    )

    instructions = (
        "You are an experienced radiation oncologist. Summarize the following Q&A data into two sentences of natural language. "
        "Indicate only the most important symptoms using the exact parenthetical keywords from the question. "
        "Include the number of radiation treatments and the last question's free-text if any. Summarize in English."
    )

    # Build the actual prompt
    keywords = define_keywords()

    prompt_lines = []
    for i, question in enumerate(questions):
        # If there's a relevant keyword, attach it in parentheses
        if keywords[i]:
            question_plus_kw = f"{question} ({keywords[i]})"
        else:
            question_plus_kw = question
        prompt_lines.append(f"{question_plus_kw}\n{answers[i]}")

    prompt_body = "\n".join(prompt_lines)
    prompt = (
        f"{prompt_body}\n---\n\n"
        f"{instructions}\n\n---\n\n"
        f"{example}"
    )

    # Combine Q&A
    QA = prompt

    # Compute numeric scores
    score_dict = define_scores_dict()
    scores = answers_to_score(answers, score_dict)

    return prompt, scores, QA, answers_no_tag
