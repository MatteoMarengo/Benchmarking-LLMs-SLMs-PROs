"""
evaluation_utils.py

Contains various functions to evaluate the summarized output:
1) compute_metric() for coverage
2) recall() for relevant symptoms
3) kappa_cohen_index() for agreement
4) LLM-based scoring via GPT

"""

import re
import os
import openai
from .scoring_utils import define_scores_dict
from .prompt_utils import define_keywords

# Globally defined dictionary for questions -> keywords
question_to_keywords = {
    2:  ['fatigue'],
    3:  ['fatigue'],
    4:  ['gas'],
    5:  ['abdominal pain'],
    6:  ['abdominal pain'],
    7:  ['abdominal pain'],
    8:  ['abdominal pain'],
    9:  ['painful urination'],
    10: ['urinary urgency'],
    11: ['urinary urgency'],
    12: ['urinary frequency'],
    13: ['urinary frequency'],
    14: ['urine color'],
    15: ['leakage'],
    16: ['leakage'],
    17: ['skin burns']
}

keywords = [
    "fatigue", 
    "gas", 
    "diarrhea", 
    "abdominal pain", 
    "painful urination", 
    "urinary urgency", 
    "urinary frequency", 
    "urine color", 
    "leakage", 
    "skin burns"
]

def scores_properties():
    """
    Return the question->keyword mapping for the entire set,
    as well as symptom/daily activity lists.
    """
    kw = define_keywords()
    keywords_dict = {i+1: val for i, val in enumerate(kw)}
    symptoms = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]  # whichever Q's you consider "symptoms"
    daily_activities = []  # Or some subset
    return keywords_dict, symptoms, daily_activities


def check_presence(output, keyword):
    """
    Returns True if the majority of words in `keyword` are found in `output`.
    For single words, direct presence is enough.
    """
    output = output.lower()
    if " " in keyword:
        words = keyword.split()
        present_count = sum(word in output for word in words)
        return present_count >= len(words)/2
    else:
        return keyword in output


def check_symptoms_question(output, keyword):
    """
    For 'symptom' style questions, we just check if the relevant keyword(s) appear in the summary.
    """
    return check_presence(output, keyword)


def compute_metric(scores, keywords_dict, output, daily_activities, symptoms):
    """
    1) Identifies which questions have severity > 0.5
    2) Checks if those are included in the summary
    3) Returns ratio = (# included) / (total > 0.5)
    """
    total_sum, in_summary = 0, 0
    for question_idx, val in scores.items():
        if val > 0.5:
            total_sum += 1
            if question_idx in symptoms:
                if check_symptoms_question(output, keywords_dict[question_idx]):
                    in_summary += 1

    # If no question had severity > 0.5, we consider the coverage perfect = 1
    if total_sum == 0:
        return 1.0

    return in_summary / total_sum


def right_number_treatment(output, answers):
    """
    Checks if the first number in `output` matches the # of treatments from `answers[0]`.
    Returns:
      1 if correct match,
      0 if no number,
      2 if mismatch
    """
    numbers = re.findall(r'\d+', output)
    if not numbers:
        return 0
    if numbers[0] == answers[0]:
        return 1
    return 2


def right_free_answer(output, answers):
    """
    If the last answer is not blank, check if it’s present in the output.
    Return 1 if found, 2 if mismatch, 0 if not applicable.
    """
    if not answers[-1]:
        return 0
    if answers[-1].lower() in output.lower():
        return 1
    return 2


def keywords_in_output(answers_score, question_to_keywords):
    """
    Gathers all keywords that should appear if the question's severity > 0.5.
    """
    keywords_in_out = []
    for idx, score in enumerate(answers_score):
        if score is not None and score > 0.5:
            if (idx+1) in question_to_keywords:
                keywords_in_out.append("".join(question_to_keywords[idx+1]))
    return list(set(keywords_in_out))


def keywords_not_in_output(all_keywords, keywords_in_output_list):
    """
    Returns those keywords that were NOT expected to appear
    (i.e., severity <= 0.5), so we check that the summarizer didn't add them.
    """
    not_expected = []
    for kw in all_keywords:
        if kw not in keywords_in_output_list:
            not_expected.append(kw)
    return not_expected


def recall(keywords_to_check, number_treatment, free_answer, output):
    """
    Simple recall measure: TP / (TP + FN).
    We increment TP if expected info is found, otherwise FN.
    """
    TP, FN = 0, 0
    # number_treatment
    if number_treatment == 1:
        TP += 1
    elif number_treatment == 2:
        FN += 1

    # free answer
    if free_answer == 1:
        TP += 1
    elif free_answer == 2:
        FN += 1

    # keywords
    for kw in keywords_to_check:
        if kw in output:
            TP += 1
        else:
            FN += 1

    if (TP + FN) == 0:
        return 1.0  # If there's literally nothing to check, we call it 1
    return TP/(TP+FN)


def kappa_cohen_index(output, keywords_to_check, not_keywords_to_check, number_treatment, free_answer):
    """
    Kappa: 
      - We treat each expected keyword as +ve class
      - Each NOT-expected keyword as -ve class
    We also factor in presence/absence of # of treatments & free answer.
    """
    TP, FN, TN, FP = 0, 0, 0, 0

    # Number treatments
    if number_treatment == 1:
        TP += 1
    elif number_treatment == 2:
        FN += 1

    # Free answer
    if free_answer == 1:
        TP += 1
    elif free_answer == 2:
        FN += 1

    # Expected keywords
    for kw in keywords_to_check:
        if kw in output:
            TP += 1
        else:
            FN += 1

    # NOT expected
    for kw in not_keywords_to_check:
        if kw in output:
            FP += 1
        else:
            TN += 1

    # Po = observed agreement
    Po = (TP + TN) / (TP + TN + FP + FN) if (TP+TN+FP+FN) else 1

    # Pe = chance agreement
    P1 = ((TP + FP) * (TP + FN)) / ((TP + TN + FP + FN) ** 2)
    P2 = ((TN + FN) * (TN + FP)) / ((TP + TN + FP + FN) ** 2)
    Pe = P1 + P2

    if Pe == 1:
        return 0.0  # degenerate case

    kappa = (Po - Pe) / (1 - Pe)
    return kappa


def LLM_evaluator(QA, summary):
    """
    Calls GPT-4 (via openai) to get a numeric score (0-1) about how accurate
    the summary is vs. the QA.

    The instructions are short: "Return only a float between 0 and 1."
    """
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    instructions = (
        "As an evaluator of LLM summarization accuracy, assign a score from 0 to 1 "
        "where 0 means 'completely inaccurate' and 1 means 'perfectly accurate'. "
        "Focus on how well the summary captures critical patient symptoms. "
        "Output only the float, nothing else."
    )

    prompt = instructions + "\n\n" + QA + "\n\n" + summary
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        grade = response.choices[0].message["content"].strip()
        try:
            return float(grade)
        except ValueError:
            # fallback if GPT output is not purely numeric
            return 0.0
    except Exception as e:
        print("LLM evaluator error:", e)
        return 0.0
