"""
scoring_utils.py

Contains utility functions to define scoring dictionaries and
convert answers into scores.
"""

import re

def define_scores_dict():
    """
    Simple scoring dictionary based on severity levels.
    Returns:
        dict: Mapping from text-based answers to numeric severity.
    """
    scores_dict = {
        "yes": 1,
        "no": 0,
        "not at all": 0,
        "a little bit": 0.25,
        "somewhat": 0.5,
        "quite a bit": 0.75,
        "very much": 1,
        "never": 0,
        "rarely": 0.25,
        "occasionally": 0.5,
        "frequently": 0.75,
        "almost constantly": 1,
        "none": 0,
        "mild": 0.25,
        "moderate": 0.5,
        "severe": 0.75,
        "very severe": 1,
        "not applicable": 0
    }
    return scores_dict


def extract_after_number(text):
    """
    Extract the answer from the prompt text,
    ignoring the 'A#: ' prefix.
    """
    match = re.search(r"A\d+\.\s*", text)
    if match:
        end_pos = match.end()
        return text[end_pos:]
    else:
        return "Pattern not found"


def answers_to_score(answers, scores_dict):
    """
    Convert a list of string answers into numeric scores
    using the specified dictionary.

    Args:
        answers (list): List of answers, e.g. ["A1. 25", "A2. Mild", ...]
        scores_dict (dict): Mapping from text to float score

    Returns:
        dict: Keyed by question index, with float scores as values.
    """
    adjusted_scores = {k.lower(): v for k, v in scores_dict.items()}
    scored_answers = []
    for i, answer in enumerate(answers):
        extracted_answer = extract_after_number(answer).lower().strip()
        score = adjusted_scores.get(extracted_answer, 0)  # default 0 if not found
        scored_answers.append(score)

    scores = {i+1: val for i, val in enumerate(scored_answers)}
    print(scores)
    return scores


def answers_to_score_kappa(answers, scores_dict):
    """
    Slightly different approach to scoring for Kappa,
    ignoring the 'A#: ' prefix entirely and returning a list.

    Args:
        answers (list): e.g. ["25", "Mild", ...]
        scores_dict (dict): Mapping from text to float

    Returns:
        list: numeric scores
    """
    answers_score = []
    adjusted_scores = {k.lower(): v for k, v in scores_dict.items()}
    for answer in answers:
        lower_answer = answer.lower().strip()
        score = adjusted_scores.get(lower_answer, None)
        answers_score.append(score)
    return answers_score
