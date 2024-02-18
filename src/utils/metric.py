from typing import List, Dict, Any, Tuple
import numpy as np
import os
from utils.categories import *

def calc_top1(answers_dict: Dict[str, Any],
              db_dict: Dict[str, Any]) -> float:
  """Calculates the top-1 accuracy and results for each category.

  Args:
    answers_dict (Dict): Dictionary containing the answers.
    db_dict (Dict): Dictionary containing the database.

  Returns:
    float: Top-1 accuracy.

  Raises:
    KeyError: Raises error if the results contain a question ID that does not
      exist in the annotations.
    ValueError: Raises error if the answer ID is outside the expected range
      [0,2].
    ValueError: Raises error if the text of the answer in the results does not
      match the expected string answer in the annotations.
    ValueError: If answers are missing from the results.

  """
  expected_total = 0
  total_correct = 0
  total = 0

  for v in db_dict.values():
    expected_total += len(v['mc_question'])

  for vid_id, vid_answers in answers_dict.items():
    for answer_info in vid_answers:
      answer_id = answer_info['answer_id']
      question_idx = answer_info['id']

      try:
        ground_truth = db_dict[vid_id]['mc_question'][question_idx]
      except KeyError as exc:
        raise KeyError('Unexpected question ID in answer.') from exc

      if answer_id > 2 or answer_id < 0:
        raise ValueError(f'Answer ID must be in range [0:2], got {answer_id}.')
      if ground_truth['options'][answer_id] != answer_info['answer']:
        raise ValueError('Answer text is not as expected.')

      gt_answer_id = ground_truth['answer_id']
      val = int(gt_answer_id == answer_id)
      total_correct += val
      total += 1

  if expected_total != total:
    raise ValueError('Missing answers in results.')

  return total_correct/total


def calc_top1_by_cat(answers_dict: Dict[str, Any],
                     db_dict: Dict[str, Any]) -> Dict[str, Any]:
  """Calculates the top-1 accuracy and results for each category.

  Args:
    answers_dict (Dict): Dictionary containing the answers.
    db_dict (Dict): Dictionary containing the database.

  Returns:
    Dict: Top-1 accuracy and results for each category.
  """
  results_dict = {k: v for k, v in zip(CAT, np.zeros((len(CAT), 2)))}

  for vid_id, vid_answers in answers_dict.items():
    for answer_info in vid_answers:
      answer_id = answer_info['answer_id']
      question_idx = answer_info['id']
      ground_truth = db_dict[vid_id]['mc_question'][question_idx]
      gt_answer_id = ground_truth['answer_id']
      val = int(gt_answer_id == answer_id)

      used_q_areas = []
      q_areas = [TAG_AREA[tag] for tag in ground_truth['tag']]
      for a in q_areas:
        if a not in used_q_areas:
          results_dict[a][0] += val
          results_dict[a][1] += 1
          used_q_areas.append(a)

      results_dict[ground_truth['reasoning']][0] += val
      results_dict[ground_truth['reasoning']][1] += 1
      for t in ground_truth['tag']:
        results_dict[t][0] += val
        results_dict[t][1] += 1

  results_dict = {k: v[0]/v[1] for k, v in results_dict.items()}
  return results_dict