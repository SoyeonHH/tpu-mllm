import random
from typing import Dict, Any, Tuple
import numpy as np


class FreqMCVQABaseline:
  """Multiple-Choice vQA Model (Frequency Baseline).

  Attributes:
    answer_db: Dictionary containing the question: answer database.
  """

  def __init__(self, db_dict: Dict[str, Any]):
    """Initializes the MCVQA class.

    Args:
      db_dict (Dict): Dictionary containing the database.
    """
    self.answer_db = self.build_answer_db(db_dict)

  def build_answer_db(self, db_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Builds a answer database.

    Returns a dictionary where the key is the question string and the value
    is a list of strings, each string a correct answer to a previous
    instance of the question.

    Args:
    	db_dict (Dict): Dictionary containing the database of
    		questions and corresponding answers.

    Returns:
    	Dict[str, Any]: Dictionary where the key is question string and
    	the value is list of full correct answers each time that question
    	was asked in the db_dict.
    """
    question_db = {}
    for vid in db_dict.values():
      for question in vid['mc_question']:
        try:
          question_db[question['question']]
        except KeyError:
          question_db[question['question']] = []

        answer = question['options'][question['answer_id']]
        question_db[question['question']].append(answer)

    return question_db

  def random_answer(self, question: Dict[str, Any]) -> Tuple[int, str]:
    """Answer a mc question with a random valid answer.

    Picks a random answer for a multiple choice question.

    Args:
      question (Dict): Question.

    Returns:
      Tuple: Answer ID and answer text.
    """
    answer = random.sample(question['options'], 1)[0]
    answer_id = question['options'].index(answer)
    return answer_id, answer

  def answer_q(self, frames: np.array, question: Dict[str, Any],
               shots: int) -> Tuple[int, str]:
    """Answer a multiple choice question.

    Given a number of shots n, we sample at random n correct answers from the
    training set, then we count the number of occurrences for each of the
    options of the given question and keep the most popular answer (the answer
    with the highest count). For multiple equally popular options, we sample
    one at random.

    Args:
      frames: Video frames (placeholder, unused).
      question (Dict): Question.
      shots: Number of shots for multi-shot answer selection. If 0, random
      sampling is used, if -1, full population is using, else if n, uses sample
      size n of population.

    Returns:
      Tuple: Answer ID and answer text.

    """
    del frames  # unused
    assert shots >= -1

    # sample answers to current question from stored db
    answer_sample = self.answer_db[question['question']]

    if shots == 0:
      return self.random_answer(question)
    elif shots == -1 or shots > len(answer_sample):
      sample_size = len(answer_sample)
    else:
      sample_size = shots

    # get subsample
    answer_subsample = random.sample(answer_sample, sample_size)

    # count frequency of valid answers
    option_freq = []
    for option in question['options']:
      option_freq.append(answer_subsample.count(option))

    # check if more than one answer has same occurence
    max_val = max(option_freq)
    indices = [i for i, value in enumerate(option_freq) if value == max_val]

    # if true sample from the indices
    if len(indices) > 1:
      answer_id = random.sample(indices, 1)[0]
    else:
      answer_id = indices[0]

    answer = question['options'][answer_id]
    return answer_id, answer