import numpy as np
import os
import requests
from PIL import Image
from typing import List, Dict, Any, Tuple


class PerceptionDataset():
    """Dataset class to store video items from dataset.

    Attributes:
        video_folder: Path to the folder containing the videos.
        task: Task type for annotations.
        split: Dataset split to load.
        pt_db_list: List containing annotations for dataset according to
        split and task availability.
    """

    def __init__(self, pt_db_dict: Dict[str, Any], video_folder: str,
                task: str, split: str) -> None:
        """Initializes the PerceptionDataset class.

        Args:
        pt_db_dict (Dict): Dictionary containing annotations for dataset.
        video_folder (str): Path to the folder containing the videos.
        task (str): Task type for annotations.
        split (str): Dataset split to load.
        """
        self.video_folder = video_folder
        self.task = task
        self.split = split
        self.pt_db_list = self.load_dataset(pt_db_dict)

    def load_dataset(self, pt_db_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Loads the dataset from the annotation file and processes.

        Dict is processed according to split and task.

        Args:
        pt_db_dict: (Dict): Dictionary containing
            annotations.

        Returns:
        List: List of database items containing annotations.
        """
        pt_db_list = []
        for _, v in pt_db_dict.items():
            # if v['metadata']['split'] == self.split:
            if v[self.task]:  # If video has annotations for this task
                pt_db_list.append(v)

        return pt_db_list

    def __len__(self) -> int:
        """Returns the total number of videos in the dataset.

        Returns:
        int: Total number of videos.
        """
        return len(self.pt_db_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Returns the video and annotations for a given index.

        example_annotation = {
            'video_10909':{
            'mc_question': [
                {'id': 0, 'question': 'Is the camera moving or static?',
                    'options': ["I don't know", 'moving', 'static or shaking'],
                    'answer_id': 2, 'area': 'physics', 'reasoning': 'descriptive',
                    'tag': ['motion']
                }
            ]
            }
        }

        Args:
        idx (int): Index of the video.

        Returns:
        Dict: Dictionary containing the video frames, metadata, annotations.
        """
        data_item = self.pt_db_list[idx]
        annot = data_item[self.task]
        metadata = data_item['metadata']
        # here we are loading a placeholder as the frames
        # the commented out function below will actually load frames
        vid_frames = np.zeros((metadata['num_frames'], 1, 1, 1))
        # frames = get_video_frames(video_item, self.video_folder)

        return {'metadata': metadata,
                self.task: annot,
                'frames': vid_frames}