import collections
import json
import os
import random
from typing import List, Dict, Any, Tuple
import zipfile
import cv2
import requests
import numpy as np
import torch

def download_and_unzip(url: str, destination: str):
  """Downloads and unzips a .zip file to a destination.

  Downloads a file from the specified URL, saves it to the destination
  directory, and then extracts its contents.

  If the file is larger than 1GB, it will be downloaded in chunks,
  and the download progress will be displayed.

  Args:
    url (str): The URL of the file to download.
    destination (str): The destination directory to save the file and
      extract its contents.
  """
  if not os.path.exists(destination):
    os.makedirs(destination)

  filename = url.split('/')[-1]
  file_path = os.path.join(destination, filename)

  if os.path.exists(file_path):
    print(f'{filename} already exists. Skipping download.')
    return

  response = requests.get(url, stream=True)
  total_size = int(response.headers.get('content-length', 0))
  gb = 1024*1024*1024

  if total_size / gb > 1:
    print(f'{filename} is larger than 1GB, downloading in chunks')
    chunk_flag = True
    chunk_size = int(total_size/100)
  else:
    chunk_flag = False
    chunk_size = total_size

  with open(file_path, 'wb') as file:
    for chunk_idx, chunk in enumerate(
        response.iter_content(chunk_size=chunk_size)):
      if chunk:
        if chunk_flag:
          print(f"""{chunk_idx}% downloading
          {round((chunk_idx*chunk_size)/gb, 1)}GB
          / {round(total_size/gb, 1)}GB""")
        file.write(chunk)
  print(f"'{filename}' downloaded successfully.")

  with zipfile.ZipFile(file_path, 'r') as zip_ref:
    zip_ref.extractall(destination)
  print(f"'{filename}' extracted successfully.")

  os.remove(file_path)


def load_db_json(db_file: str) -> Dict[str, Any]:
  """Loads a JSON file as a dictionary.

  Args:
    db_file (str): Path to the JSON file.

  Returns:
    Dict: Loaded JSON data as a dictionary.

  Raises:
    FileNotFoundError: If the specified file doesn't exist.
    TypeError: If the JSON file is not formatted as a dictionary.
  """
  if not os.path.isfile(db_file):
    raise FileNotFoundError(f'No such file: {db_file}')

  with open(db_file, 'r') as f:
    db_file_dict = json.load(f)
    if not isinstance(db_file_dict, dict):
      raise TypeError('JSON file is not formatted as a dictionary.')
    return db_file_dict


def load_mp4_to_frames(filename: str) -> np.array:
  """Loads an MP4 video file and returns its frames as a NumPy array.

  Args:
    filename (str): Path to the MP4 video file.

  Returns:
    np.array: Frames of the video as a NumPy array.
  """
  assert os.path.exists(filename), f'File {filename} does not exist.'
  cap = cv2.VideoCapture(filename)

  num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

  vid_frames = np.empty((num_frames, height, width, 3), dtype=np.uint8)

  idx = 0
  while True:
    ret, frame = cap.read()
    if not ret:
      break

    vid_frames[idx] = frame
    idx += 1

  cap.release()
  return vid_frames


def get_video_frames(data_item: Dict[str, Any],
                     video_folder_path: str) -> np.array:
  """Loads frames of a video specified by an item dictionary.

  Assumes format of annotations used in the Perception Test Dataset.

  Args:
    data_item (Dict): Item from dataset containing metadata.
    video_folder_path (str): Path to the directory containing videos.

  Returns:
    np.array: Frames of the video as a NumPy array.
  """
  video_file = os.path.join(video_folder_path,
                            data_item['metadata']['video_id']) + '.mp4'
  vid_frames = load_mp4_to_frames(video_file)
  assert data_item['metadata']['num_frames'] == vid_frames.shape[0]
  return vid_frames


def get_mask(lengths, max_length, device):
  """Computes a batch of padding masks given batched lengths"""
  mask = 1 * (
    torch.arange(max_length).unsqueeze(1).to(device) < lengths
  ).transpose(0, 1)
  return mask