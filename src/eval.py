import os
# os.environ["PJRT_DEVICE"] = "TPU"

import requests
from PIL import Image
from typing import List, Dict, Any, Tuple
import json
from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq
from utils import *
from data_loader import PerceptionDataset
from models.baseline import FreqMCVQABaseline


def main():

    # Load the dataset
    data_path = "/home/soyeon/workspace/perception-test/data/all_valid.json"
    video_path = "/home/soyeon/workspace/perception-test/data/valid_videos"
    audio_path = "/home/soyeon/workspace/perception-test/data/valid_audios"

    with open(data_path, "r") as f:
        pt_db_dict = json.load(f)

    dataset = PerceptionDataset(pt_db_dict, video_path, "mc_question", "test")

    # Load the model
    model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

    # Set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Evaluate the model
    model.eval()
    answers = {}

    for video_item in dataset:
        video_id = video_item["metadata"]["video_id"]
        video_answers = []

        # video processing
        video_frames = get_video_frames(video_item, video_path)
        # video_values = processor(images=video_frames, return_tensors="pt")["pixel_values"]
        # video_values = torch.concat(video_values, dim=0).to(device)
        # video_len = video_values.size(0)
        # video_mask = get_mask(video_len, video_values.size(1), device).to(device)

        for q_idx, q in enumerate(video_item['mc_question']):
            # question processing
            question = q["question"]
            video_prompt = "<i>" * video_frames.shape[0]
            prompt = f"<s> {video_prompt} {question}</s>"
            # inputs = processor(images=video_frames, text=question, return_tensors="pt")
            inputs = process_interleaved_example(processor, prompt, images=video_frames, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # model inference
            generated_ids = model.generate(
                pixel_values=inputs['pixel_values'],
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                image_embeds=None,
                image_embeds_position_mask=inputs["image_embeds_position_mask"],
                use_cache=True,
                max_new_tokens=128,
            )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)

            video_answers.append({
                "id": q_idx,
                "question": question,
                "answer": processed_text
            })

        answers[video_id] = video_answers

        # save the answers
        with open("/home/soyeon/workspace/perception-test/result/answers.json", "w") as f:
            json.dump(answers, f, indent=4)
    
    # Evaluate the model
    # calc overall top-1
    # top1 = calc_top1(answers, pt_db_dict)



            
            



if __name__ == "__main__":
    main()