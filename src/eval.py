import os
import argparse
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


def get_args():
    parser = argparse.ArgumentParser(description="Perception Test")
    parser.add_argument("--data_path", type=str, default="/data1/soyeon/perception-test/data/all_valid.json")
    parser.add_argument("--video_path", type=str, default="/data1/soyeon/perception-test/data/valid_videos")
    parser.add_argument("--audio_path", type=str, default="/data1/soyeon/perception-test/data/valid_audios")

    # model
    parser.add_argument("--model_name", type=str, default="microsoft/kosmos-2-patch14-224")
    parser.add_argument("--frame_num", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--frame_sampling", type=str, default="uniform")

    args = parser.parse_args()

    args.output_path = f"/data1/soyeon/perception-test/result/{args.model_name}/{args.frame_sampling}_{args.frame_num}f.json"

    return args


def main():

    args = get_args()

    # Load the dataset
    data_path = args.data_path
    video_path = args.video_path
    audio_path = args.audio_path

    with open(data_path, "r") as f:
        pt_db_dict = json.load(f)

    dataset = PerceptionDataset(pt_db_dict, video_path, "mc_question", "test")

    # Load the model
    model = AutoModelForVision2Seq.from_pretrained(args.model_name)
    processor = AutoProcessor.from_pretrained(args.model_name)

    # Set the device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Evaluate the model
    model.eval()
    answers = {}

    for video_item in dataset:
        video_id = video_item["metadata"]["video_id"]
        video_answers = []

        # video processing
        video_frames = get_video_frames(video_item, video_path)
        video_frames = sample_video_frames(video_frames, args.frame_num, args.frame_sampling)

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

            # 

            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)

            video_answers.append({
                "id": q_idx,
                "question": question,
                "answer": processed_text
            })

        answers[video_id] = video_answers

        # save the answers
        with open(args.output_path, "w") as f:
            json.dump(answers, f, indent=4)
    
    # Evaluate the model
    # calc overall top-1
    # top1 = calc_top1(answers, pt_db_dict)



            
            



if __name__ == "__main__":
    main()