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


PATH_DIR = "/home/soyeon/workspace/perception-test"


def get_args():
    parser = argparse.ArgumentParser(description="Perception Test")
    parser.add_argument("--data_path", type=str, default=f"{PATH_DIR}/data/all_valid.json")
    parser.add_argument("--video_path", type=str, default=f"{PATH_DIR}/data/valid_videos")
    parser.add_argument("--audio_path", type=str, default=f"{PATH_DIR}/data/valid_audios")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")

    # model
    parser.add_argument("--model_name", type=str, default="microsoft/kosmos-2-patch14-224")
    parser.add_argument("--frame_num", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--frame_sampling", type=str, default="uniform")

    args = parser.parse_args()

    # output path
    if not os.path.exists(f"{PATH_DIR}/result/{args.model_name}"):
        os.makedirs(f"{PATH_DIR}/result/{args.model_name}")

    args.output_path = f"{PATH_DIR}/result/{args.model_name}/{args.frame_sampling}_{args.frame_num}f.json"

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
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
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
            options = q["options"]
            video_prompt = "<i>" * video_frames.shape[0]
            prompt = f"<s> {video_prompt} {question}</s>"
            # inputs = processor(images=video_frames, text=question, return_tensors="pt")
            inputs = process_interleaved_example(processor, prompt, images=video_frames, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # model inference
            # generated_ids = model.generate(
            #     pixel_values=inputs['pixel_values'],
            #     input_ids=inputs['input_ids'],
            #     attention_mask=inputs['attention_mask'],
            #     image_embeds=None,
            #     image_embeds_position_mask=inputs["image_embeds_position_mask"],
            #     use_cache=True,
            #     max_new_tokens=128,
            # )

            # generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)

            # get top-1 answer
            logits = []
            for option in options:
                inputs = process_interleaved_example(processor, f"<s> {video_prompt} {question} {option}</s>", images=video_frames, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    generated_ids = model.generate(
                        pixel_values=inputs['pixel_values'],
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        image_embeds=None,
                        image_embeds_position_mask=inputs["image_embeds_position_mask"],
                        use_cache=True,
                        max_new_tokens=args.max_new_tokens,
                        return_dict_in_generate=True,
                        output_scores=True
                    )

                    # get option's logit
                    option_ids = processor.tokenizer.encode(option, add_special_tokens=False)
                    logit = generated_ids.scores[0].squeeze(0)
                    option_logit = logit[option_ids].mean()
                    logits.append(option_logit.item())
            
            logits_prob = F.softmax(torch.tensor(logits), dim=0)
            top1_idx = torch.argmax(logits_prob).item()
            top1_answer = options[top1_idx]

            # video_answers.append({
            #     "id": q_idx,
            #     "question": question,
            #     "answer": processed_text
            # })
            video_answers.append({
                "id": q_idx,
                "answer_id": top1_idx,
                "answer": top1_answer
            })

        answers[video_id] = video_answers

        # save the answers
        with open(args.output_path, "w") as f:
            json.dump(answers, f, indent=4)
    
    # Evaluate the model
    # calc overall top-1
    top1 = calc_top1(answers, pt_db_dict)
    print(f"Overall top-1: {top1:.2f}")

    # calc top-1 by area, reasoning and tag
    top1_by_cat = calc_top1_by_cat(answers, pt_db_dict)
    print("Top-1 by category:")
    for k, v in top1_by_cat.items():
        print(f"{k}: {v[0]/v[1]:.2f}")


if __name__ == "__main__":
    main()