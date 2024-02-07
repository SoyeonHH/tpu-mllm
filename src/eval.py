import numpy as np
import os
import requests
from PIL import Image
from typing import List, Dict, Any, Tuple
import json

from utils import *
from data_loader import PerceptionDataset
from models.baseline import FreqMCVQABaseline

from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from torch.utils.data import DataLoader

def main():

    # Load the dataset
    data_path = "./data/annotated_test.json"
    video_path = "/data/perception_test/mc-vqa/train_videos"
    audio_path = "/data/perception_test/mc-vqa/train_audios"

    with open(data_path, "r") as f:
        pt_db_dict = json.load(f)

    dataset = PerceptionDataset(pt_db_dict, video_path, "mc_question", "test")
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load the model
    model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

    # Evaluate the model
    all_answers = []
    for batch in data_loader:
        video_id = batch["metadata"]["video_id"]
        question = batch["mc_question"]["question"]
        image = Image.open(requests.get(video_id, stream=True).raw)

        inputs = processor(text=question, images=image, return_tensors="pt")

        generated_ids = model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=True,
            max_new_tokens=128,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        processed_text, entities = processor.post_process_generation(generated_text)
        all_answers.append(entities)



if __name__ == "__main__":
    main()