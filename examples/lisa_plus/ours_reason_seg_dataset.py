# -*- coding: utf-8 -*-
"""Reference LISA++ adapter: SonoIntent split jsonl -> ReasonSeg training format.

Drop this file into the LISA++ codebase as ``utils/ours_reason_seg_dataset.py``
and register it in the training entry point; the imports below are LISA++'s.

Record layout read here: ``{dataset_dir}/{split}.jsonl`` with ``rendered_image``
and ``rendered_mask`` relative to ``FINAL_ROOT = dirname(dataset_dir)``, the
question in ``problem`` and the reference reasoning in ``thinking``. Masks are
binary images; the training target is the reasoning followed by the ``[SEG]``
token. ``max_samples`` caps the loaded pool for debugging.
"""
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from model.llava1p5 import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import DEFAULT_IMAGE_TOKEN


class OurReasonSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        reason_seg_data="ours|train",
        explanatory=0.1,
        max_samples=None,
        random_sample=True,
    ):
        self.samples_per_epoch = samples_per_epoch
        self.image_size = image_size
        self.num_classes_per_sample = num_classes_per_sample
        self.precision = precision
        self.tokenizer = tokenizer
        self.random_sample = random_sample

        # dataset_dir = <...>/final/split ; FINAL_ROOT = <...>/final
        self.dataset_dir = base_image_dir
        self.final_root = os.path.dirname(base_image_dir)

        _, split = reason_seg_data.split("|")
        jsonl_path = os.path.join(self.dataset_dir, f"{split}.jsonl")
        self.rows = [json.loads(l) for l in open(jsonl_path)]
        if max_samples:
            self.rows = self.rows[:max_samples]

        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        row = random.choice(self.rows) if self.random_sample else self.rows[idx % len(self.rows)]
        image_path = os.path.join(self.final_root, row["rendered_image"])
        mask_path = os.path.join(self.final_root, row["rendered_mask"])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]

        mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)

        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        question = row["problem"]
        answer = row["thinking"] + " Corresponding segmentation: [SEG]"

        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
        conv.append_message(conv.roles[1], answer)
        conversations = [conv.get_prompt()]

        questions = [question]
        sampled_sents = [question]

        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        masks = np.stack([mask], axis=0)
        masks = torch.from_numpy(masks)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_sents,
        )
