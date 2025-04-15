from underthesea import word_tokenize, text_normalize
# from vncorenlp import VnCoreNLP
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from config import config
import pandas as pd
import os

def process_text(df):
    df['question'] = [word_tokenize(text_normalize(x), format='text') for x in df['question']]
    df['answer'] = [word_tokenize(text_normalize(str(x)), format='text') for x in df['answer']]
    return df

class VQADataset(Dataset):
    def __init__(self, dataframe, transform=None, mode="train"):
        self.data = dataframe if mode != "train" else process_text(dataframe)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, question, answer = self.data.iloc[idx]
        image_path = os.path.join(config.FOLDER_IMAGE_PATH, image_name)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "question": question,
            "answer": answer,
        }