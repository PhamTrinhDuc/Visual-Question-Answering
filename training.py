import torch
import pandas as pd
import numpy as np
import wandb
import os
import evaluate
import dotenv
dotenv.load_dotenv()
from torchvision import transforms
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Dict, Tuple, List
from transformers import (
  Trainer, 
  AutoTokenizer,
  TrainingArguments, 
  EarlyStoppingCallback, 
  AutoFeatureExtractor,
)
from data_processor import VQADataset
from config import config
from model.vqa_model import VQAModel
from model.processor import VQAProcessor

# wandb.login(key=os.getenv("WANDB_API_KEY"))
tokenizer = AutoTokenizer.from_pretrained(config.text_model)
processor = AutoFeatureExtractor.from_pretrained(config.image_model)
vocab = tokenizer.get_vocab()
vocab_swap = {value: key for key, value in vocab.items()}


def count_parameters(model, only_trainable=True):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

 
@dataclass
class VQADataCollatorForGeneration:
  """
  Data collator cho VQA, sử dụng VQAProcessor.
  """
  processor: VQAProcessor
  padding: bool = True

  def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Gộp danh sách các mẫu thành batch.

    Args:
        batch: List[Dict] chứa:
            - image: PIL Image
            - question: Chuỗi câu hỏi
            - label: Chuỗi câu trả lời

    Returns:
        Dict chứa:
            - pixel_values: [batch_size, channels, height, width]
            - input_ids: [batch_size, question_seq_len]
            - attention_mask: [batch_size, question_seq_len]
            - labels: [batch_size, answer_seq_len]
    """
    # Tách các thành phần
    images = [item["image"] for item in batch]
    questions = [item["question"] for item in batch]
    answers = [item["answer"] for item in batch]

    # Gọi processor để xử lý
    batch_features = self.processor(
        images=images,
        questions=questions,
        answers=answers,
        padding="max_length" if self.padding else False,
        return_tensors="pt",
    )
    return batch_features

class CustomTrainer(Trainer):
  def get_train_dataloader(self):
    return DataLoader(
        self.train_dataset,
        collate_fn=self.data_collator,
        num_workers=4,
        shuffle=True
    )

  def get_eval_dataloader(self, eval_dataset=None):
    return DataLoader(
        eval_dataset or self.eval_dataset,
        collate_fn=self.data_collator,
        num_workers=4,
        shuffle=False
    )
  
def compute_metrics(eval_tuple: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
  print("eval_tuple", eval_tuple)
  print("eval tuple shape: ", eval_tuple[0].shape, eval_tuple[1].shape)

  bleu_metric = evaluate.load("bleu")
  rouge_metric = evaluate.load("rouge")
  predictions, references = eval_tuple
  
  # Decode nếu là ID (bắt buộc nếu Trainer trả ra token ID)
  predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
  references = tokenizer.batch_decode(references, skip_special_tokens=True)

  predictions = [pred.strip() for pred in predictions]
  references = [ref.strip() for ref in references]

  # BLEU expects list of list of tokens
  bleu = bleu_metric.compute(predictions=[p.split() for p in predictions],
                              references=[[r.split()] for r in references])
  # ROUGE expects plain text
  rouge = rouge_metric.compute(predictions=predictions,
                                references=references,
                                use_stemmer=True)
  return {
      "bleu": bleu["bleu"],
      "rougeL": rouge["rougeL"]
  }

if __name__ == "__main__":
  df_train = pd.read_csv(config.train_csv)
  df_dev = pd.read_csv(config.dev_csv)
  df_test = pd.read_csv(config.test_csv)

  train_dataset = VQADataset(dataframe=df_train, mode="train")
  eval_dataset = VQADataset(dataframe=df_dev, mode="dev")
  test_dataset = VQADataset(dataframe=df_test, mode="test")

  model = VQAModel(text_model=config.text_model, 
                    image_model=config.image_model, 
                    device=config.DEVICE).to(config.DEVICE)
  
  data_collator = VQADataCollatorForGeneration(processor=VQAProcessor())
  early_stop = EarlyStoppingCallback(early_stopping_patience = 2,early_stopping_threshold = 0)

  training_args = TrainingArguments(
    output_dir="./vqa_results",
    per_device_train_batch_size=config.BATCH_SIZE,
    per_device_eval_batch_size=max(1, config.BATCH_SIZE // 2),  # Giảm batch size cho evaluation
    # max_steps=1000,
    num_train_epochs=5,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    warmup_steps=200,
    save_steps=500,
    logging_steps=10,
    eval_strategy="epoch",
    eval_steps=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_strategy="epoch",  # Hoặc "no"
    save_total_limit=2,
    fp16=True,  # Mixed precision cho GPU
    max_grad_norm=1.0,  # Gradient clipping để ổn định huấn luyện
    # report_to="wandb",
    gradient_accumulation_steps=2,
  )

  # wandb.init(
  #    project="vqa-training",
  #    name="vqa-from-scratch", 
  #    config=training_args
  # )

  # Tạo Trainer
  trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[early_stop],
  )

  # Kiểm tra batch
  batch = trainer.get_train_dataloader().__iter__().__next__()
  print("Batch from Trainer:", batch.keys())

  print("Starting training...") 
  trainer.train()
  print("Training completed.")