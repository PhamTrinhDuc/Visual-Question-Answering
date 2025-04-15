import torch
from PIL import Image
from config import config
from typing import Dict, List, Union
from transformers import AutoTokenizer
from transformers import ViTImageProcessor



class VQAProcessor:
    """
    Processor chung cho VQA, kết hợp ViTImageProcessor và AutoTokenizer.
    Xử lý hình ảnh và văn bản thành pixel_values, input_ids, attention_mask, và labels (nếu có).
    """
    def __init__(
        self,
        feature_extractor_name: str = config.image_model,
        tokenizer_name: str = config.text_model,
        max_question_length: int = 24,
        max_answer_length: int = 64,
    ):
        self.feature_extractor = ViTImageProcessor.from_pretrained(feature_extractor_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_question_length = max_question_length
        self.max_answer_length = max_answer_length

    def __call__(
        self,
        images: Union[Image.Image, List[Image.Image]] = None,
        questions: Union[str, List[str]] = None,
        answers: Union[str, List[str]] = None,
        padding: Union[bool, str] = "max_length",
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """
        Xử lý hình ảnh, câu hỏi, và câu trả lời (nếu có).

        Args:
            images: PIL Image hoặc list của PIL Images.
            questions: Chuỗi hoặc list các chuỗi câu hỏi.
            answers: Chuỗi hoặc list các chuỗi câu trả lời (tùy chọn, dùng trong huấn luyện).
            padding: Chế độ padding ("max_length", "longest", hoặc False).
            truncation: Cắt chuỗi nếu vượt quá max_length.
            return_tensors: Loại tensor trả về ("pt" cho PyTorch).

        Returns:
            Dict chứa:
                - pixel_values: [batch_size, channels, height, width]
                - input_ids: [batch_size, question_seq_len]
                - attention_mask: [batch_size, question_seq_len]
                - labels: [batch_size, answer_seq_len] (nếu có answers)
        """
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(questions, str):
            questions = [questions]
        if answers is not None and isinstance(answers, str):
            answers = [answers]

        image_features = self.feature_extractor(
            images=images,
            return_tensors=return_tensors,
        )
        pixel_values = image_features["pixel_values"]  # [batch_size, channels, height, width]

        question_features = self.tokenizer(
            text=questions,
            padding=padding,
            max_length=self.max_question_length,
            truncation=truncation,
            return_tensors=return_tensors,
            return_attention_mask=True,
        )
        input_ids = question_features["input_ids"]  # [batch_size, question_seq_len]
        attention_mask = question_features["attention_mask"]  # [batch_size, question_seq_len]

        output = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Xử lý câu trả lời (nếu có)
        if answers is not None:
            answer_features = self.tokenizer(
                text=answers,
                padding=padding,
                max_length=self.max_answer_length,
                truncation=truncation,
                return_tensors=return_tensors,
            )
            output["labels"] = answer_features["input_ids"]  # [batch_size, answer_seq_len]

        return output
   