import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoImageProcessor,
    ViTModel,
    MBartForConditionalGeneration,
)
from transformers.modeling_outputs import BaseModelOutput

class ImageEncoder(nn.Module):
    """Mã hóa hình ảnh sử dụng Vision Transformer"""
    def __init__(self, output_dim=768):
        super().__init__()
        # Sử dụng ViT pre-trained
        self.model_name = "google/vit-base-patch16-224-in21k"
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = ViTModel.from_pretrained(self.model_name)
        if output_dim != 768:
            self.projection = nn.Linear(768, output_dim)
        else:
            self.projection = nn.Identity()

        for param in list(self.model.parameters())[:-4]:  # Đóng băng tất cả trừ 4 lớp cuối
            param.requires_grad = False

    def forward(self, images, device):
        if isinstance(images, list):
            inputs = self.processor(images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
        else:
            inputs = self.processor(images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

        # Lấy embedding từ ViT
        outputs = self.model(**inputs)
        image_embeddings = outputs.last_hidden_state[:, 0, :]
        image_embeddings = self.projection(image_embeddings)

        return image_embeddings


class QuestionEncoder(nn.Module):
    """Mã hóa câu hỏi sử dụng PhoBERT"""
    def __init__(self, output_dim=768):
        super().__init__()
        self.model_name = "vinai/phobert-base-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

        # Thêm LSTM để xử lý chuỗi đầu ra từ PhoBERT
        self.lstm = nn.LSTM(
            input_size=self.model.config.hidden_size,
            hidden_size=output_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Lớp projection để kết hợp 2 chiều của LSTM
        self.projection = nn.Linear(output_dim * 2, output_dim)

        # Đóng băng một phần mô hình PhoBERT
        for param in list(self.model.parameters())[:-2]:  # Đóng băng tất cả trừ 2 lớp cuối
            param.requires_grad = False

    def forward(self, questions, device):
        # Tokenize câu hỏi
        inputs = self.tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=50
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Lấy embedding từ PhoBERT
        outputs = self.model(**inputs)

        # Đưa qua LSTM
        lstm_output, (hidden, _) = self.lstm(outputs.last_hidden_state)

        # Kết hợp 2 chiều của LSTM
        hidden = torch.cat([hidden[0], hidden[1]], dim=1)
        question_embeddings = self.projection(hidden)

        return question_embeddings


class VQAModel(nn.Module):
    """Mô hình VQA kết hợp encoder hình ảnh, encoder câu hỏi và decoder câu trả lời"""
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model_name = "vinai/bartpho-syllable-base"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.bartpho = MBartForConditionalGeneration.from_pretrained(self.model_name)

        self.image_encoder = ImageEncoder(output_dim=hidden_dim)
        self.question_encoder = QuestionEncoder(output_dim=hidden_dim)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        bartpho_dim = self.bartpho.config.d_model
        if hidden_dim != bartpho_dim:
            self.adapter = nn.Linear(hidden_dim, bartpho_dim)
        else:
            self.adapter = nn.Identity()

    def encode_inputs(self, images, questions):
        """Mã hóa đầu vào thành các vector đặc trưng"""
        image_embeddings = self.image_encoder(images, self.device)
        question_embeddings = self.question_encoder(questions, self.device)
        combined = torch.cat([image_embeddings, question_embeddings], dim=1)
        fused = self.fusion(combined)
        batch_size = fused.size(0)
        seq_length = 8
        sequence = fused.unsqueeze(1).expand(-1, seq_length, -1)
        sequence_t = sequence.transpose(0, 1)
        attn_output, _ = self.cross_attention(sequence_t, sequence_t, sequence_t)
        attn_output = attn_output.transpose(0, 1)
        attn_output = self.layer_norm(attn_output + sequence)
        attn_output = self.adapter(attn_output)

        return attn_output

    def forward(self, images, questions, labels=None):
        """Forward pass của mô hình"""
        encoder_hidden_states = self.encode_inputs(images, questions)
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
        decoder_input_ids = None
        outputs = self.bartpho(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )

        return outputs

    def generate_answer(self, image, question, max_length=50, num_beams=4):
        """Sinh câu trả lời cho một cặp hình ảnh-câu hỏi"""
        device = next(self.parameters()).device
        self.eval()

        with torch.no_grad():
            images = [image] if not isinstance(image, list) else image
            questions = [question] if not isinstance(question, list) else question
            encoder_hidden_states = self.encode_inputs(images, questions)
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
            outputs = self.bartpho.generate(
                encoder_outputs=encoder_outputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                temperature=1.0,
                do_sample=True,    # Sử dụng sampling
                top_p=0.9,         # Nucleus sampling
                no_repeat_ngram_size=2  # Tránh lặp lại n-gram
            )
            answer = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ).strip()

            return answer