import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Union, List
from PIL import Image
from .processor import VQAProcessor
from transformers import AutoModel, AutoTokenizer, ViTModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

class StackAttention(nn.Module):
    def __init__(self, d_model=768, ff_dim=512, dropout=True):
        super(StackAttention, self).__init__()
        self.ff_image = nn.Linear(d_model, ff_dim)
        self.ff_ques = nn.Linear(d_model, ff_dim)
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        self.ff_attention = nn.Linear(ff_dim, d_model)

    def forward(self, image_embed, ques_embed):
        # [N, 768] -> [N, 512]
        h_image = self.ff_image(image_embed)
        # [N, 768] -> [N, 512] -> [N, 1, 512]
        h_question = self.ff_ques(ques_embed)
        # [N, 49, 512]
        h_attn = F.tanh(h_image + h_question)
        if getattr(self, 'dropout'):
            h_attn = self.dropout(h_attn)
        # [N, 49, 512] -> [N, 49, 1] -> [N, 49]
        h_attn = self.ff_attention(h_attn)
        return h_attn

class VQAModel(nn.Module):
    def __init__(self, 
                 text_model: str, 
                 image_model: str,  
                 d_model: int=768, 
                 vocab_size: int=64001,
                 ff_dim: int=512,
                 max_len: int=50,
                 n_head: int=8, 
                 num_layers: int=2, 
                 dropout: float=0.1,
                 device: str = "cpu"
                ):
        super(VQAModel, self).__init__()
        self.max_len = max_len
        self.device = device
        
        # Component models
        self.vit = ViTModel.from_pretrained(image_model)
        for param in self.vit.parameters():
            param.requires_grad = False
            self.vit.eval()

        self.phobert = AutoModel.from_pretrained(text_model)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model)

        self.san_model = nn.ModuleList(
            [StackAttention(d_model=d_model, 
                            ff_dim=ff_dim, 
                            dropout=True)] * num_layers)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model, 
                nhead=n_head,
                dim_feedforward=d_model*4,
                dropout=dropout,
                batch_first=True
            ), 
            num_layers=num_layers)
        
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)

    def _shift_right(self, input_ids):
        """Dịch chuyển input_ids sang phải để tạo decoder_input_ids."""
        pad_token_id = self.tokenizer.pad_token_id
        shifted_input_ids = torch.zeros_like(input_ids)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = pad_token_id  # Hoặc BOS token
        return shifted_input_ids

    def _generate_square_subsequent_mask(self, sz):
        """Tạo mask cho decoder."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask
        
    def forward(self, 
                pixel_values: torch.FloatTensor, 
                input_ids: torch.LongTensor,
                attention_mask: torch.FloatTensor,
                labels: torch.LongTensor = None,
                return_dict: bool = False):
        # [B, 3, 224, 224] -> [B, 768]
        with torch.cuda.amp.autocast(enabled=True):  # Sử dụng mixed precision để tiết kiệm bộ nhớ
            with torch.no_grad():  # Không cần tính gradient của vit trong forward
                vit_outputs = self.vit(pixel_values)
                vit_hidden = vit_outputs.pooler_output
                
                # Giải phóng bộ nhớ các biến trung gian
                del vit_outputs
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # [B, max_len] -> [B, 768]
            phobert_outputs = self.phobert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            phobert_hidden = phobert_outputs.pooler_output 
            
            # Giải phóng bộ nhớ
            del phobert_outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # [N, 768] + [N, 768] => [N, 768]
            fused_hidden = phobert_hidden.clone()  # Khởi tạo
            for att_layer in self.san_model:
                fused_hidden = att_layer(vit_hidden, phobert_hidden)

            fused_hidden = self.tanh(fused_hidden)
            fused_hidden = self.dropout(fused_hidden)

            # START DECODER
            # Chuẩn bị input cho decoder
            if labels is not None:
                decoder_input_ids = self._shift_right(labels)
            else:
                # Trong suy luận, khởi tạo với token bắt đầu (nếu cần)
                decoder_input_ids = torch.ones(
                    (input_ids.size(0), 1), dtype=torch.long, device=input_ids.device
                ) * self.tokenizer.bos_token_id
                
            # Để tiết kiệm bộ nhớ, chia decoder thành các batch nhỏ nếu batch lớn
            batch_size = decoder_input_ids.size(0)
            max_batch_size = 8  # Có thể điều chỉnh dựa trên kích thước GPU
            
            if batch_size > max_batch_size and not self.training:
                # Xử lý theo mini-batch cho validation
                all_logits = []
                for i in range(0, batch_size, max_batch_size):
                    end_idx = min(i + max_batch_size, batch_size)
                    mini_decoder_input_ids = decoder_input_ids[i:end_idx]
                    mini_fused_hidden = fused_hidden[i:end_idx]
                    
                    # Tạo embedding cho decoder
                    mini_decoder_embeds = self.lm_head.weight[mini_decoder_input_ids]
                    
                    # Forward qua decoder
                    mini_decoder_outputs = self.transformer_decoder(
                        tgt=mini_decoder_embeds,
                        memory=mini_fused_hidden.unsqueeze(1),  # [batch_size, 1, hidden_size]
                        tgt_mask=self._generate_square_subsequent_mask(mini_decoder_embeds.size(1)).to(mini_decoder_embeds.device),
                    )
                    
                    # Tính logits
                    mini_logits = self.lm_head(mini_decoder_outputs)
                    all_logits.append(mini_logits)
                    
                    # Giải phóng bộ nhớ
                    del mini_decoder_input_ids, mini_fused_hidden, mini_decoder_embeds, mini_decoder_outputs, mini_logits
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Kết hợp lại logits
                logits = torch.cat(all_logits, dim=0)
                del all_logits
            else:
                # Xử lý bình thường cho training hoặc batch nhỏ
                # Tạo embedding cho decoder
                decoder_embeds = self.lm_head.weight[decoder_input_ids]
                
                decoder_outputs = self.transformer_decoder(
                    tgt=decoder_embeds,
                    memory=fused_hidden.unsqueeze(1),  # [batch_size, 1, hidden_size]
                    tgt_mask=self._generate_square_subsequent_mask(decoder_embeds.size(1)).to(decoder_embeds.device),
                )
                
                # Tính logits
                logits = self.lm_head(decoder_outputs)  # [batch_size, answer_seq_len, vocab_size]
                
                # Giải phóng bộ nhớ cho biến không cần thiết
                del decoder_embeds, decoder_outputs
            
            # Tính loss nếu có labels
            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            if return_dict:
                return CausalLMOutputWithCrossAttentions(
                    loss=loss,
                    logits=logits,
                    hidden_states=None,  # Không trả về hidden_states để tiết kiệm bộ nhớ
                )
            return (loss, logits)
    
    def generate(
        self,
            images: Union[Image.Image, List[Image.Image]],
            questions: Union[str, List[str]],
            processor: VQAProcessor,
            max_length: int = 64,
            **kwargs
        ) -> torch.Tensor:
            """
            Sinh câu trả lời từ hình ảnh và câu hỏi.

            Args:
                images: PIL Image hoặc list của PIL Images.
                questions: Chuỗi hoặc list các chuỗi câu hỏi.
                processor: VQAProcessor để xử lý input.
                max_length: Độ dài tối đa của chuỗi sinh ra.

            Returns:
                torch.Tensor: Token IDs của câu trả lời.
            """
            # Xử lý input bằng processor
            inputs = processor(
                images=images,
                questions=questions,
                padding="max_length",
                return_tensors="pt",
            )
            pixel_values = inputs["pixel_values"].to(self.vit.device)
            input_ids = inputs["input_ids"].to(self.phobert.device)
            attention_mask = inputs["attention_mask"].to(self.phobert.device)

            # Mã hóa hình ảnh
            vit_outputs = self.vit(pixel_values)
            vit_hidden = vit_outputs.pooler_output  # [batch_size, hidden_size]

            # Mã hóa câu hỏi
            phobert_outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
            phobert_hidden = phobert_outputs.pooler_output  # [batch_size, hidden_size]

            # Kết hợp đặc trưng
            combined_hidden = vit_hidden
            for fusion_model in self.san_model: 
                combined_hidden = fusion_model(vit_hidden, phobert_hidden)
            
            # [N, 768] -> [N, 768]
            fused_hidden = self.tanh(combined_hidden)
            fused_hidden = self.dropout(fused_hidden)

            # Sinh chuỗi
            generated_ids = [self.tokenizer.bos_token_id]
            decoder_input_ids = torch.tensor([[self.tokenizer.bos_token_id]], device=input_ids.device)
            
            for _ in range(max_length):
                decoder_embeds = self.lm_head.weight[decoder_input_ids]
                decoder_outputs = self.transformer_decoder(
                    tgt=decoder_embeds,
                    memory=fused_hidden.unsqueeze(1),
                    tgt_mask=self._generate_square_subsequent_mask(decoder_input_ids.size(1)).to(decoder_input_ids.device),
                )
                logits = self.lm_head(decoder_outputs[:, -1, :])
                next_token = torch.argmax(logits, dim=-1)
                generated_ids.append(next_token.item())
                decoder_input_ids = torch.cat([decoder_input_ids, next_token.unsqueeze(0)], dim=1)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
            
            return torch.tensor(generated_ids, device=input_ids.device)
