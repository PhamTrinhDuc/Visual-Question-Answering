from model.vqa_model import VQAModel
from config import config


model = VQAModel(
  text_model=config.text_model, 
  image_model=config.image_model, 
  d_model=config.D_MODEL, 
  vocab_size=config.VOCAB_SIZE, 
  ff_dim=config.FF_DIM,
  max_len=config.MAX_LEN,
  n_head=config.NUM_HEADS,
  num_layers=config.DECODER_LAYERS,
  dropout=config.DROPOUT,
  device=config.DEVICE
)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(f'The model has {count_parameters(model):,} trainable parameters.')
# print(f'The model image has {count_parameters(model.image_encoder):,} trainable parameters.')
print(f'The model text has {count_parameters(model.text_encoder):,} trainable parameters.')
