import os
import torch

class config:
    train_csv = './dataset/csv/train_data.csv'
    dev_csv = './dataset/csv/dev_data.csv'
    test_csv = './dataset/csv/test_data.csv'
    image_model = 'google/vit-base-patch16-224-in21k'
    text_model = "vinai/phobert-base"
    vncore_path = './dataset/VnCoreNLP-1.2.jar'
    imgmodel_dir = 'hf-hub:timm/convnext_small.fb_in22k_ft_in1k_384'
    FOLDER_IMAGE_PATH = './dataset/images'
    IMAGE_SIZE = 224
    SEED = 42
    EPOCHS = 7
    MAX_LEN = 64
    BATCH_SIZE = 4
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    NUM_WORKERS = os.cpu_count()
    
    # Model architecture parameters
    D_MODEL = 768  # Size of hidden layers/embeddings
    FF_DIM = 512      # Feed-forward dimension in attention
    VOCAB_SIZE = 64001  # Vocabulary size for answer generation
    DECODER_LAYERS = 2  # Number of transformer decoder layers
    NUM_HEADS = 8      # Number of attention heads
    DROPOUT = 0.1      # Dropout probability
    
torch.manual_seed(config.SEED)
torch.cuda.manual_seed(config.SEED)