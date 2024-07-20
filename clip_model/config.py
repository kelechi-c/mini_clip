import torch


class config:
    batch_size = 32
    project_dim = 256
    temperature = 0.1
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder_id = "distilbert-base-uncased"
    img_size = 200
