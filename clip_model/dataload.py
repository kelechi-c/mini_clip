import torch
import math
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from utils import read_image
from transformers import AutoTokenizer
from config import config

# latex_data_id = 'OleehyO/latex-formulas'
imagenet_id = "visual-layer/imagenet-1k-vl-enriched"
train_split = 0.9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(config.text_encoder_id)

# load dataset from huggingface
dataset = load_dataset(imagenet_id, split="train")

images = dataset["image"]
captions = dataset["caption_enriched"]

print(f"data type of samples: image => {images[0]}, captions => {images[0]}")
print(f"count: {len(images)} images, and {len(captions)} text captions")


class CaptionDataset(Dataset):
    def __init__(self, images, captions, tokenizer: AutoTokenizer, config: config):
        super().__init__()
        self.captions = captions
        self.images = images
        self.encoded_captions = tokenizer.encode(
            captions, padding=True, truncation=True, max_len=config.max_len
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        image = read_image(image)
        image = torch.tensor(image).to(device)

        caption = self.captions[index]
        caption = torch.tensor(caption).to(device)

        return image, caption


caption_dataset = CaptionDataset(images, captions, tokenizer, config)

train_size = math.floor(len(caption_dataset) * train_split)
val_size = len(caption_dataset) - train_size

train_data, valid_data = random_split(caption_dataset, (train_size, val_size))

train_loader = DataLoader(
    train_data, batch_size=config.batch_size, shuffle=True)

valid_loader = DataLoader(
    valid_data, batch_size=config.batch_size, shuffle=False)
