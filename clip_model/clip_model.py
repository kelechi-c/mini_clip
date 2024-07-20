import torch
import timm
from torch import nn
from torchvision import models
from transformers import AutoTokenizer, AutoModelForTextEncoding
from config import config


tokenizer = AutoTokenizer.from_pretrained(config.text_encoder_id)


class ImageEncoder(nn.Module):
    def __init__(self, config: config, pretrained=True):
        super().__init__()
        self.resnet = timm.create_model(
            model_name="resnet50", pretrained=True, num_classes=0, pool_layer="avg"
        )

        for par in self.resnet.parameters():
            par.requires_grad = True

    def forward(self, image):
        x = self.resnet(image)
        return x
