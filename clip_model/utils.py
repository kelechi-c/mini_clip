import torch
import cv2
import numpy as np
import torch.nn.functional as func_nn
from config import config

# configs
img_size = 200 if torch.cuda.is_available() else 128


def read_image(img):
    img_array = np.array(img)
    img = cv2.resize(img_array, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    return img


def get_similarities(
    img_feats: torch.Tensor, text_tokens: torch.Tensor, temp=config.temperature
):
    img_similarity = img_feats @ img_feats.T
    text_similarity = text_tokens @ text_tokens.T

    targets = func_nn.softmax((img_similarity + text_similarity) / (2 * temp), dim=-1)

    return targets


def custom_cross_entropy(target, pred):
    softmax_loss = torch.nn.LogSoftmax(dim=-1)
    loss = (-target * softmax_loss(pred)).sum(1)

    return loss
