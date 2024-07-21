import torch
from torch import optim
from tqdm.auto import tqdm
from clip_model import ClipModel
from dataload import train_loader, valid_loader
from config import config

clip_model = ClipModel()
optimizer = optim.Adam(clip_model.parameters(), lr=1e-4)


def compute_loss(t_embed, img_embed, temp=clip_model.temp):
    logits = (t_embed @ img_embed.T) / temp


def train_step(batch, model):
    pass


def eval_step():
    pass


def train_loop(batch, epochs, model, dataloader):
    pass
