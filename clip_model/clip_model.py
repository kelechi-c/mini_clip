import torch
import timm
from torch import nn
from config import config
from transformers import BertModel


# CLIP model components
# - text encoder, image encoder, projection layer


class ImageEncoder(nn.Module):  # resnet for image encoding/feature extraction
    def __init__(self):
        super().__init__()
        self.resnet = timm.create_model(
            model_name="resnet50", pretrained=True, num_classes=0, global_pool="avg"
        )

        for par in self.resnet.parameters():
            par.requires_grad = True

    def forward(self, image: torch.Tensor):
        x = self.resnet(image)
        return x


class TextEncoder(nn.Module):  # BERT model class for text encoding
    def __init__(self, config: config):
        super().__init__()
        self.encoder = BertModel.from_pretrained(config.text_encoder_id)

        for par in self.encoder.parameters():
            par.requires_grad = True

        self.cls_token = 0

    def forward(self, input: torch.Tensor, attn_mask):
        out_tokens = self.encoder(input_ids=input, attention_mask=attn_mask)
        hidden_state = out_tokens.last_hidden_state

        return hidden_state[:, self.cls_token, :]


class ProjectLayer(nn.Module):  # projection head
    def __init__(
        self, embed_dim, project_dim: int = 256, dropout: float = config.dropout
    ):
        super().__init__()
        self.project_layer = nn.Linear(embed_dim, project_dim)

        self.linear = nn.Sequential(
            nn.GELU(), nn.Linear(project_dim, project_dim), nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(project_dim)

    def forward(self, input):
        # project vector dim to 256 embedding space
        project = self.project_layer(input)
        x = self.linear(project)  # apply gelu, linear layer and dropout

        x = x + project  # residual connection

        x = self.layer_norm(x)  # apply layer normalization

        return x
