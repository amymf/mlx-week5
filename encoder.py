import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads):
        super(EncoderLayer, self).__init__()
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.self_attn = nn.MultiheadAttention(model_dim, num_heads)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim)
        )

    def forward(self, x):
        # x shape (batch_size, n_frames, model_dim)
        x = self.layer_norm1(x)
        attn_output, _ = self.self_attn(x, x, x)
        x = x + attn_output
        # x shape (batch_size, n_frames, model_dim)
        x = self.layer_norm2(x)
        ff_output = self.feed_forward(x)
        x = x + ff_output
        return x


class UrbanSoundTransformerEncoder(nn.Module):
    def __init__(
        self,
        n_mels=128,
        duration=4,
        sr=22050,
        hop_length=512,
        num_classes=10,
        num_layers=4,
        model_dim=256,
        num_heads=4,
    ):
        super(UrbanSoundTransformerEncoder, self).__init__()

        self.conv1 = nn.Conv1d(n_mels, model_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(model_dim, model_dim, kernel_size=3, stride=1, padding=1)
        self.layers = nn.ModuleList(
            [EncoderLayer(model_dim, num_heads) for _ in range(num_layers)]
        )
        self.n_frames = int(sr * duration / hop_length)
        self.positional_embedding = nn.Parameter(
            torch.randn(1, self.n_frames + 1, model_dim)
        )
        self.layer_norm = nn.LayerNorm(
            model_dim
        )  # Final layer normalization because we use pre-norm
        self.fc = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        # x shape (batch_size, num_channels, n_mels, n_frames)
        x = x.squeeze(1)
        # x shape (batch_size, n_mels, n_frames)
        x = F.gelu(self.conv1(x))
        # x shape (batch_size, model_dim, n_frames)
        x = F.gelu(self.conv2(x))
        # x shape (batch_size, model_dim, n_frames)
        x = x.permute(0, 2, 1)  # reorder to (batch_size, n_frames, model_dim)
        assert (
            x.shape[1:] == self.positional_embedding.shape[1:]
        ), f"incorrect audio shape, {self.positional_embedding.shape[1:]} expected, got {x.shape[1:]}"
        x = x + self.positional_embedding[:, : x.shape[1], :]
        # x shape (batch_size, n_frames, model_dim)
        for layer in self.layers:
            x = layer(x)
        # x shape (batch_size, n_frames, model_dim)
        x = self.layer_norm(x)
        x = x.mean(dim=1)  # mean across frames
        # x shape (batch_size, model_dim)
        x = self.fc(x)
        # x shape (batch_size, num_classes)
        return x
