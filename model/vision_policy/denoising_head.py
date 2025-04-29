import torch
import torch.nn as nn


class DenoisingHead(nn.Module):
    """Simple MLP head to predict the denoised actions."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: x [B, T, D_transformer_output]
        Output: [B, T, D_action]
        """
        return self.net(x)

# Alternative: Transformer-based Denoising Head (more complex)
# class TransformerDenoisingHead(nn.Module):
#     def __init__(self, input_dim, output_dim, nhead=8, num_layers=2):
#         super().__init__()
#         decoder_layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=nhead, batch_first=True)
#         self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
#         self.fc_out = nn.Linear(input_dim, output_dim)
#
#     def forward(self, tgt, memory):
#         # tgt: [B, T_action, D]
#         # memory: [B, T_context, D]
#         output = self.transformer_decoder(tgt, memory)
#         return self.fc_out(output)
