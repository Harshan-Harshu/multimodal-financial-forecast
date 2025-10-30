import torch
import torch.nn as nn

class MultimodalFusion(nn.Module):
    def __init__(self, ts_dim, text_dim):
        super(MultimodalFusion, self).__init__()
        self.fc_ts = nn.Linear(ts_dim, 64)
        self.fc_text = nn.Linear(text_dim, 64)
        self.attention = nn.MultiheadAttention(64, num_heads=4, batch_first=True)
        self.output_layer = nn.Linear(64, 1)

    def forward(self, ts_feat, text_feat):
        ts_proj = self.fc_ts(ts_feat).unsqueeze(1)
        text_proj = self.fc_text(text_feat).unsqueeze(1)
        fusion_input = torch.cat((ts_proj, text_proj), dim=1)
        attn_output, _ = self.attention(fusion_input, fusion_input, fusion_input)
        out = self.output_layer(attn_output[:, -1, :])
        return out
