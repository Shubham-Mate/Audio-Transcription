import torch
from .componants import PositionalEncoding


class TransformerEncoderBlock(torch.nn.Module):
    def __init__(self, attn_embed_dims, num_heads, ff_dim=512) -> None:
        super().__init__()
        self.multi_head_attn = torch.nn.MultiheadAttention(
            attn_embed_dims, num_heads, batch_first=True
        )
        self.norm_1 = torch.nn.LayerNorm(attn_embed_dims)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(attn_embed_dims, ff_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(ff_dim, attn_embed_dims),
        )
        self.norm_2 = torch.nn.LayerNorm(attn_embed_dims)

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.multi_head_attn(x, x, x, key_padding_mask=key_padding_mask)
        attn_out = self.norm_1(x + attn_out)

        ff_out = self.ff(attn_out)
        ff_out = self.norm_2(ff_out + attn_out)

        return ff_out


class Encoder(torch.nn.Module):
    def __init__(self, inp_dims, out_dims, num_heads=8, num_blocks=4) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(inp_dims, out_dims)
        self.pos_enc = PositionalEncoding(out_dims)
        self.encoder_blocks = torch.nn.ModuleList(
            [TransformerEncoderBlock(out_dims, num_heads) for i in range(num_blocks)]
        )

    def forward(self, x, padding_mask=None):
        x = self.proj(x)
        x = self.pos_enc(x)
        for encoder in self.encoder_blocks:
            x = encoder(x, key_padding_mask=padding_mask)

        return x
