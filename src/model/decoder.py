import torch
from .componants import PositionalEncoding


class TransformerDecoderBlock(torch.nn.Module):
    def __init__(self, attn_embed_dims, num_heads, ff_dim=512) -> None:
        super().__init__()
        self.self_multi_head_attn = torch.nn.MultiheadAttention(
            attn_embed_dims, num_heads, batch_first=True
        )
        self.cross_multi_head_attn = torch.nn.MultiheadAttention(
            attn_embed_dims, num_heads, batch_first=True
        )
        self.norm_1 = torch.nn.LayerNorm(attn_embed_dims)
        self.norm_2 = torch.nn.LayerNorm(attn_embed_dims)
        self.norm_3 = torch.nn.LayerNorm(attn_embed_dims)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(attn_embed_dims, ff_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(ff_dim, attn_embed_dims),
        )

    def forward(self, x, encoder_output, casual_mask=True):
        attn_out, _ = self.self_multi_head_attn(x, x, x, casual_mask=casual_mask)
        attn_out = self.norm_1(x + attn_out)
        cross_attn_out, _ = self.cross_multi_head_attn(
            encoder_output, encoder_output, attn_out
        )
        cross_attn_out = self.norm_2(attn_out + cross_attn_out)
        ff_out = self.ff(cross_attn_out)
        ff_out = self.norm_3(ff_out + cross_attn_out)

        return ff_out


class Decoder(torch.nn.Module):
    def __init__(self, vocab_size, out_dims, num_heads=8, num_blocks=4) -> None:
        super().__init__()
        self.embedding_layer = torch.nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=out_dims
        )
        self.pos_enc = PositionalEncoding(out_dims)
        self.decoder_blocks = torch.nn.ModuleList(
            [TransformerDecoderBlock(out_dims, num_heads) for i in range(num_blocks)]
        )
        self.output_layer = torch.nn.ModuleList(
            [
                torch.nn.Linear(in_features=out_dims, out_features=vocab_size),
                torch.nn.Softmax(dim=vocab_size),
            ]
        )

    def forward(self, x, encoder_output):
        x = self.embedding_layer(x)
        x = self.pos_enc(x)
        for decoder in self.decoder_blocks:
            x = decoder(x, encoder_output)

        output = self.output_layer(x)
        return output
