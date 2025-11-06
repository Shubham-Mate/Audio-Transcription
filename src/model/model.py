import torch
from .encoder import Encoder
from .decoder import Decoder


class TransformerModel(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        inp_dims: int,
        out_dims: int,
        num_heads: int = 8,
        num_blocks: int = 4,
        pad_token: int = 0,
    ):
        super().__init__()
        self.encoder = Encoder(inp_dims, out_dims, num_heads, num_blocks)
        self.decoder = Decoder(vocab_size, out_dims, num_heads, num_blocks)
        self.pad_token = pad_token

    def forward(self, encoder_inp, decoder_inp, encoder_mask):
        """
        Forward pass for training (uses teacher forcing)
        encoder_inp: Tensor (batch, seq_len_enc, inp_dims)
        decoder_inp: Tensor (batch, seq_len_dec)
        """
        encoder_output = self.encoder(encoder_inp, padding_mask=encoder_mask)
        output = self.decoder(decoder_inp, encoder_output)
        return output

    @torch.no_grad()
    def generate(
        self, encoder_inp, start_token, end_token, encoder_mask=None, max_len=50
    ):
        """
        Autoregressive generation:
        - Pass input once through encoder
        - Iteratively feed decoder outputs back into itself
        """
        device = next(self.parameters()).device

        # Pass through encoder once
        encoder_output = self.encoder(encoder_inp, padding_mask=encoder_mask)

        # Start with batch of start tokens
        batch_size = encoder_inp.size(0)

        generated = torch.full(
            (batch_size, max_len), start_token, dtype=torch.long, device=device
        )
        for i in range(1, max_len):
            out = self.decoder(generated[:, :i], encoder_output)
            logits = out[:, -1, :]
            next_token = torch.argmax(logits, dim=-1)
            generated[:, i] = next_token
            if torch.all(next_token == end_token):
                break

        return generated
