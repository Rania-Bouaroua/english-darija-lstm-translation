"""
Encoder-decoder (sequence-to-sequence) translation model built on the custom cells.

  English tokens -> Embedding -> [custom LSTM layers] -> final (h, c) of every layer
                                                          |  (initial state of the decoder)
  <bos> y1 y2 ... -> Embedding -> [custom LSTM layers] -> Linear -> softmax over Darija vocab

* Encoder and decoder use the SAME cell type, so the comparison isolates the cell.
* Source padding is handled by a mask: on padded steps the state is carried over, so the
  state given to the decoder is the state at the last real source token (EOS).
* Training uses teacher forcing (ratio 1.0 by default: the decoder input is the gold previous
  token). With ratio 1.0 the decoder can be unrolled layer by layer over the whole target,
  which lets every layer pre-compute its input projections (faster). Ratio < 1 switches to
  step-by-step decoding where the model's own prediction is sometimes fed back.
* Inference: greedy autoregressive decoding, stops when every sentence has produced <eos>.
"""
import random

import torch
import torch.nn as nn

from .cells import RecurrentLayer, build_cell
from .data import BOS_ID, EOS_ID, PAD_ID


class Encoder(nn.Module):
    def __init__(self, vocab, emb, hidden, layers, cell, dropout, proj=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb, padding_idx=PAD_ID)
        self.layers, in_size = nn.ModuleList(), emb
        for _ in range(layers):
            c = build_cell(cell, in_size, hidden, proj)
            self.layers.append(RecurrentLayer(c))
            in_size = c.output_size
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        mask = (src != PAD_ID).float()
        x, states = self.dropout(self.embedding(src)), []
        for layer in self.layers:
            x, st = layer(x, mask)
            states.append(st)
            x = self.dropout(x)
        return states


class Decoder(nn.Module):
    def __init__(self, vocab, emb, hidden, layers, cell, dropout, proj=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb, padding_idx=PAD_ID)
        self.layers, in_size = nn.ModuleList(), emb
        for _ in range(layers):
            c = build_cell(cell, in_size, hidden, proj)
            self.layers.append(RecurrentLayer(c))
            in_size = c.output_size
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(in_size, vocab)

    def forward_teacher(self, tgt_in, states):
        """Full teacher forcing, layer-wise unrolling. tgt_in: (B, T)."""
        mask = (tgt_in != PAD_ID).float()
        x = self.dropout(self.embedding(tgt_in))
        for layer, st in zip(self.layers, states):
            x, _ = layer(x, mask, st)
            x = self.dropout(x)
        return self.out(x)

    def step(self, token, states):
        """One decoding step for all layers. token: (B,)."""
        x, new_states = self.dropout(self.embedding(token)), []
        for layer, st in zip(self.layers, states):
            cell = layer.cell
            x, st = cell(cell.W_x(x), st)
            new_states.append(st)
            x = self.dropout(x)
        return self.out(x), new_states


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, cell="vanilla", emb=256, hidden=256, layers=1,
                 dropout=0.3, proj=None):
        super().__init__()
        self.encoder = Encoder(src_vocab, emb, hidden, layers, cell, dropout, proj)
        self.decoder = Decoder(tgt_vocab, emb, hidden, layers, cell, dropout, proj)

    def forward(self, src, tgt, teacher_forcing=1.0):
        """Returns logits for tgt[:, 1:] (B, T-1, V)."""
        states = self.encoder(src)
        tgt_in = tgt[:, :-1]
        if teacher_forcing >= 1.0:
            return self.decoder.forward_teacher(tgt_in, states)
        logits, token = [], tgt_in[:, 0]
        for t in range(tgt_in.size(1)):
            lg, states = self.decoder.step(token, states)
            logits.append(lg)
            if t + 1 < tgt_in.size(1):
                token = tgt_in[:, t + 1] if random.random() < teacher_forcing else lg.argmax(-1)
        return torch.stack(logits, 1)

    @torch.no_grad()
    def greedy(self, src, max_len=50):
        """Greedy autoregressive decoding. Returns a list of token-id lists (without BOS/EOS)."""
        states = self.encoder(src)
        B = src.size(0)
        token = torch.full((B,), BOS_ID, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        out = []
        for _ in range(max_len):
            lg, states = self.decoder.step(token, states)
            token = lg.argmax(-1)
            out.append(token)
            done |= token == EOS_ID
            if bool(done.all()):
                break
        out = torch.stack(out, 1).tolist()
        res = []
        for seq in out:
            res.append(seq[:seq.index(EOS_ID)] if EOS_ID in seq else seq)
        return res


def count_parameters(model):
    n = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
    return {"total": n(model),
            "encoder_recurrent": n(model.encoder.layers),
            "decoder_recurrent": n(model.decoder.layers),
            "embeddings": n(model.encoder.embedding) + n(model.decoder.embedding),
            "output_layer": n(model.decoder.out)}


def init_weights(model, scheme="default", forget_bias=0.0):
    """Initialisation of the recurrent cells only.
    default: PyTorch nn.Linear default | xavier: Xavier-uniform, zero biases |
    orthogonal: orthogonal per gate block for recurrent matrices + Xavier for inputs, zero biases.
    forget_bias: constant added to the forget-gate bias (e.g. 1.0)."""
    stacks = list(model.encoder.layers) + list(model.decoder.layers)
    if scheme != "default":
        for layer in stacks:
            for name, p in layer.cell.named_parameters():
                if p.dim() == 1:
                    nn.init.zeros_(p)
                elif scheme == "orthogonal" and (name.startswith("W_h") or name.startswith("W_c")):
                    rows, cols = p.shape
                    blocks = p.data.split(cols, 0) if rows % cols == 0 else [p.data]
                    for b in blocks:
                        nn.init.orthogonal_(b)
                elif scheme in ("xavier", "orthogonal"):
                    nn.init.xavier_uniform_(p)
                else:
                    raise ValueError(f"unknown init scheme {scheme}")
    if forget_bias:
        with torch.no_grad():
            for layer in stacks:
                layer.cell.W_x.bias[layer.cell.forget_slice] += forget_bias
