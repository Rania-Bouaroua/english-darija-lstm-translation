"""
LSTM cells implemented from scratch (shared by encoder and decoder) with nn.Linear (no nn.LSTM), following the
equations of the project appendix.

Every cell exposes the same interface:
    cell.W_x(x)               -> input projections for ALL gates (computed once for the
                                 whole sequence, outside the time loop, for speed)
    cell(xw_t, state)         -> (output_t, new_state)
    cell.init_state(B, dev)   -> initial (h, c)
    cell.output_size          -> size of h_t (H, or P for LSTMP)
    cell.forget_slice         -> where the forget-gate bias lives in W_x.bias

Each gate has exactly one bias: it lives in W_x, and W_h has bias=False,
so the parametrisation matches the equations exactly.
"""
import torch
import torch.nn as nn


class VanillaLSTMCell(nn.Module):
    """Equations (1)-(6). Gate order inside the fused matrices: i, f, g, o."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        H = hidden_size
        self.hidden_size = H
        self.output_size = H
        self.W_x = nn.Linear(input_size, 4 * H)          # W_xi, W_xf, W_xg, W_xo + b
        self.W_h = nn.Linear(H, 4 * H, bias=False)       # W_hi, W_hf, W_hg, W_ho
        self.forget_slice = slice(H, 2 * H)

    def init_state(self, batch_size, device):
        z = torch.zeros(batch_size, self.hidden_size, device=device)
        return (z, z.clone())

    def forward(self, xw, state):
        h, c = state
        i, f, g, o = (xw + self.W_h(h)).chunk(4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g                                # (5)
        h = o * torch.tanh(c)                            # (6)
        return h, (h, c)


class WMCLSTMCell(nn.Module):
    """Working Memory Connections, equations (7)-(12).
    The cell state feeds the gates through a learned, tanh-squashed connection.
    i_t and f_t look at c_{t-1}; o_t looks at the *new* c_t, so c_t must be computed
    before the output gate."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        H = hidden_size
        self.hidden_size = H
        self.output_size = H
        self.W_x = nn.Linear(input_size, 4 * H)          # i, f, g, o
        self.W_h = nn.Linear(H, 4 * H, bias=False)
        self.W_c_if = nn.Linear(H, 2 * H, bias=False)    # W_ci, W_cf
        self.W_co = nn.Linear(H, H, bias=False)          # W_co
        self.forget_slice = slice(H, 2 * H)

    def init_state(self, batch_size, device):
        z = torch.zeros(batch_size, self.hidden_size, device=device)
        return (z, z.clone())

    def forward(self, xw, state):
        h, c_prev = state
        gi, gf, gg, go = (xw + self.W_h(h)).chunk(4, dim=-1)
        ci, cf = self.W_c_if(c_prev).chunk(2, dim=-1)
        i = torch.sigmoid(gi + torch.tanh(ci))           # (7)
        f = torch.sigmoid(gf + torch.tanh(cf))           # (8)
        g = torch.tanh(gg)                               # (10)
        c = f * c_prev + i * g                           # (11)
        o = torch.sigmoid(go + torch.tanh(self.W_co(c)))  # (9) uses c_t
        h = o * torch.tanh(c)                            # (12)
        return h, (h, c)


class CIFGLSTMCell(nn.Module):
    """Coupled Input-Forget Gate, equations (13)-(18): i_t = 1 - f_t.
    Only 3 gate blocks -> ~25% fewer recurrent parameters than vanilla.
    Gate order: f, g, o."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        H = hidden_size
        self.hidden_size = H
        self.output_size = H
        self.W_x = nn.Linear(input_size, 3 * H)
        self.W_h = nn.Linear(H, 3 * H, bias=False)
        self.forget_slice = slice(0, H)

    def init_state(self, batch_size, device):
        z = torch.zeros(batch_size, self.hidden_size, device=device)
        return (z, z.clone())

    def forward(self, xw, state):
        h, c = state
        f, g, o = (xw + self.W_h(h)).chunk(3, dim=-1)
        f = torch.sigmoid(f)                             # (13)
        i = 1.0 - f                                      # (14)
        o = torch.sigmoid(o)                             # (15)
        g = torch.tanh(g)                                # (16)
        c = f * c + i * g                                # (17)
        h = o * torch.tanh(c)                            # (18)
        return h, (h, c)


class LSTMPCell(nn.Module):
    """LSTM with projection, equations (19)-(25).
    c_t, m_t in R^H ; h_t = W_p m_t + b_p in R^P (P < H).
    The recurrence uses the small h_t, so W_h is (4H x P) instead of (4H x H)."""

    def __init__(self, input_size, hidden_size, proj_size):
        super().__init__()
        H, P = hidden_size, proj_size
        self.hidden_size, self.proj_size = H, P
        self.output_size = P
        self.W_x = nn.Linear(input_size, 4 * H)          # i, f, g, o
        self.W_h = nn.Linear(P, 4 * H, bias=False)
        self.W_p = nn.Linear(H, P)                       # projection (25)
        self.forget_slice = slice(H, 2 * H)

    def init_state(self, batch_size, device):
        return (torch.zeros(batch_size, self.proj_size, device=device),
                torch.zeros(batch_size, self.hidden_size, device=device))

    def forward(self, xw, state):
        h, c = state
        i, f, g, o = (xw + self.W_h(h)).chunk(4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g                                # (23)
        m = o * torch.tanh(c)                            # (24)
        h = self.W_p(m)                                  # (25)
        return h, (h, c)


CELLS = {"vanilla": VanillaLSTMCell, "wmc": WMCLSTMCell,
         "cifg": CIFGLSTMCell, "lstmp": LSTMPCell}


def build_cell(cell_type, input_size, hidden_size, proj_size=None):
    if cell_type == "lstmp":
        return LSTMPCell(input_size, hidden_size, proj_size or hidden_size // 2)
    return CELLS[cell_type](input_size, hidden_size)


class RecurrentLayer(nn.Module):
    """Unrolls a cell over time. Padding is handled with a mask: on padded steps
    the state is simply carried over, so the final state is the state at the last
    *real* token of each sequence."""

    def __init__(self, cell):
        super().__init__()
        self.cell = cell

    def forward(self, x, mask, state=None):
        # x: (B, T, D), mask: (B, T) float {0,1}; state: optional initial (h, c)
        B, T, _ = x.shape
        if state is None:
            state = self.cell.init_state(B, x.device)
        xw = self.cell.W_x(x)                            # all input projections at once
        outputs = []
        for t in range(T):
            _, new_state = self.cell(xw[:, t], state)
            m = mask[:, t].unsqueeze(-1)
            state = tuple(m * ns + (1.0 - m) * s for ns, s in zip(new_state, state))
            outputs.append(state[0])
        return torch.stack(outputs, dim=1), state
