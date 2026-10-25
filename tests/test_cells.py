"""Sanity checks. nn.LSTM is used ONLY as a reference to verify our from-scratch cell.
Run:  python -m tests.test_cells"""
import warnings
warnings.filterwarnings("ignore", message=".*oneDNN.*")
import torch
import torch.nn as nn

from src.cells import (CIFGLSTMCell, LSTMPCell, RecurrentLayer, VanillaLSTMCell,
                       WMCLSTMCell)

torch.manual_seed(0)
B, T, D, H, P = 3, 7, 5, 6, 4


def test_vanilla_matches_torch():
    cell = VanillaLSTMCell(D, H)
    ref = nn.LSTM(D, H, batch_first=True)          # same gate order i, f, g, o
    with torch.no_grad():
        ref.weight_ih_l0.copy_(cell.W_x.weight); ref.bias_ih_l0.copy_(cell.W_x.bias)
        ref.weight_hh_l0.copy_(cell.W_h.weight); ref.bias_hh_l0.zero_()
    x = torch.randn(B, T, D)
    out, (h, c) = RecurrentLayer(cell)(x, torch.ones(B, T))
    out_ref, (h_ref, c_ref) = ref(x)
    assert torch.allclose(out, out_ref, atol=1e-6) and torch.allclose(c, c_ref[0], atol=1e-6)
    print("vanilla == nn.LSTM  OK")


def test_lstmp_matches_torch():
    cell = LSTMPCell(D, H, P)
    with torch.no_grad():
        cell.W_p.bias.zero_()                      # nn.LSTM's projection has no bias
    ref = nn.LSTM(D, H, batch_first=True, proj_size=P)
    with torch.no_grad():
        ref.weight_ih_l0.copy_(cell.W_x.weight); ref.bias_ih_l0.copy_(cell.W_x.bias)
        ref.weight_hh_l0.copy_(cell.W_h.weight); ref.bias_hh_l0.zero_()
        ref.weight_hr_l0.copy_(cell.W_p.weight)
    x = torch.randn(B, T, D)
    out, _ = RecurrentLayer(cell)(x, torch.ones(B, T))
    assert torch.allclose(out, ref(x)[0], atol=1e-6)
    print("lstmp (b_p=0) == nn.LSTM(proj_size)  OK")


def test_masking_gives_last_real_state():
    for cls in (VanillaLSTMCell, WMCLSTMCell, CIFGLSTMCell):
        cell = cls(D, H)
        x = torch.randn(1, T, D)
        mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0]], dtype=torch.float)
        _, (h_pad, _) = RecurrentLayer(cell)(x, mask)
        _, (h_short, _) = RecurrentLayer(cell)(x[:, :4], torch.ones(1, 4))
        assert torch.allclose(h_pad, h_short, atol=1e-6)
    print("padding mask OK for vanilla / wmc / cifg")


def test_cifg_coupling_and_params():
    v, c, w = VanillaLSTMCell(D, H), CIFGLSTMCell(D, H), WMCLSTMCell(D, H)
    n = lambda m: sum(p.numel() for p in m.parameters())
    print(f"params  vanilla={n(v)}  cifg={n(c)}  wmc={n(w)}  lstmp={n(LSTMPCell(D, H, P))}")
    assert n(c) == 3 * n(v) // 4
    assert n(w) == n(v) + 3 * H * H




def test_gradients_and_shapes():
    """Every parameter of every cell receives a finite, non-zero gradient."""
    for name, cls in (("vanilla", VanillaLSTMCell), ("cifg", CIFGLSTMCell), ("wmc", WMCLSTMCell)):
        cell = cls(D, H)
        out, (h, c) = RecurrentLayer(cell)(torch.randn(B, T, D), torch.ones(B, T))
        assert out.shape == (B, T, H) and h.shape == (B, H) and c.shape == (B, H)
        out.sum().backward()
        for pn, p in cell.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, (name, pn)
    print("shapes + gradients OK for vanilla / cifg / wmc")


def test_wmc_equations_by_hand():
    """Recompute one WMC step directly from equations (7)-(12)."""
    cell = WMCLSTMCell(D, H)
    x, h, c = torch.randn(B, D), torch.randn(B, H), torch.randn(B, H)
    Wxi, Wxf, Wxg, Wxo = cell.W_x.weight.chunk(4, 0)
    bi, bf, bg, bo = cell.W_x.bias.chunk(4, 0)
    Whi, Whf, Whg, Who = cell.W_h.weight.chunk(4, 0)
    Wci, Wcf = cell.W_c_if.weight.chunk(2, 0)
    Wco = cell.W_co.weight
    i = torch.sigmoid(x @ Wxi.T + h @ Whi.T + torch.tanh(c @ Wci.T) + bi)
    f = torch.sigmoid(x @ Wxf.T + h @ Whf.T + torch.tanh(c @ Wcf.T) + bf)
    g = torch.tanh(h @ Whg.T + x @ Wxg.T + bg)
    c_new = f * c + i * g
    o = torch.sigmoid(x @ Wxo.T + h @ Who.T + torch.tanh(c_new @ Wco.T) + bo)
    h_new = o * torch.tanh(c_new)
    out, (h2, c2) = cell(cell.W_x(x), (h, c))
    assert torch.allclose(h2, h_new, atol=1e-6) and torch.allclose(c2, c_new, atol=1e-6)
    print("WMC step == equations (7)-(12)  OK")


def test_cifg_equations_by_hand():
    cell = CIFGLSTMCell(D, H)
    x, h, c = torch.randn(B, D), torch.randn(B, H), torch.randn(B, H)
    Wxf, Wxg, Wxo = cell.W_x.weight.chunk(3, 0)
    bf, bg, bo = cell.W_x.bias.chunk(3, 0)
    Whf, Whg, Who = cell.W_h.weight.chunk(3, 0)
    f = torch.sigmoid(h @ Whf.T + x @ Wxf.T + bf)
    i = 1 - f
    o = torch.sigmoid(h @ Who.T + x @ Wxo.T + bo)
    g = torch.tanh(h @ Whg.T + x @ Wxg.T + bg)
    c_new = f * c + i * g
    _, (h2, c2) = cell(cell.W_x(x), (h, c))
    assert torch.allclose(c2, c_new, atol=1e-6) and torch.allclose(h2, o * torch.tanh(c_new), atol=1e-6)
    print("CIFG step == equations (13)-(18)  OK")


def run_all():
    test_vanilla_matches_torch(); test_lstmp_matches_torch()
    test_masking_gives_last_real_state(); test_cifg_coupling_and_params()
    test_gradients_and_shapes(); test_wmc_equations_by_hand(); test_cifg_equations_by_hand()
    print("ALL CELL TESTS PASSED")


if __name__ == "__main__":
    run_all()
