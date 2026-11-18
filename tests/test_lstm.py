# -*- coding: utf-8 -*-
"""手写 LSTM 与 nn.LSTM 的数值对齐测试。"""
import torch

from llm_lab.models import alignment_max_diff, build_aligned_nn_lstm, MyLSTM


def test_custom_lstm_matches_nn_lstm():
    for num_layers in (1, 2):
        assert alignment_max_diff(num_layers) < 1e-6


def test_my_lstm_output_shapes():
    seq_len, batch, input_size, hidden_size = 5, 3, 4, 8
    model = MyLSTM(input_size, hidden_size, num_layers=2)
    out, (h, c) = model(torch.randn(seq_len, batch, input_size))
    assert out.shape == (seq_len, batch, hidden_size)
    assert h.shape == (2, batch, hidden_size)
    assert c.shape == (2, batch, hidden_size)


def test_aligned_nn_lstm_receives_parameters():
    torch.manual_seed(0)
    mine = MyLSTM(4, 8, num_layers=1)
    ref = build_aligned_nn_lstm(mine)
    assert torch.equal(ref.weight_ih_l0, mine.cells[0].W_ih.weight)
    assert torch.equal(ref.bias_hh_l0, mine.cells[0].W_hh.bias)
