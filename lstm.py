# -*- coding: utf-8 -*-
"""用 PyTorch 从零实现 LSTM（手写门控单元），并与 nn.LSTM 对齐验证 + 训练小demo。

LSTM 单步公式：
    i_t = sigmoid(W_i x_t + b_i + W_h h_{t-1} + b_h)   输入门
    f_t = sigmoid(...)                                 遗忘门
    g_t = tanh(...)                                    候选状态
    o_t = sigmoid(...)                                 输出门
    c_t = f_t * c_{t-1} + i_t * g_t                    细胞状态更新
    h_t = o_t * tanh(c_t)                              隐藏状态输出
"""
import math

import torch
from torch import nn


class MyLSTMCell(nn.Module):
    """单层 LSTM 单元（四个门共用两次矩阵乘，一次完成）。"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 一次算出 4 组门/候选的线性变换，输出 4*hidden，随后 chunk 成 i/f/g/o
        self.W_ih = nn.Linear(input_size, 4 * hidden_size)   # 作用于 x_t
        self.W_hh = nn.Linear(hidden_size, 4 * hidden_size)  # 作用于 h_{t-1}

    def forward(self, x, state):
        h, c = state
        gates = self.W_ih(x) + self.W_hh(h)          # (batch, 4*hidden)
        i, f, g, o = gates.chunk(4, dim=-1)          # PyTorch 官方门顺序: i, f, g, o
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * c + i * g                            # 细胞状态
        h = o * torch.tanh(c)                        # 隐藏状态
        return h, c


class MyLSTM(nn.Module):
    """多层 LSTM，按时间步循环展开。

    约定与 nn.LSTM 默认一致：输入 (seq_len, batch, input_size)。
    返回 outputs: (seq_len, batch, hidden) 以及最后一步的 (h_n, c_n)。
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            MyLSTMCell(input_size if layer == 0 else hidden_size, hidden_size)
            for layer in range(num_layers)
        ])

    def forward(self, x, state=None):
        seq_len, batch, _ = x.shape
        device = x.device
        if state is None:
            h = [torch.zeros(batch, self.hidden_size, device=device)
                 for _ in range(self.num_layers)]
            c = [torch.zeros(batch, self.hidden_size, device=device)
                 for _ in range(self.num_layers)]
        else:
            h, c = list(state[0]), list(state[1])

        outputs = []
        for t in range(seq_len):
            xt = x[t]
            for layer, cell in enumerate(self.cells):
                h[layer], c[layer] = cell(xt, (h[layer], c[layer]))
                xt = h[layer]                        # 上层输入 = 下层输出
            outputs.append(h[-1])
        return torch.stack(outputs, dim=0), (torch.stack(h), torch.stack(c))


def build_aligned_nn_lstm(my_lstm: MyLSTM) -> nn.LSTM:
    """把自定义参数拷贝进 nn.LSTM，用于数值对齐验证（门顺序 i/f/g/o 一致）。"""
    ref = nn.LSTM(my_lstm.cells[0].input_size, my_lstm.hidden_size,
                  num_layers=my_lstm.num_layers)
    with torch.no_grad():
        for layer, cell in enumerate(my_lstm.cells):
            getattr(ref, f"weight_ih_l{layer}").copy_(cell.W_ih.weight)
            getattr(ref, f"weight_hh_l{layer}").copy_(cell.W_hh.weight)
            getattr(ref, f"bias_ih_l{layer}").copy_(cell.W_ih.bias)
            getattr(ref, f"bias_hh_l{layer}").copy_(cell.W_hh.bias)
    return ref


def verify_against_nn_lstm():
    """随机输入下，自定义实现应与 nn.LSTM 输出完全一致。"""
    torch.manual_seed(0)
    for num_layers in (1, 2):
        seq_len, batch, input_size, hidden_size = 7, 5, 3, 16
        mine = MyLSTM(input_size, hidden_size, num_layers)
        ref = build_aligned_nn_lstm(mine)
        x = torch.randn(seq_len, batch, input_size)
        out_mine, (h_mine, c_mine) = mine(x)
        out_ref, (h_ref, c_ref) = ref(x)
        ok = (torch.allclose(out_mine, out_ref, atol=1e-6)
              and torch.allclose(h_mine, h_ref, atol=1e-6)
              and torch.allclose(c_mine, c_ref, atol=1e-6))
        print(f"对齐验证 num_layers={num_layers}: {'PASS' if ok else 'FAIL'} "
              f"(max diff = {(out_mine - out_ref).abs().max().item():.2e})")


def demo_sine_forecast():
    """训练demo：给定过去 20 个点的正弦窗口，预测下一个点的值。"""
    torch.manual_seed(0)

    # 构造数据：滑窗切分正弦序列
    t = torch.arange(0, 200, dtype=torch.float32)
    wave = torch.sin(t * 0.1)
    window = 20
    xs, ys = [], []
    for i in range(len(wave) - window):
        xs.append(wave[i:i + window])
        ys.append(wave[i + window])
    X = torch.stack(xs).unsqueeze(-1)   # (N, window, 1) -> 转 (window, N, 1)
    y = torch.stack(ys)                 # (N,)
    X = X.permute(1, 0, 2)              # (window, N, 1)

    model = nn.Sequential(MyLSTM(1, 32), nn.Identity())
    lstm = model[0]
    head = nn.Linear(32, 1)

    optimizer = torch.optim.Adam(list(lstm.parameters()) + list(head.parameters()), lr=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(1, 301):
        outputs, _ = lstm(X)            # (window, N, 32)
        pred = head(outputs[-1]).squeeze(-1)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"epoch {epoch:3d}  loss = {loss.item():.6f}")

    with torch.no_grad():
        pred = head(lstm(X)[0][-1]).squeeze(-1)
        print(f"最终 MSE = {loss_fn(pred, y).item():.6f}")
        print("前5个样本 预测 vs 真实:")
        for p, gt in zip(pred[:5].tolist(), y[:5].tolist()):
            print(f"  {p:+.4f}  vs  {gt:+.4f}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}  torch: {torch.__version__}")
    verify_against_nn_lstm()
    print("-" * 40)
    demo_sine_forecast()
