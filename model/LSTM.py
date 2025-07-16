import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        concat_size = input_size + hidden_size

        self.forget_gate = nn.Linear(concat_size, hidden_size)
        self.input_gate = nn.Linear(concat_size, hidden_size)
        self.cell_gate = nn.Linear(concat_size, hidden_size)
        self.output_gate = nn.Linear(concat_size, hidden_size)

        self.hidden_to_output = nn.Linear(hidden_size, output_size)

        self.init_weights()

    def init_weights(self):
        for gate in [self.forget_gate, self.input_gate, self.cell_gate, self.output_gate]:
            nn.init.xavier_uniform_(gate.weight)
            nn.init.zeros_(gate.bias)

        # forget gate의 bias는 1로 설정 (기억 유지)
        nn.init.constant_(self.forget_gate.bias, 1.0)

        nn.init.xavier_uniform_(self.hidden_to_output.weight)
        nn.init.zeros_(self.hidden_to_output.bias)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        h_t = torch.zeros(batch_size, self.hidden_size)
        c_t = torch.zeros(batch_size, self.hidden_size)

        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, input_size)
            combined = torch.cat((x_t, h_t), dim=1)

            f_t = torch.sigmoid(self.forget_gate(combined))
            i_t = torch.sigmoid(self.input_gate(combined))
            update_c_t = torch.tanh(self.cell_gate(combined))
            o_t = torch.sigmoid(self.output_gate(combined))

            c_t = f_t * c_t + i_t * update_c_t
            h_t = o_t * torch.tanh(c_t)

        # 마지막 시점의 h_t로 예측
        y = self.hidden_to_output(h_t)
        return y
