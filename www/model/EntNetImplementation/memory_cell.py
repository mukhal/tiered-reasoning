import torch.nn as nn
import torch

class MemoryCell(nn.Module):
    def __init__(self, config, num_blocks=5, input_all_tokens=True):
        super().__init__()

        self.hidden_size = config.hidden_size

        # Learnable parameters U, V, W can be viewed as linear layers with no bias
        self.U = nn.Linear(self.hidden_size, config.hidden_size, bias=False)
        self.V = nn.Linear(self.hidden_size, config.hidden_size, bias=False)
        self.W = nn.Linear(self.hidden_size, config.hidden_size, bias=False)

        # We need num_blocks of gates and hidden states, so store this for later
        self.num_blocks = num_blocks
        self.keys = {str(i): nn.Parameter(torch.zeros(self.hidden_size)) for i in range(num_blocks)}
        self.keys = nn.ParameterDict(self.keys)

        # Activation function with parameters as in the EntNet paper
        # authors also tried linear activation
        self.activation = nn.PReLU(self.hidden_size, init=1.0)

    def forward(self, features, states=None):
        if self.input_all_tokens:
          encoded_input = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        else:
          encoded_input = features

        if states is None:
            states = torch.zeros(self.hidden_size * self.num_blocks)

        chunked_states = torch.chunk(states, self.num_blocks)

        new_states = []

        for i in range(self.num_blocks):
            hidden_i = chunked_states[i]
            key_i = self.keys[str(i)]

            # Calculate gate value
            gate_i = torch.sigmoid(encoded_input.matmul(hidden_i) + encoded_input.matmul(key_i))

            # Calculate candidate hidden value
            candidate_hidden_i = self.activation(self.U(hidden_i) + self.V(key_i) + self.W(encoded_input))

            # Update hidden value for this block
            hidden_i = hidden_i + torch.mul(gate_i, candidate_hidden_i)
            hidden_i_no_zeros = hidden_i
            hidden_i_no_zeros[hidden_i_no_zeros==0] = 0.1 # Avoid dividing by zero, value can be set to anything as 0 on the numerator will result in 0 anyway
            hidden_i = hidden_i / torch.abs(hidden_i_no_zeros)

            new_states.append(hidden_i)

        return torch.cat(new_states)
