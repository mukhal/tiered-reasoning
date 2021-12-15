import torch.nn as nn
import torch


class MemoryCell(nn.Module):
  def __init__(self, config, num_blocks=5, input_all_tokens=True, device=None):
    super().__init__()

    self.hidden_size = config.hidden_size

    # Learnable parameters U, V, W can be viewed as linear layers with no bias
    self.U = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
    self.V = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
    self.W = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    # We need num_blocks of gates and hidden states, so store this for later
    self.num_blocks = num_blocks
    self.keys = {str(i): nn.Parameter(torch.zeros(self.hidden_size)) for i in range(num_blocks)}
    self.keys = nn.ParameterDict(self.keys)

    # Activation function with parameters as in the EntNet paper
    # authors also tried linear activation
    self.activation = nn.PReLU(self.hidden_size, init=1.0)

    self.input_all_tokens = input_all_tokens

    self.device = device

  def forward(self, features, states=None):
    if self.input_all_tokens:
      encoded_input = features[:, 0, :]  # take <s> token (equiv. to [CLS])
    else:
      encoded_input = features

    batch_size = features.shape[0]
    if states is None:
      # Want a different internal state for each different item in batch
      states = torch.zeros(batch_size, self.hidden_size * self.num_blocks)
      if self.device is not None:
        states = states.to(self.device)

    chunked_states = torch.chunk(states, self.num_blocks, dim=1)

    new_states = []

    for i in range(self.num_blocks):
      hidden_i = chunked_states[i]

      # Gates/keys should be shared amongst each item in batch
      # unsqueeze to become size (1, hidden) then repeat to become (batch_size, hidden)
      key_i = self.keys[str(i)].unsqueeze(0).repeat(batch_size, 1)

      # Calculate gate value
      # encoded_input * hidden_i is element wise multiplication, operation described by entnet paper transposes
      # input for multiplication, this is a dot product, so we want to do element wise multiplication and then sum
      # final dimension
      input_hidden_dot_prod = (encoded_input * hidden_i).sum(dim=-1)
      input_key_dot_prod = (encoded_input * key_i).sum(dim=-1)
      gate_i = torch.sigmoid(input_hidden_dot_prod + input_key_dot_prod)
      del input_key_dot_prod, input_hidden_dot_prod

      # Calculate candidate hidden value
      candidate_hidden_i = self.activation(self.U(hidden_i) + self.V(key_i) + self.W(encoded_input))

      # Update hidden value for this block
      # want the hadamard product of gate_i and candidate
      hidden_i = hidden_i + torch.mul(gate_i.unsqueeze(dim=1), candidate_hidden_i)
      hidden_i_no_zeros = hidden_i
      hidden_i_no_zeros[hidden_i_no_zeros == 0] = 0.1  # Avoid dividing by zero, value can be set to anything as 0 on the numerator will result in 0 anyway
      hidden_i = hidden_i / torch.abs(hidden_i_no_zeros)

      new_states.append(hidden_i)
      del hidden_i, hidden_i_no_zeros, candidate_hidden_i, gate_i, key_i

    return torch.cat(new_states, dim=1)
