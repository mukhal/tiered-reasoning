import torch
import torch.nn as nn


class OutputModule(nn.Module):
  def __init__(self, config, hidden_size, num_blocks, input_all_tokens=True, device=None):
    super().__init__()
    self.num_labels = config.num_labels
    self.input_all_tokens = input_all_tokens
    self.device = device
    self.num_blocks = num_blocks
    self.hidden_size = hidden_size
    self.R = nn.Linear(self.hidden_size, config.num_labels, bias=False)
    self.H = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
    self.activation = nn.PReLU(self.hidden_size, init=1.0)


  def forward(self, entity_encoding, states):
    # EntNet output module takes a query string - we will use the encoding of the specific entity
    # hopefully the gated cells with learn about the states of different entities and then given the query
    # string of the entity name the states can be retrieved.
    chunked_states = torch.chunk(states, self.num_blocks, dim=1)

    if self.input_all_tokens:
      x = entity_encoding[:, 0, :]  # take <s> token (equiv. to [CLS])
    else:
      x = entity_encoding

    # u will be batch size * hidden size
    u = torch.zeros(states.shape[0], self.hidden_size).to(self.device)
    for i, h_i in enumerate(chunked_states):
      p_i = torch.softmax((x * h_i).sum(dim=-1), dim=0)
      u[i,:] = p_i.matmul(h_i)
      del p_i

    y = self.R(self.activation(x + self.H(u)))
    # Skip the calculation of u and y since p can be viewed as a distribution of potential answers
    return y
