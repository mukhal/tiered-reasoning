import torch
import torch.nn as nn


class OutputModule(nn.Module):
  def __init__(self, config, num_blocks, input_all_tokens=True, device=None):
    super().__init__()
    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    drop_out = getattr(config, "cls_dropout", None)
    if drop_out is None:
      drop_out = getattr(config, "dropout_rate", None)
    if drop_out is None:
      drop_out = getattr(config, "hidden_dropout_prob", None)
    assert drop_out is not None, "Didn't set dropout!"
    self.dropout = nn.Dropout(drop_out)
    self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
    self.num_labels = config.num_labels
    self.input_all_tokens = input_all_tokens
    self.device = device
    self.num_blocks = num_blocks

  def forward(self, entity_encoding, states):
    # EntNet output module takes a query string - we will use the encoding of the specific entity
    # hopefully the gated cells with learn about the states of different entities and then given the query
    # string of the entity name the states can be retrieved.
    chunked_states = torch.chunk(states, self.num_blocks, dim=1)

    if self.input_all_tokens:
      x = entity_encoding[:, 0, :]  # take <s> token (equiv. to [CLS])
    else:
      x = entity_encoding

    p_vals = []
    for h_i in chunked_states:
      p_i = torch.softmax(x.t().matmul(h_i), dim=0)
      p_vals.append(p_i)

    p = torch.cat(p_vals)

    # Skip the calculation of u and y since p can be viewed as a distribution of potential answers
    return p
