import torch.nn as nn


class OutputModule(nn.Module):
    def __init__(self, config, input_all_tokens=True):
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

    def forward(self, states):
        chunked_states = torch.chunk(states, self.num_blocks)

        if self.input_all_tokens:
          x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        else:
          x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        if return_embeddings:
          emb = x
        x = self.dropout(x)
        x = self.out_proj(x)
        if not return_embeddings:
          return x
        else:
          return x, emb
