import torch.nn as nn

from www.model.EntNetImplementation.memory_cell import MemoryCell
from www.model.EntNetImplementation.output_module import OutputModule


class EntNetHead(nn.Module):
    def __init__(self, config, num_blocks=5, input_all_tokens=True):
        super().__init__()
        self.memory_cell = MemoryCell(config, num_blocks, input_all_tokens)
        self.output_module = OutputModule(config, input_all_tokens)
        self.output_layer = nn.Linear(config.hidden_size, config.num_labels) # Maybe remove?

    def forward(self, features_sentence, features_entity, states=None, return_embeddings=False):
        states = self.memory_cell(features_sentence, states)
        output = self.output_module(features_entity, states)
        return output