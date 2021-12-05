import torch.nn as nn

from www.model.EntNetImplementation.memory_cell import MemoryCell
from www.model.EntNetImplementation.output_module import OutputModule


class EntNetHead(nn.Module):
    def __init__(self, config, num_blocks=5, input_all_tokens=True):
        super().__init__()
        self.memory_cell = MemoryCell(config, num_blocks, input_all_tokens)
        self.output_model = OutputModule(config, input_all_tokens)

    def forward(self, features_sentence, features_entity, states=None, return_embeddings=False):
        states = self.memory_cell(features_sentence, states)


