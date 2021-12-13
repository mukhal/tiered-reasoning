import torch.nn as nn

from www.model.EntNetImplementation.memory_cell import MemoryCell
from www.model.EntNetImplementation.output_module import OutputModule


class EntNetHead(nn.Module):
  def __init__(self, config, num_blocks=5, input_all_tokens=True):
    super().__init__()
    self.memory_cell = MemoryCell(config, num_blocks, input_all_tokens)
    self.output_module = OutputModule(config, input_all_tokens)
    self.output_layer = nn.Linear(config.hidden_size, config.num_labels)  # Maybe remove?

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    self.memory_cell = self.memory_cell.to(*args, **kwargs)
    self.output_module = self.output_module.to(*args, **kwargs)

  def forward(self, features_sentence, features_entity):
    # Input is of shape (num_sents, batch_size * num_stories * num_entities, self.num_attributes)
    # We want to pass the stories into the memory cells sentence by sentence.
    # num_stories is 2, and so we want to have batch_size * 2 * num_entities number of states, passing sentences
    # through the head in a sequential manner.
    states = None
    for sentence in features_sentence:
      states = self.memory_cell(sentence, states)

    output = self.output_module(features_entity, states)
    return output
