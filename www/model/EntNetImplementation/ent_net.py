import torch
import torch.nn as nn

from www.model.EntNetImplementation.memory_cell import MemoryCell
from www.model.EntNetImplementation.output_module import OutputModule


class EntNetHead(nn.Module):
  def __init__(self, config, memory_hidden_size, num_blocks=5, input_all_tokens=True, device=None):
    super().__init__()
    self.embedding_projection = nn.Linear(config.hidden_size, memory_hidden_size)
    self.memory_cell = MemoryCell(memory_hidden_size, num_blocks, input_all_tokens, device=device)
    self.output_module = OutputModule(config, memory_hidden_size, num_blocks, input_all_tokens, device=device)
    self.output_layer = nn.Linear(num_blocks, config.num_labels)
    self.num_labels = config.num_labels
    self.device = device

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    self.memory_cell = self.memory_cell.to(*args, **kwargs)
    self.output_module = self.output_module.to(*args, **kwargs)
    return self

  def forward(self, features_sentence, features_entity):
    # Input is of shape (num_sents, batch_size * num_stories * num_entities, self.num_attributes)
    # We want to pass the stories into the memory cells sentence by sentence.
    # num_stories is 2, and so we want to have batch_size * 2 * num_entities number of states, passing sentences
    # through the head in a sequential manner.
    output_shape = (features_sentence.shape[0], features_sentence.shape[1], self.num_labels)
    predictions = torch.zeros(output_shape)
    if self.device is not None:
      predictions = predictions.to(self.device)

    states = None
    features_sentence = self.embedding_projection(features_sentence)
    features_entity = self.embedding_projection(features_entity)
    for i, sentence in enumerate(features_sentence): # Want to make a prediction at each of these
      states = self.memory_cell(sentence, states)
      sentence_predictions = self.output_module(features_entity[i], states)
      # batch_size * num_labels for sentence_predictions coming from the output module
      predictions[i,:,:] = sentence_predictions
      del sentence_predictions

    del states
    return predictions.view(features_sentence.shape[0] * features_sentence.shape[1], self.num_labels)
