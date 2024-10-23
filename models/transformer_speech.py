import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.file_utils import ModelOutput
from transformers import (HubertPreTrainedModel, HubertModel,
                          Wav2Vec2PreTrainedModel, Wav2Vec2Model)


@dataclass
class SpeechOutputClassifier(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class ClassifierModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.dense(x)
        output = self.linear(x)
        return output


class HubertEmotion(HubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = self.config.num_labels
        self.hubert = HubertModel(config)
        self.classifier = ClassifierModule(config)

    def freeze_feature_extractor(self):
        self.hubert.feature_extractor._freeze_parameters()
    def forward(self,
                input_values,
        attention_mask = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict =  None,
        labels = None):
        outputs = self.hubert(input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        hidden_states = outputs[0]
        x = torch.mean(hidden_states, dim=1)
        logits = self.classifier(x)
        celoss = nn.CrossEntropyLoss()
        loss = celoss(logits.view(-1, self.num_labels), labels.view(-1))
        return SpeechOutputClassifier(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class Wav2VecEmotion(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = self.config.num_labels
        self.w2v = Wav2Vec2Model(config)
        self.classifier = ClassifierModule(config)

    def freeze_feature_extractor(self):
        self.w2v.feature_extractor._freeze_parameters()

    def forward(self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None):

        outputs = self.w2v(input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        hidden_states = outputs[0]
        x = torch.mean(hidden_states, dim=1)
        logits = self.classifier(x)
        celoss = nn.CrossEntropyLoss()
        loss = celoss(logits.view(-1, self.num_labels), labels.view(-1))
        return SpeechOutputClassifier(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )