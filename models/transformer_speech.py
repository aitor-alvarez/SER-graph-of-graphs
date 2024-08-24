import torch
import torch.nn as nn
from transformers import HubertPreTrainedModel, HubertModel, Wav2Vec2PreTrainedModel, Wav2Vec2Model


class ClassifierModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.dense(x)
        x = self.dropout(x)
        x = self.linear(x)
        return nn.Softmax(x)


class HubertEmotion(HubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.hubert = HubertModel(config)
        self.classifier = ClassifierModule(config)
        self.init_weights()

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
        x = self.classifier(x)
        return x

class Wav2VecEmotion(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.w2v = Wav2Vec2Model(config)
        self.classifier = ClassifierModule(config)
        self.init_weights()

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
        x = self.classifier(x)
        return x