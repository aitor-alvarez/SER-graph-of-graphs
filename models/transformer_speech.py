import torch
import torch.nn as nn
from transformers import HubertPreTrainedModel, HubertModel, Wav2Vec2PreTrainedModel, Wav2Vec2Model


class ClassifierModule(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(config.hidden_size, num_classes)

    def forward(self, x):
        x = self.dense(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x


class HubertEmotion(HubertPreTrainedModel):
    def __init__(self, config, num_classes):
        super().__init__(config)
        self.hubert = HubertModel(config)
        self.classifier = ClassifierModule(config, num_classes)
        self.init_weights()

    def forward(self, x):
        outputs = self.hubert(x)
        hidden_states = outputs[0]
        x = torch.mean(hidden_states, dim=1)
        x = self.classifier(x)
        return x

class Wav2VecEmotion(Wav2Vec2PreTrainedModel):
    def __init__(self, config, num_classes):
        super().__init__(config)
        self.w2v = Wav2Vec2Model(config)
        self.classifier = ClassifierModule(config, num_classes)
        self.init_weights()

    def forward(self, x):
        outputs = self.w2v(x)
        hidden_states = outputs[0]
        x = torch.mean(hidden_states, dim=1)
        x = self.classifier(x)
        return x