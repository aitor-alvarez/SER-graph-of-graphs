import numpy as np
import torch
from transformers import (AutoConfig, EarlyStoppingCallback,
                          TrainingArguments, Trainer, AutoFeatureExtractor)
from models.transformer_speech import HubertEmotion, Wav2VecEmotion
from transformers.models.wavlm import WavLMForSequenceClassification
from transformers.models.hubert import HubertForSequenceClassification
from transformers.models.wav2vec2 import Wav2Vec2ForSequenceClassification
from sklearn.metrics import balanced_accuracy_score
import evaluate

accuracy = evaluate.load("accuracy")

recall = evaluate.load('recall')

F1 = evaluate.load('f1')

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    acc = accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
    acc_w = balanced_accuracy_score(eval_pred.label_ids, predictions)
    rec_w = recall.compute(predictions=predictions, references=eval_pred.label_ids, average='weighted')
    f1 = F1.compute(predictions=predictions, references=eval_pred.label_ids, average='weighted')
    return {'accuracy':acc['accuracy'], 'weighted_accuracy':acc_w , 'weighted_recall':rec_w['recall'],
            'weighted_f1':f1['f1']}


def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, sampling_rate=feature_extractor.sampling_rate, padding=True
    )
    return inputs


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch


def collate_fn(batch):
    tensors, targets = [], []
    for b in batch:
        tensors += [b['audio']['array']]
        targets += [b['label']]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets

def emotion_classification_pretrained(model_name, dataset, output_dir, batch_size, num_epochs, deepspeed=None):
    labels = dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    num_labels = len(id2label)
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name,
                                        num_labels=num_labels, label2id=label2id, id2label=id2label)
    if 'hubert' in model_name:
        model = HubertEmotion.from_pretrained(model_name, config=config)
        #model = HubertForSequenceClassification.from_pretrained(config._name_or_path, config=config)
        model.freeze_feature_extractor()
    elif 'wav2vec' in model_name:
        #Below for full training of the model
        #model = Wav2VecEmotion.from_pretrained(model_name, config=config)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name, config=config)
        model.freeze_feature_extractor()
    elif 'wavlm' in model_name:
        model = WavLMForSequenceClassification.from_pretrained(model_name, config=config)
        model.freeze_feature_encoder()


    encoded_dataset = dataset.map(preprocess_function, remove_columns="audio", batched=True, batch_size=10, writer_batch_size=100)

    #Early stopping if best metric does not decrease
    early_stop = EarlyStoppingCallback(3)

    training_args = TrainingArguments(
            output_dir=output_dir,
            remove_unused_columns=False,
            per_device_train_batch_size=int(batch_size),
            gradient_accumulation_steps=2,
            evaluation_strategy="steps",
            num_train_epochs=int(num_epochs),
            gradient_checkpointing=True,
            fp16=False,
            save_steps=500,
            eval_steps=500,
            logging_steps=100,
            learning_rate=3e-4,
            warmup_steps=500,
            save_total_limit=2,
            push_to_hub=False,
            deepspeed= deepspeed,
            load_best_model_at_end=True,
            metric_for_best_model='eval_weighted_recall'
        )

    trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=encoded_dataset["train"].with_format("torch"),
            eval_dataset=encoded_dataset["test"].with_format("torch"),
            tokenizer=feature_extractor,
            callbacks=[early_stop]
        )

    trainer.train(resume_from_checkpoint=False)