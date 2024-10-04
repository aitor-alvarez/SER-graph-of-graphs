import numpy as np
import torch, os
from transformers import (AutoConfig, EarlyStoppingCallback,
                          TrainingArguments, Trainer, AutoFeatureExtractor)
from models.transformer_speech import HubertEmotion, Wav2VecEmotion
import evaluate

accuracy = evaluate.load("accuracy")

recall = evaluate.load('recall')

F1 = evaluate.load('f1')

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/hubert-large-ll60k")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    acc = accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
    rec_w = recall.compute(predictions=predictions, references=eval_pred.label_ids, average='macro')
    f1 = F1.compute(predictions=predictions, references=eval_pred.label_ids, average='macro')
    return {'accuracy':acc, 'weighted_recall':rec_w, 'f1':f1}


def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, padding=True, truncation=True
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

def emotion_classification_pretrained(model_name, dataset, output_dir, batch_size, num_epochs):
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
    elif 'wav2vec' in model_name:
        model = Wav2VecEmotion.from_pretrained(model_name, config=config)

    model.freeze_feature_extractor()
    encoded_dataset = dataset.map(preprocess_function, remove_columns="audio", batched=True)

    #Eary stopping if the
    early_stop = EarlyStoppingCallback(2, 1.0)

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
            metric_for_best_model='eval_loss'
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