import numpy as np
import torch
from torch import utils
from transformers import AutoConfig, Wav2Vec2FeatureExtractor, TrainingArguments, Trainer, AutoModelForAudioClassification, AutoFeatureExtractor
from models.resnet50_blstm import LightResnet, ResnetBLSTM, Bottleneck
import evaluate
import lightning as L

accuracy = evaluate.load("accuracy")

recall = evaluate.load('recall')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/hubert-large-ll60k")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    acc = accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
    rec_w = recall.compute(predictions=predictions, references=eval_pred.label_ids, average='weighted')
    rec_u = recall.compute(predictions=predictions, references=eval_pred.label_ids, average=None)
    return acc, rec_w, rec_u


def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, padding=True ,truncation=True
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

def emotion_classification_pretrained(model_name, dataset, output_dir, batch_size, num_epochs,train_test):
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name)
    #hubert_emotion = HubertEmotion.from_pretrained(model_name,config=config).to(device)
    labels = dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    num_labels = len(id2label)
    encoded_dataset = dataset.map(preprocess_function, remove_columns="audio", batched=True)

    model = AutoModelForAudioClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    ).to(device)

    if train_test == 'train':
        training_args = TrainingArguments(
            output_dir=output_dir,
            remove_unused_columns=False,
            per_device_train_batch_size=32,
            gradient_accumulation_steps=2,
            evaluation_strategy="steps",
            num_train_epochs=100,
            gradient_checkpointing=True,
            fp16=True,
            save_steps=400,
            eval_steps=1000,
            logging_steps=100,
            learning_rate=3e-4,
            warmup_steps=500,
            save_total_limit=2,
            push_to_hub=False,
        )
        model.freeze_feature_extractor()

        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=encoded_dataset["train"].with_format("torch"),
            eval_dataset=encoded_dataset["test"].with_format("torch"),
            tokenizer=feature_extractor,
        )

        trainer.train(resume_from_checkpoint=True)

def train_torch_model(model_name, dataset, output_dir, batch_size, num_epochs,train_test):
    if train_test == 'train':
        train_data = dataset["train"].select_columns(['audio', 'label'])
        train_loader = utils.data.DataLoader(train_data.with_format("torch", device=device), batch_size=int(batch_size), shuffle=True, collate_fn=collate_fn)
        if model_name == 'resblstm':
            resnet = ResnetBLSTM(Bottleneck, [3, 6, 3])
            print(resnet.parameters())

            model = LightResnet(resnet)
            trainer = L.Trainer(limit_train_batches=int(batch_size), max_epochs=int(num_epochs))
            trainer.fit(model=model, train_dataloaders=train_loader)
