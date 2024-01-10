import argparse
import os
from datasets import load_dataset
from train import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( '--model_id')
    parser.add_argument('--num_epochs')
    parser.add_argument('--batch_size')
    parser.add_argument('--data_folder')
    parser.add_argument('--output_dir')
    parser.add_argument('--train_test')
    args = parser.parse_args()

    if args.model_id and args.data_folder:
        dataset = load_dataset("audiofolder", data_dir=args.data_folder)
        emotion_classification_hubert(args.model_id, dataset, args.output_dir, args.batch_size, args.num_epochs, args.train_test)

