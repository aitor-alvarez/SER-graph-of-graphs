import argparse
from datasets import load_dataset
from train_encoder import *
from create_multiGraph import MultiGraph
from utils.create_graphs import *

models = ['gnn']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--model_id')
    parser.add_argument('--num_epochs')
    parser.add_argument('--batch_size')
    parser.add_argument('--data_folder')
    parser.add_argument('--graph_folder')
    parser.add_argument('--output_dir')
    parser.add_argument('--create_graphs')
    parser.add_argument('--create_multi')
    parser.add_argument('--deepspeed')
    args = parser.parse_args()

    batch_size = int(args.batch_size)
    num_epochs = int(args.num_epochs)

    #For training/fine-tuning the audio encoder.
    if args.model_id not in models and args.data_folder:
        dataset = load_dataset("audiofolder", data_dir=args.data_folder, drop_labels=False)
        if args.deepspeed:
            emotion_classification_pretrained(args.model_id, dataset, args.output_dir, batch_size, num_epochs, args.deepspeed)
        else:
            emotion_classification_pretrained(args.model_id, dataset, args.output_dir, batch_size, num_epochs)
    if args.create_graphs and args.data_folder:
        generate_graphs(args.data_folder)
    elif args.create_multi =='y':
       #First, train the local graphs and obtain embeddings
       mg = MultiGraph(args.graph_data, num_class=4, emb_size=512, batch_size=128, is_local_trained = False)
       mg.run()
       #Once local embeddings are obtained, create and train the multilevel graph.
       mg.is_local_trained = True
       mg.run()
    else:
        print("No model or dataset has been selected")
