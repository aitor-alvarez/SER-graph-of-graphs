import argparse
from datasets import load_dataset
from train_encoder import *
from create_multiGraph import MultiGraph
from utils.intonation_patterns import generate_initial_graph

models = ['resblstm', 'gnn']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( '--model_id')
    parser.add_argument('--num_epochs')
    parser.add_argument('--batch_size')
    parser.add_argument('--data_folder')
    parser.add_argument('--graph_folder')
    parser.add_argument('--output_dir')
    parser.add_argument('--create_graphs')
    parser.add_argument('--create_multi')
    args = parser.parse_args()

    #For training/tuning the audio encoder.
    if args.model_id not in models and args.data_folder:
        dataset = load_dataset("audiofolder", data_dir=args.data_folder)
        emotion_classification_pretrained(args.model_id, dataset, args.output_dir, args.batch_size, args.num_epochs, args.train_test)
    elif args.model_id in models and args.data_folder:
        if 'resblstm' in args.model_id and args.data_folder:
            dataset = load_dataset("audiofolder", data_dir=args.data_folder)
            train_torch_model(args.model_id, dataset, args.output_dir, args.batch_size, args.num_epochs, args.train_test)
    if args.create_graphs and args.data_folder:
        generate_initial_graph(args.data_folder)
    elif args.create_multi =='y':
       #First, train the local graphs and obtain embeddings
       mg = MultiGraph(args.graph_data, num_class=4, emb_size=512, batch_size=128, is_local_trained = False)
       mg.run()
       #Once local embeddings are obtained, create and train the multilevel graph.
       mg.is_local_trained = True
       mg.run()
    else:
        print("No model or dataset has been selected")
