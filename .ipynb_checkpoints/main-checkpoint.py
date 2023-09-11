from models.LP import LabelPropagation , LabelSpreading
from utilits.utilits import *
from models.Classifier_model import ClassifierModel
from models.Label_Data import DataLabeling
import argparse

if __name__ == "__main__" :
    
    arg_parse = argparse.ArgumentParser(
        description="""Two option :
                    one : generate Annotated Label For Data
                    two : train Classifier""")
    arg_parse.add_argument('--generate', dest='generate',
                           action='store_true',
                           required=False,
                          help="""
                          generate Annotated data from [path] that contain a train data so
                          will split it with annotated data and None Annotated data with split it with ratio of [split]
                          can save new generated data with [save_path]
                          """)
    arg_parse.add_argument('--path', dest='path' , type=str,required=False)
    arg_parse.add_argument('--split', dest='split',type=float , required=False)
    arg_parse.add_argument('--save_model', dest='save_model' , type=str,required=False)
    arg_parse.add_argument('--save_data', dest='save_data' , type=str,required=False)
    
    arg_parse.add_argument('--train',
                           dest='train',
                           action='store_true',
                           required=False,
                           help="""
                           train model with annotated data and the one we generate and see that we can get good results
                           train-model : distelbert
                           for CL provide [data_path] and [num_epoch] 
                           """)
    
    arg_parse.add_argument('--data_path' , dest='data_path',type=str, required=False)
    arg_parse.add_argument('--num_epoch' , dest='num_epoch',type=int, required=False)
    
    arg_parse.add_argument('--test',
                           dest='test',
                           action='store_true',
                          required=False)
    
    args =  arg_parse.parse_args()
    
    if args.generate:
        
        olid_dataset = prepare_data(args.path)
        annoteted_data = split_data(olid_dataset["train"] , annotated_data_prec=args.split)
        LP = DataLabeling()
        preds_labels = LP.generate_labels(annoteted_data["train"]["tweets"],
                                   annoteted_data["train"]["labels"],
                                   annoteted_data["test"]["tweets"])
        
        if args.save_model :
            LP.save(LP , args.save_model)
            
        new_annotated_data = create_dataset_of_label_propagation(annoteted_data["test"]["tweets"],
                                                                 preds_labels)
        if args.save_data :
            save_data_json(new_annotated_data ,arg.save_data)
    
    if args.train :
        data = prepare_data(args.data_path)
        train_data = split_data(data["train"], annotated_data_prec=0.8)
        m = ClassifierModel()
        m.train(train_data["train"],
                train_data["test"],
                num_epochs=args.num_epoch)
        
        if args.test :
            preds = m.test(data["test"])
            print(preds)
            