from models import LabelPropagation , LabelSpreading
from utilits.utilits import *
from models import ClassifierModel
from models import DataLabeling
from llm_annotator import LLMAnnotator
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
    
    #Annotate data with LLMs
    arg_parse.add_argument('--annotate_with_llms',
                           dest='annotate_with_llms',
                           action='store_true',
                           required=False,
                           help="""
                           Annotate Data with large language models and the one we generate and see that we can get good results
                           LLM : Cohere-"command"
                           for CL provide [data_path] and [COHERE-API-TOKEN] 
                           """)
    
    arg_parse.add_argument('--data_path_llm' , dest='data_path_llm',type=str, required=False)
    arg_parse.add_argument('--api_token' , dest='api_token',type=int, required=False)
    
    
    
    args =  arg_parse.parse_args()
    
    #Annotate data with LabelPropagation
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
            save_data_json(new_annotated_data ,args.save_data)
    
    #Annotate data with LLMs Command
    if args.annotate_with_llms :
        llm_annotator = LLMAnnotator(args.api_token)
        llm_annotator.annotate(args.data_path_llm)
        
    #train data with Distelbert for sentence Classification Model 
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
            