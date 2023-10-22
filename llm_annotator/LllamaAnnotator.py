from transformers import AutoConfig
from torch import cuda, bfloat16
from datasets import load_dataset
from tqdm.auto import tqdm
from utilits import *
import transformers
import torch
import time


class LllamaAnnotator :
    
    def __init__(self,API_TOKEN : str , 
                 model_ckpt :str ='meta-llama/Llama-2-13b-chat-hf'):
        
        self.model_ckpt = model_ckpt
        self.device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
        self.API_TOKEN = API_TOKEN
        
        
    def load_llama(self):
        #set quantization configuration to load large model with less GPU memory
        # this requires the `bitsandbytes` library
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )

        # begin initializing HF items, need auth token for these
        model_config = AutoConfig.from_pretrained(
            self.model_ckpt,
            use_auth_token=self.API_TOKEN
        )
        #load tokenizer from huggingface 
        tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_ckpt,
                use_auth_token=self.API_TOKEN)
        #Load llama from huggingface 
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_ckpt,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
            use_auth_token=self.API_TOKEN
        )
        #put model in inference mode
        model.eval()
        #Create pipeline 
        generate_text = transformers.pipeline(
            model=model, tokenizer=tokenizer,
            return_full_text=True,  # langchain expects the full text
            task='text-generation',
            # we pass model parameters here too
            temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            max_new_tokens=512,  # mex number of tokens to generate in the output
            #repetition_penalty=1.1,  # without this output begins repeating
            torch_dtype = torch.float16)
        
        return generate_text
        
    
    def Annotate(self,tweets : str):
        
        LLM = self.load_llama()
        ann_labels , labels = [] , []
        
        for example in tqdm(range(len(tweets))):
            best_prompt = f""" 
            [INST]<>
            Assistant is a expert JSON builder designed to classify tweet as offensive and not offensive.
            Assistant MUST output JSON file tweet ,Label and explaination, for example:
            ```json
            {{"Tweet": Came from user,
              "label": offensive and not offensiv,
              "Explaintion" : Reasoning why this tweet classified as not-offensive}}
            ```
            <>
            {tweets[example]}
            [/INST]
            \n\n
            ##
            """
            gen_label = LLM(best_prompt)
            labels.append(gen_label[0]["generated_text"])
            time.sleep(5)
        #deserialize data 
        outs = [[i for i in x.split("##")][1].strip() for x in labels]
        for i in outs :
            out = deserialize_relations(i)
            if out == {} :
                ann_labels.append("NULL")
            else :
                ann_labels.append(out)
                
                
        return ann_labels