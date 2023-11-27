from torch import cuda, bfloat16
import transformers
from transformers import AutoConfig
import torch
from datasets import load_dataset
from utilits import *
from tqdm.auto import tqdm
import time

class LllamaAnnotator:
    """
    Annotator class for Lllama model.

    Args:
        API_TOKEN (str): API token for authentication.
        model_ckpt (str): Model checkpoint path.

    Attributes:
        model_ckpt (str): Model checkpoint path.
        device (str): Device to use for inference.
        API_TOKEN (str): API token for authentication.
    """

    def __init__(self, API_TOKEN: str, model_ckpt: str = 'meta-llama/Llama-2-13b-chat-hf'):
        self.model_ckpt = model_ckpt
        self.device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
        self.API_TOKEN = API_TOKEN

    def load_llama(self):
        """
        Load Lllama model.

        Returns:
            pipeline: Hugging Face pipeline for text generation.
        """
        # Set quantization configuration to load large model with less GPU memory
        # This requires the `bitsandbytes` library
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )

        # Begin initializing HF items, need auth token for these
        model_config = AutoConfig.from_pretrained(
            self.model_ckpt,
            use_auth_token=self.API_TOKEN
        )
        # Load tokenizer from Hugging Face
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_ckpt,
            use_auth_token=self.API_TOKEN
        )
        # Load Lllama from Hugging Face
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_ckpt,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
            use_auth_token=self.API_TOKEN
        )
        # Put model in inference mode
        model.eval()
        # Create pipeline
        generate_text = transformers.pipeline(
            model=model, tokenizer=tokenizer,
            return_full_text=True,  # langchain expects the full text
            task='text-generation',
            temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            max_new_tokens=512,  # max number of tokens to generate in the output
            torch_dtype=torch.bfloat16
        )

        return generate_text

    def Annotate(self, tweets: str):
        """
        Annotate tweets using Lllama model.

        Args:
            tweets (str): List of tweets.

        Returns:
            list: List of annotations for each tweet.
        """
        LLM = self.load_llama()
        alls, labels = [], []

        for example in tqdm(range(len(tweets))):
            best_prompt = f"""
            [INST]<>
            Assistant is an expert JSON builder designed to classify tweets as offensive and not offensive.
            Assistant MUST output a JSON file with tweet, label, and explanation, for example:
            ```json
            {{
              "Tweet": "Came from user",
              "label": "offensive and not offensive",
              "Explanation": "Reasoning why this tweet classified as not-offensive"
            }}
            ```
            <>
            {tweets[example]}
            [/INST]
            \\n\\n
            ##
            """
            gen_label = LLM(best_prompt)
            labels.append(gen_label[0]["generated_text"])
            time.sleep(5)

        # Deserialize data
        outs = [[i for i in x.split("##")][1].strip() for x in labels]
        for i in outs:
            out = deserialize_relations(i)
            if out == {}:
                alls.append("NULL")
            else:
                alls.append(out)

        return alls