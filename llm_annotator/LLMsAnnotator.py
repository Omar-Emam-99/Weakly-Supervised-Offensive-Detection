from langchain import PromptTemplate, HuggingFaceHub, LLMChain, FewShotPromptTemplate
from utilits import create_dataset_of_label_propagation
from langchain.llms import Cohere
from tqdm.auto import tqdm

class LLMAnnotator:
    def __init__(self,
                 api_key,
                 model_ckpt: str = "command",
                 temperature: float = 0.0):
        """
        Initialize the LLMAnnotator class.

        Args:
        - api_key: API key for the Cohere model
        - model_ckpt: Checkpoint for the LLM model
        - temperature: Temperature value for generating responses
        """
        self.LLM = Cohere(model="command-xlarge",
                          cohere_api_key=api_key,
                          temperature=temperature)
        self.prompt = """
        You are an expert to classify this sentence as "offensive" or "not offensive":

        Sentence: {sentence}
        label: <label: offensive, not offensive>
        """
        self.prompt_template = PromptTemplate(template=self.prompt, input_variables=["sentence"])

    def annotate(self, data):
        """
        Annotate the given data with offensive or not offensive labels.

        Args:
        - data: List of sentences to annotate

        Returns:
        - labeled_data: Dataset with annotated labels
        """
        llm_cohere = LLMChain(prompt=self.prompt_template, llm=self.LLM)
        labels = []
        for example in tqdm(data):
            gen_label = llm_cohere.run(example)
            labels.append(0 if gen_label == "Not offensive" else 1)

        return create_dataset_of_label_propagation(data, labels)
