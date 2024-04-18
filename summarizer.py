from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64
import os


from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_core.documents import Document

class Summer:
    def __init__(self, userName):
        self.userName = userName
    
    def get_user_details():
        pass

    def get_vetordb(self, docs):

        if not os.path.exists(f'{self.userName}'):
            self.vectorDB = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db", collection_name=self.userName)
        else:
            # load from disk
            self.vectorDB = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function, collection_name=self.userName)
            self.vectorDB.add_documents(docs)

# split it into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents("text")

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# save to disk



# docs = db3.similarity_search(query)


checkpoint = "./LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint )
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint , device_map='auto', torch_dtype = torch.float32)

pipe = pipeline(
    "summarization",
    model = base_model,
    tokenizer=tokenizer,
    max_length = 500,
    min_length = 50
)

hf = HuggingFacePipeline(pipeline=pipe)

template =""" Summarize the given input text.

Input text : {input_text}

return the answer in the following format.

Answer: 
"""
prompt = PromptTemplate.from_template(template)

chain = prompt | hf



# print(chain.invoke({"input_text": input_text}))
