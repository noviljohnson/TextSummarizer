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
from langchain.chains.question_answering import load_qa_chain

import sqlite3
from flask import Flask
from flask import Flask, render_template, request, redirect, session
from flask import jsonify


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def get_my_ip():
    return request.remote_addr, 200



############################################################################
#### Summarization Chain 
# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


checkpoint = "./LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint, max_length=1024, truncation=True )
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint , device_map='auto', offload_folder="offload",torch_dtype = torch.float32)
print(f"\n ==== {tokenizer.model_max_length} ==== \n")
pipe = pipeline(
    "summarization",
    model = base_model,
    tokenizer=tokenizer,
    max_length = 1000,
    min_length = 50
)

hf = HuggingFacePipeline(pipeline=pipe)

template =""" Summarize the given input text.
        Input text : {input_text}
        return the answer in the following format.
        Answer: 
        """
title_template =""" give a title for the given input text.
        Input text : {input_text}
        return the answer in the following format.
        Answer: 
        """
prompt = PromptTemplate.from_template(template)
title_prompt = PromptTemplate.from_template(title_template)

chain = prompt | hf
title_chain = title_prompt | hf


#### QA Pipeline

model_name = "Intel/dynamic_tinybert"

# Load the tokenizer associated with the specified model
tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=1024)
print(f"\n 2222==== {tokenizer.model_max_length} ==== \n")
# Define a question-answering pipeline using the model and tokenizer
question_answerer = pipeline(
    "question-answering", 
    model=model_name, 
    tokenizer=tokenizer,
    return_tensors='pt'
)

# Create an instance of the HuggingFacePipeline, which wraps the question-answering pipeline
# with additional model-specific arguments (temperature and max_length)
llm = HuggingFacePipeline(
    pipeline=question_answerer,
    model_kwargs={"temperature": 0.2, "max_length": 1024},
)

prompt_template = """Use the following pieces of context to answer the question at the end. 
If the answer can be inferred from the context, provide it with proper punctuation and capitalization. 
If the answer is not present in the context but can be addressed with your model knowledge up to your last update, use that information to answer. 
If you still don't know the answer, state that you don't know. Do not fabricate an answer.

{context}

Question: {question}
Generate point-wise answers for the question , and format them clearly with bullet points or numbers.
Conclude or explain the context based on the question's intent when possible.
If the answer is not present in the context, attempt to provide a close enough answer using the model's knowledge without making assumptions beyond that knowledge.
Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]#, "sources"]
)

qa_chain=load_qa_chain(llm,  chain_type="stuff", prompt=PROMPT)

#######################################################################


Total_users = 0

class Summer:
    def __init__(self, email='', embedding_function=embedding_function):
        self.email = email
        self.embedding_function = embedding_function
        if not os.path.exists('./UserDb.db'):
            self.create_db()

    def create_db(self):
        conn = sqlite3.connect('UserDb.db')
        # Creating a cursor object to execute SQL commands
        cur = conn.cursor()
        # Creating a table named users with two columns: id and name
        cur.execute('''
            CREATE TABLE users (
                user_id INTEGER PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                full_name TEXT,
                password TEXT
            );
        ''')

        conn.commit()
        conn.close()

    def get_user_details(self):
        conn = sqlite3.connect('UserDb.db')
        cur = conn.cursor()   
        # Creating a cursor object to execute SQL queries
        cur.execute(f'''Select * from users where email = "{self.email}"''')
        records = cur.fetchall()
        conn.close()
        return records
    
    def add_user(self, details):
        conn = sqlite3.connect('UserDb.db')
        cur = conn.cursor() 

        cur.execute(f'''INSERT INTO users ( email, full_name, password) 
                    VALUES ( '{details["email"]}', '{details["full_name"]}', '{details["password"]}')''')
        
        # Total_users += 1
       
        # Committing the changes to the database
        conn.commit()
        conn.close()

    def create_docs(self, user_text, title):
        # split it into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        docs = text_splitter.split_text(user_text)

        langchain_docs = []
        for doc in docs:
            ld = Document(page_content=doc, metadata={
                "userName" :self.email,
                "title" : title
            })

            langchain_docs.append(ld)
        return langchain_docs

    def addDocs_2_vetordb(self, docs, id):

        if not os.path.exists("./chroma_db"):
            self.vectorDB = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db", collection_name="collection_"+str(id))
        else:
            # load from disk
            self.vectorDB = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function, collection_name="collection_"+str(id))
            self.vectorDB.add_documents(docs)

    def query_VectorDb(self, query, id):
        print("\nid===",str(id), "\n")
        self.vectorDB = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function, collection_name="collection_"+str(id))
        retriever = self.vectorDB.as_retriever(search_type="mmr",search_kwargs={"k": 4})
        docs = retriever.get_relevant_documents(query)
        return docs


@app.route('/login', methods = ['GET'])
def login():
    login_details = request.get_json()

    obj = Summer(login_details['email'])
    user_details = obj.get_user_details()

    if len(user_details) == 0:
        return jsonify({'log': 'No Details Found'})
    elif login_details['password'] == user_details[0][-1]:
        return jsonify({"log":True})
    else:
        return jsonify({'log':'Wrong Password'}), 200


@app.route('/update_user', methods = ['GET'])
def update_user():
    update_data = request.get_json()

    summer_obj = Summer(update_data['email'])

    # check if the user already exist
    user_details = summer_obj.get_user_details()
    if len(user_details) == 0:
        summer_obj.add_user(update_data)
    else:
        return jsonify({'log': 'Exists'})
    
    return jsonify({'log':'Added New User Successfully'})

@app.route('/summarize', methods=['GET', 'POST'])
def get_summary():
    user_input = request.get_json()
    obj = Summer(user_input['email'])
    user_details = obj.get_user_details()

    title = title_chain.invoke({"input_text": user_input['text']})
    docs = obj.create_docs(user_input['text'], title)
    obj.addDocs_2_vetordb(docs, user_details[0][0])

    try:
        summary = chain.invoke({"input_text": user_input['text']})
    except Exception as e:
        summary = ''
        print(e)
        
    if len(user_input['text']) < 500:
        docs = obj.query_VectorDb(user_input['text'], user_details[0][0])
        qa_chain_res = qa_chain({"input_documents": docs, 
                        "question":[ user_input['text']]})
        
        print(qa_chain_res['output_text'])
    return jsonify({'summary':summary}), 200


# main driver function
if __name__ == '__main__':
    
    app.run(debug=True, host='0.0.0.0')