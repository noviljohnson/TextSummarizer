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
tokenizer = T5Tokenizer.from_pretrained(checkpoint )
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint , device_map='auto', offload_folder="offload",torch_dtype = torch.float32)

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
title_template =""" give a title for the given input text.
        Input text : {input_text}
        return the answer in the following format.
        Answer: 
        """
prompt = PromptTemplate.from_template(template)
title_prompt = PromptTemplate.from_template(title_template)

chain = prompt | hf
title_chain = title_prompt | hf
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

    def query_VectorDb(self, query):
        self.vectorDB = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function, collection_name="collection_"+str(id))
        docs = self.vectorDB.as_retriever().get_relevant_documents(query)
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

    summary = chain.invoke({"input_text": user_input['text']})
    
    return jsonify({'summary':summary}), 200


# main driver function
if __name__ == '__main__':
    
    app.run(debug=True, host='0.0.0.0')