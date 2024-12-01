import tree_sitter_python as tspython
from tree_sitter import Language, Parser
import ast
import re
import json
from data_manager import DataManager
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)
import os
import logging
import tiktoken
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
load_dotenv()

tokenizer = tiktoken.get_encoding("cl100k_base")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

max_tokens=500
def parser_function(file_path):
    with open(file_path)as f:
        data=f.read()
  
    data=data.replace('\\','/')
    data=re.sub(r'^!.*\n','',data,flags=re.MULTILINE)
    tree = ast.parse(data)
    functions=[(node.name,node.lineno) for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name!='__init__']
    line_no_to_chunk=[]
    for item in functions:
        line_no_to_chunk.append(item[1]-1)
    line_no_to_chunk.insert(0,0)
    with open(file_path) as f:
        data2=f.readlines()

    chunks_to_embed=[]
    for i,item in enumerate(line_no_to_chunk[1:]):
        if i>0:
            chunk_text=data2[line_no_to_chunk[i-1]:item]
            chunks_to_embed.append(''.join(chunk_text))
    chunks_to_embed.append(''.join(data2[line_no_to_chunk[-1]:]))
    print('No of chunks',len(chunks_to_embed))
    token_split_texts = []
    for item in chunks_to_embed:
        tokens=tokenizer.encode(item)
        if len(tokens)<max_tokens:
            token_split_texts.append(item)
        else:
            for i in range(0,len(tokens),max_tokens):
                token_split_texts.append(tokenizer.decode(tokens[i:i+max_tokens]))
        

    print('No of chunks after processing',len(token_split_texts))
    return token_split_texts


def insert_to_chroma(tokenized_text,index_name):
    client = chromadb.PersistentClient(path="./chroma")
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ['OPENAI_API_KEY'],
                model_name="text-embedding-3-small"
            )
    chroma_collection = client.get_or_create_collection("repo", embedding_function=openai_ef)
    for i,data in enumerate(tokenized_text):
        #print(i,data)
        chroma_collection.add(ids=index_name+str(i), documents=data)
    print('Inserted',index_name)
    print(chroma_collection.count())



def helper(data_manager: DataManager):
    local_path = os.path.join(data_manager.local_dir, data_manager.repo_id)
    print(local_path)
    for root, _, files in os.walk(local_path):
        file_paths = [os.path.join(root, file) for file in files if file.lower().endswith(".py")]
        for item in file_paths:
            tokenized_text=parser_function(item)
            #insert_to_chroma(tokenized_text,item)

