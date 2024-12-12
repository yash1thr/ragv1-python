from tree_sitter import Language, Parser
import ast
import re
import json
from data_manager import DataManager
import os
import logging
import tiktoken
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
import esprima
import javalang
load_dotenv()
from subprocess import check_output
tokenizer = tiktoken.get_encoding("cl100k_base")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

max_tokens=500

def python_ast(data):
    #data=data.replace('\\','/')
    data=re.sub(r'^!.*\n','',data,flags=re.MULTILINE)
    tree = ast.parse(data)
    functions=[(node.name,node.lineno) for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name!='__init__']
    line_no_to_chunk=[]
    for item in functions:
        line_no_to_chunk.append(item[1]-1)
    line_no_to_chunk.insert(0,0)
    return line_no_to_chunk



def js_ast(typescript_code):
    #Language.build_library("build/my-languages.so", ["tree-sitter-typescript/typescript"])
    TS_LANGUAGE = Language("build/my-languages.so", "typescript")
    parser = Parser()
    parser.set_language(TS_LANGUAGE)
    def traverse_tree(node, code, functions):
        """
        Traverse the AST and extract functions.
        """
        if node.type in ["function_declaration", "method_definition", "arrow_function", "function_expression"]:
            functions.append(node.start_point[0])
        for child in node.children:
            traverse_tree(child, code, functions)

    functions = []
    tree = parser.parse(bytes(typescript_code, "utf8"))
    traverse_tree(tree.root_node, typescript_code, functions)
    
    return functions

def java_ast(java_code):
    tree = javalang.parse.parse(java_code)
    functions = []
    for path, node in tree.filter(javalang.tree.MethodDeclaration):
        if hasattr(node, 'name') and hasattr(node, 'position') and node.position:
            functions.append(node.position.line)
    
    return functions

def parser_function(file_path):
    with open(file_path)as f:
        data=f.read()
    if file_path.endswith('.ts'):
        line_no_to_chunk=js_ast(data)
        print(line_no_to_chunk)
    elif file_path.endswith('.java'):
        line_no_to_chunk=java_ast(data)
    else:
        line_no_to_chunk=python_ast(data)
    with open(file_path) as f:
        data2=f.readlines()
    chunks_to_embed=[]
    if len(line_no_to_chunk)<=1:
        return [data]
    else:
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
        file_paths = [os.path.join(root, file) for file in files if file.lower().endswith((".py",".ts",".java"))]
        for item in file_paths:
            print(item)
            tokenized_text=parser_function(item)
            #insert_to_chroma(tokenized_text,item)

