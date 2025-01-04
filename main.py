import os
import re
from transformers import AutoTokenizer, AutoModel
import uuid
import torch
import numpy as np
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI


def chunking(directory_path, tokenizer, chunk_size, para_seperator=" /n /n", separator=" "):

    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    documents = {}
    all_chunks = {}
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        print(filename)
        base = os.path.basename(file_path)
        sku = os.path.splitext(base)[0]
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            doc_id = str(uuid.uuid4())    

            paragraphs = re.split(para_seperator, text)

            for paragraph in paragraphs:
                words = paragraph.split(separator)
                current_chunk_str = ""
                chunk = []
                for word in words:
                    if current_chunk_str:
                        new_chunk = current_chunk_str + separator + word
                    else:
                        new_chunk = current_chunk_str + word    
                    if len(tokenizer.tokenize(new_chunk)) <= chunk_size:
                        current_chunk_str = new_chunk
                    else:
                        if current_chunk_str:
                            chunk.append(current_chunk_str)
                        current_chunk_str = word
                

                if current_chunk_str:   
                    chunk.append(current_chunk_str)

                for chunk in chunk:
                    chunk_id = str(uuid.uuid4())
                    all_chunks[chunk_id] = {"text": chunk, "metadata": {"file_name":sku}}
        documents[doc_id] = all_chunks
    return documents    

