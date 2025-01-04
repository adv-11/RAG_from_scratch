import os
import re
from transformers import AutoTokenizer, AutoModel
import uuid
import torch
import numpy as np
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI


def chunking(directory_path, model_name, tokenizer, chunk_size, para_seperator=" /n /n", separator=" "):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    documents = {}
    all_chunks = {}
    
    # Debug: Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return {}

    if not os.listdir(directory_path):
        print(f"Directory '{directory_path}' is empty.")
        return {}

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        print(f"Processing file: {filename}")
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

                for chunk_item in chunk:
                    chunk_id = str(uuid.uuid4())
                    all_chunks[chunk_id] = {"text": chunk_item, "metadata": {"file_name": sku}}
        documents[doc_id] = all_chunks
    
    # Debug: Check if documents were created
    print(f"Documents created: {len(documents)}")
    if not documents:
        print("No documents created. Check chunking logic.")
    return documents


def map_document_embeddings(documents, tokenizer, model):
    mapped_document_db = {}

    for id, dict_content in documents.items():
        mapped_embeddings = {}

        for content_id, text_content in dict_content.items():
            text = text_content.get("text")

            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

            with torch.no_grad():
                embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()

            mapped_embeddings[content_id] = embeddings

        mapped_document_db[id] = mapped_embeddings

    # Debug: Check mapped embeddings
    print(f"Mapped embeddings created for {len(mapped_document_db)} documents.")
    return mapped_document_db


def compute_embeddings(query, tokenizer, model):
    query_inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    query_embeddings = model(**query_inputs).last_hidden_state.mean(dim=1).squeeze()
    query_embeddings = query_embeddings.tolist()
    print("Query embeddings generated:", query_embeddings)
    if not query_embeddings:
        print("Query embeddings not generated. Check tokenizer or model.")
    return query_embeddings


def calculate_cosine_similarity_score(query_embeddings, chunk_embeddings):
    normalized_query = np.linalg.norm(query_embeddings)
    normalized_chunk = np.linalg.norm(chunk_embeddings)
    if normalized_chunk == 0 or normalized_query == 0:
        return 0
    else:
        return np.dot(chunk_embeddings, query_embeddings) / (normalized_chunk * normalized_query)


def retrieve_top_k_scores(query_embeddings, mapped_document_db, top_k):
    scores = {}

    for doc_id, chunk_dict in mapped_document_db.items():
        for chunk_id, chunk_embeddings in chunk_dict.items():
            chunk_embeddings = np.array(chunk_embeddings)
            score = calculate_cosine_similarity_score(query_embeddings, chunk_embeddings)
            scores[(doc_id, chunk_id)] = score

    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

    # Debug: Check sorted scores
    print("Sorted scores:", sorted_scores)
    return sorted_scores


def retrieve_top_results(sorted_scores):
    top_results = []
    for ((doc_id, chunk_id), score) in sorted_scores:
        results = (doc_id, chunk_id, score)
        top_results.append(results)
    print("Top results:", top_results)
    if not top_results:
        print("No relevant results found.")
    return top_results


def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {path}")


def read_json(path):
    if not os.path.exists(path):
        print(f"File '{path}' does not exist.")
        return {}
    with open(path, 'r') as f:
        data = json.load(f)
    print(f"Data read from {path}")
    return data


def retrieve_text(top_results, document_data):
    if not top_results:
        print("Top results are empty. Cannot retrieve text.")
        return None

    first_match = top_results[0]
    doc_id = first_match[0]
    chunk_id = first_match[1]
    related_text = document_data.get(doc_id, {}).get(chunk_id, None)
    print("Retrieved text:", related_text)
    if not related_text:
        print("No text retrieved. Check document structure.")
    return related_text


def generate_llm_response(mistralai_model, query, relevant_text):
    if not relevant_text:
        print("Relevant text is empty. Cannot generate response.")
        return None

    template = """
    You are a search engine. You will be provided with some retrieved context, as well as the user query.

    Your job is to understand the query, and answer based on the retrieved context. If you can't find the answer, you can say "I don't know".
    Here is context:

    <context>

    {context}

    </context>

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template=template)
    chain = prompt | mistralai_model
    response = chain.invoke({"context": relevant_text["text"], "question": query})
    print("LLM Response:", response)
    if not response:
        print("No response generated by MistralAI.")
    return response


def main():
    try:
        directory_path = "documents"
        model_name = "BAAI/bge-small-en-v1.5"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        chunk_size = 200
        para_seperator = " /n /n"
        separator = " "
        top_k = 2
        mistralai_model = ChatMistralAI(model="open-mistral-7b", api_key=os.environ.get("MISTRAL_API_KEY"))

        print("Creating document store with chunk ID, doc ID, and text.")
        documents = chunking(directory_path, model_name, tokenizer, chunk_size, para_seperator, separator)

        print("Now generating embeddings and mapping in database.")
        mapped_document_db = map_document_embeddings(documents, tokenizer, model)

        print("Saving document store to JSON.")
        save_json('database/doc_store_2.json', documents)
        save_json('database/vector_store_2.json', mapped_document_db)

        print("Retrieving most relevant data chunks.")
        query = "What is RAG?"
        query_embeddings = compute_embeddings(query, tokenizer, model)
        sorted_scores = retrieve_top_k_scores(query_embeddings, mapped_document_db, top_k)
        top_results = retrieve_top_results(sorted_scores)

        print("Reading document store from JSON.")
        document_data = read_json("database/doc_store_2.json")

        print("Retrieving text of relevant chunk embeddings.")
        relevant_text = retrieve_text(top_results, document_data)

        print("Generating LLM response.")
        response = generate_llm_response(mistralai_model, query, relevant_text)
        print("Final response:", response)

    except Exception as e:
        print("An error occurred:", e)


if __name__ == "__main__":
    main()
