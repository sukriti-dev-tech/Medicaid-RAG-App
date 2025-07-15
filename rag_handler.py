import os
import textwrap
from openai import OpenAI
from qdrant_client import QdrantClient
from bs4 import BeautifulSoup

# This function will be the main entry point for the Streamlit app
def get_final_answer(user_question: str, qdrant_url: str, qdrant_api_key: str, openai_api_key: str) -> str:
    """Main function to execute the RAG process."""
    try:
        # 1. Initialize API clients with credentials
        openai_client = OpenAI(api_key=openai_api_key)
        qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

        # 2. Retrieve relevant documents from Qdrant
        search_results = perform_qdrant_search(user_question, qdrant_client, openai_client)

        if not search_results:
            return "Could not find any relevant documents in the database to answer the question."

        # 3. Generate a complete answer using the retrieved context
        final_answer = generate_rag_answer(user_question, search_results, openai_client)
        return final_answer

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        return f"An error occurred while processing your request: {e}"

def perform_qdrant_search(query, qdrant_client, openai_client):
    """Searches Qdrant and returns the search results."""
    collection_name = "langchain_qdrant"
    model_name = "text-embedding-ada-002"
    
    response = openai_client.embeddings.create(input=query, model=model_name)
    query_vector = response.data[0].embedding

    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=3,
        with_payload=True
    )
    return search_results

def generate_rag_answer(query, search_results, openai_client):
    """Generates an answer using OpenAI with the provided search results as context."""
    context_str = ""
    source_urls = []
    for result in search_results:
        payload = result.payload
        soup = BeautifulSoup(payload.get('page_content', ''), "html.parser")
        page_content_text = soup.get_text(separator=" ", strip=True)
        file_name = payload.get('metadata', {}).get('file_name', 'N/A')
        
        context_str += f"Source (File: {file_name}):\n{page_content_text}\n---\n"
        # Append the full URL
        source_urls.append(f"https://ldh.la.gov/assets/medicaid/MedicaidEligibilityPolicy/{file_name}")

    unique_urls = sorted(list(set(source_urls)))
    
    system_prompt = textwrap.dedent("""
        You are a helpful AI assistant. Your task is to answer the user's question based ONLY on the provided context.
        Do not use any external knowledge. After providing the answer, list all the source document URLs under a 'Files Referred:' heading.
        If the context does not contain the answer, state that you cannot answer based on the provided information.
    """)
    
    user_prompt = f"Context:\n{context_str}\nQuestion: {query}"

    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    
    answer = response.choices[0].message.content
    # Append the unique URLs to the final answer
    answer += "\n\n**Files Referred:**\n" + "\n".join([f"- {url}" for url in unique_urls])
    
    return answer
