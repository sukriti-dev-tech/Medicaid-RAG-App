import os
import textwrap
from bs4 import BeautifulSoup

# LangChain & Qdrant imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Qdrant client for vector store initialization
from qdrant_client import QdrantClient

# Langsmith for logging and tracing
from langsmith import traceable

# To enable Langsmith tracing, set the following environment variables:
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = "YOUR_LANGSMITH_API_KEY"
# os.environ["LANGCHAIN_PROJECT"] = "YOUR_PROJECT_NAME" # Optional: "default" is used if not set

# This function will be the main entry point for the Streamlit app
@traceable(name="RAG Pipeline")
def get_final_answer(user_question: str, qdrant_url: str, qdrant_api_key: str, openai_api_key: str) -> str:
    """
    Main function to execute the RAG process using LangChain and log with Langsmith.
    """
    try:
        # 1. Initialize LangChain components
        # Initialize embeddings model
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            api_key=openai_api_key
        )

        # Initialize Qdrant client and LangChain vector store
        qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        vector_store = Qdrant(
            client=qdrant_client,
            collection_name="medicaid_app",
            embeddings=embeddings
        )

        # Create a retriever to fetch relevant documents
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # 2. Retrieve relevant documents from Qdrant
        retrieved_docs = retriever.invoke(user_question)

        if not retrieved_docs:
            return "Could not find any relevant documents in the database to answer the question."

        # 3. Prepare context and source URLs from retrieved documents
        context_str = ""
        source_urls = []
        for doc in retrieved_docs:
            soup = BeautifulSoup(doc.page_content, "html.parser")
            page_content_text = soup.get_text(separator=" ", strip=True)
            file_name = doc.metadata.get('file_name', 'N/A')

            context_str += f"Source (File: {file_name}):\n{page_content_text}\n---\n"
            source_urls.append(f"https://ldh.la.gov/assets/medicaid/MedicaidEligibilityPolicy/{file_name}")

        unique_urls = sorted(list(set(source_urls)))

        # 4. Generate a complete answer using the retrieved context with a LangChain chain
        # Define the prompt template
        system_prompt = textwrap.dedent("""
            You are a helpful AI assistant. Your task is to answer the user's question based ONLY on the provided context.
            Do not use any external knowledge.
            If the context does not contain the answer, state that you cannot answer based on the provided information.
        """)
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Context:\n{context}\nQuestion: {question}")
        ])

        # Initialize the language model
        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.0,
            api_key=openai_api_key
        )

        # Create the generation chain using LangChain Expression Language (LCEL)
        rag_chain = prompt_template | llm | StrOutputParser()
        
        # Invoke the chain
        answer = rag_chain.invoke({"context": context_str, "question": user_question})

        # Append the unique URLs to the final answer
        answer += "\n\n**Files Referred:**\n" + "\n".join([f"- {url}" for url in unique_urls])
        
        return answer

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        return f"An error occurred while processing your request: {e}"