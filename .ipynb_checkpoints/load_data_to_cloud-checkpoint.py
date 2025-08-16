import os
import openai
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from website_scraper import webScraper
from pdf_chunker import PDFChunkerForQdrant

# --- Load credentials from environment variables ---
# This is a more secure practice than hardcoding keys in the script.
# See the instructions below the code on how to set these variables.
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set the OpenAI API key for LangChain and the OpenAI client
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    openai.api_key = OPENAI_API_KEY

def main():
    """
    Main function to scrape data, create embeddings, and load to Qdrant.
    """
    print("Starting the data loading process...")

    # --- API Key and Connection Validation Block ---
    try:
        print("Validating OpenAI API key and connection...")
        # Make a lightweight API call to test the key and connection
        client = openai.OpenAI()
        client.models.list()
        print("OpenAI API key is valid and connection is successful.")
    except openai.AuthenticationError:
        print("ERROR: OpenAI API key is invalid or incorrect. Please check your credentials.")
        return # Stop execution if the key is wrong
    except openai.APIConnectionError as e:
        print(f"ERROR: Failed to connect to OpenAI API. Please check your network connection, firewall, or proxy settings.")
        print(f"Underlying error: {e.__cause__}")
        return # Stop execution if connection fails
    # --- End of Validation Block ---

    # 1. Scrape PDFs and chunk them
    chunker = PDFChunkerForQdrant(max_char_limit=5000)
    scraper = webScraper("user")
    
    print("Scraping website for PDF URLs and processing documents...")
    documents = scraper.getWebsitePdfUrls(chunker)
    
    if not documents:
        print("No documents were processed. Exiting.")
        return

    print(f"\nSuccessfully processed {len(documents)} documents. Now loading to Qdrant Cloud...")
    
    # 2. Initialize embeddings
    embeddings = OpenAIEmbeddings()
    
    # 3. Load documents into Qdrant Cloud
    collection_name = "medicaid_app"
    
    print(f"Attempting to load documents into Qdrant collection: '{collection_name}'...")
    qdrant_vectorstore = Qdrant.from_documents(
        documents,
        embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=collection_name,
        force_recreate=True, # Use True to start fresh. Use False to add to an existing collection.
    )

    print(f"\nFinished persisting {len(documents)} documents to Qdrant Cloud in collection '{collection_name}'.")

if __name__ == "__main__":
    # Check if all required environment variables are loaded before running main()
    if not all([QDRANT_URL, QDRANT_API_KEY, OPENAI_API_KEY]):
        print("---")
        print("ERROR: One or more required environment variables are not set.")
        print("Please set QDRANT_URL, QDRANT_API_KEY, and OPENAI_API_KEY before running the script.")
        print("---")
    else:
        main()