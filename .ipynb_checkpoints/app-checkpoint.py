import streamlit as st
import rag_handler

# Get secrets from Streamlit's secrets management
# These will be set in the Streamlit Community Cloud dashboard
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# --- Streamlit UI ---
st.set_page_config(page_title="Medicaid Policy Q&A", layout="wide")
st.title("Louisiana Medicaid Policy Q&A App")

st.markdown("""
Enter your Louisiana Medicaid policy question below. The system will search through official policy documents
to find the most relevant information and generate an answer.
""")

question = st.text_input("Your Question:", placeholder="e.g., How are applications for medical assistance processed?")

if st.button("Get Answer"):
    if question:
        with st.spinner("Searching documents and generating answer..."):
            # Pass the secrets to the handler function
            answer = rag_handler.get_final_answer(
                question,
                QDRANT_URL,
                QDRANT_API_KEY,
                OPENAI_API_KEY
            )
        st.success("Answer:")
        st.markdown(answer) # Use markdown to render formatted text
    else:
        st.warning("Please enter a question.")
        
st.markdown("---")
st.markdown("""
**Sample questions you can ask:**
*   What are the responsibilities of home owners?
*   What are the rules and regulations of the HOA?
*   What do the documents say about decorations on the outside of the homes?
*   What zone is Expedition street?
*   Are sports courts allowed in Marlboro?
*   How do I dispose of batteries?
*   What are the recyling dates for Zone 4 in August?
""")