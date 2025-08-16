import streamlit as st
import rag_handler_langchain

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

# Use a form to group the text input and button together
with st.form(key="question_form"):
    question = st.text_input(
        "Your Question:", 
        placeholder="e.g., How are applications for medical assistance processed?"
    )
    # The submit button for the form
    submit_button = st.form_submit_button(label="Get Answer")

# The logic is now executed only when the form's submit button is clicked
if submit_button:
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
*   How is eligibility of QMB determined?
*   Tell me about continued medicaid
*   How to establish non-financial eligibility for QI program?
""")