import streamlit as st
from adaptive_rag import AdaptiveRAG

# Initialize session state
if 'rag' not in st.session_state:
    st.session_state.rag = AdaptiveRAG()

st.title("Adaptive RAG Q&A System")

# Question answering section
st.header("Ask Questions")
question = st.text_input("Enter your question")

if st.button("Get Answer"):
    with st.spinner("Thinking..."):
        result = st.session_state.rag.query(question)
        
        # Display processing information
        st.subheader("Processing Information")
        st.write(f"**Documents Used:** {result['documents_used']}")
        st.write(f"**Documents Retrieved from RAG:** {'Yes' if result['retrieved'] else 'No'}")
        st.write(f"**Web Search Used:** {'Yes' if result['web_search_used'] else 'No'}")
        
        # Display answer
        st.subheader("Answer")
        st.write(result['answer'])
