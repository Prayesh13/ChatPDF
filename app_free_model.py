import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config2 import process_pdf
import html

load_dotenv(override=True)

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Streamlit App Config
st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📚")


def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        docs = new_db.similarity_search(user_question)

        chain = process_pdf.get_conversational_chain()

        response = chain.invoke({
            "input_documents": docs,
            "question": user_question
            })


        answer = response["output_text"] if "output_text" in response else response.get("answer", "❌ No valid answer returned.")

        # Display nicely formatted response
        st.markdown("### 🤖 Chatbot Reply")
        st.markdown(
                    f"""
                    <div style='
                        background-color: #1e1e1e;
                        padding: 1.2rem;
                        border-radius: 10px;
                        font-size: 1.05rem;
                        line-height: 1.6;
                        color: #ffffff;
                        font-family: "Segoe UI", "Roboto", "Helvetica Neue", sans-serif;
                        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                        margin-top: 1rem;
                        border-left: 5px solid #00c7a3;
                    '>
                        {answer}

                    </div>
                    """,
                    unsafe_allow_html=True
                )


    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.stop()


def main():
    st.header("📚 PDF Chatbot 🤖")
    st.write("Ask questions about your uploaded PDF files!")

    user_question = st.text_input("📝 Type your question here:")
    generate_button = st.button("Generate Answer")

    if user_question:
        if os.path.exists("faiss_index/index.faiss"):
            if generate_button:
                user_input(user_question)
        else:
            st.warning("⚠️ Please upload PDF(s) and click Submit before asking questions.")

    with st.sidebar:
        st.image("image.png", use_column_width=True)
        st.markdown("---")
        st.title("📁 Upload PDF Files")

        pdf_docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Submit"):
            if not pdf_docs:
                st.warning("⚠️ Please upload at least one PDF file.")
            else:
                with st.spinner("⚙️ Processing PDFs..."):
                    raw_text = process_pdf.get_pdf_text(pdf_docs)
                    text_chunks = process_pdf.get_text_chunks(raw_text)
                    process_pdf.get_vector_store(text_chunks)
                    st.success("✅ PDFs processed and vector store created!")


if __name__ == "__main__":
    main()
