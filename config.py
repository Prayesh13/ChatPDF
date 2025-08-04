from pdfminer.high_level import extract_text
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile

class process_pdf():

    @staticmethod
    def get_pdf_text(pdf_docs):
        text = ""
        for pdf in pdf_docs:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf.read())
                tmp_file.flush()
                extracted_text = extract_text(tmp_file.name)
                text += extracted_text
        return text

    @staticmethod
    def get_text_chunks(text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
        return text_splitter.split_text(text)

    @staticmethod
    def get_vector_store(text_chunks):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")

    @staticmethod
    def get_conversational_chain():
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
        If the answer is not in provided context just say "answer is not available in the context". 
        Do not make up information.\n\n
        Context:\n{context}\n
        Question:\n{question}\n
        Answer:
        """
        model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.2)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        return load_qa_chain(model, chain_type="stuff", prompt=prompt)
    