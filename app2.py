import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv

# Load the environment variable (GOOGLE_API_KEY) for Google Gemini
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

def get_pdf_text_and_metadata(pdf_docs):
    combined_text = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        metadata = pdf.name
        for page in pdf_reader.pages:
            combined_text.append({"text": page.extract_text(), "metadata": metadata})
    return combined_text

def get_text_chunks_with_metadata(combined_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks_with_metadata = []
    for entry in combined_text:
        chunks = text_splitter.split_text(entry["text"])
        for chunk in chunks:
            chunks_with_metadata.append(Document(page_content=chunk, metadata={"source": entry["metadata"]}))
    return chunks_with_metadata

def get_vector_store(chunks_with_metadata):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(chunks_with_metadata, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are a diabetes expert with extensive knowledge in diabetes prevention, management, and treatment. 
    Answer questions intelligently using the provided context, while also providing actionable advice. Tailor your response as follows:

    1. **Precise Answer**: Provide a direct, concise response to the query based on the context.
    2. **Actionable Advice for Laypersons**: Include lifestyle, dietary, or medical advice that the user can apply practically to manage or improve their condition.
    3. **Insights for Medical Professionals**: Highlight detailed clinical insights, such as diagnostic criteria, treatment protocols, or research findings, where applicable.
    4. **Agreement Across Guidelines**: Compare the information across the sources to highlight similarities or agreements.
    5. **Discrepancies Across Guidelines**: Note any differences or conflicts in the information.

    Context: {context}
    Question: {question}
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=5)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    response_text = response["output_text"].strip()

    invalid_answers = [
        "I don't have enough information",
        "The answer is not available",
        "Answer is not available in the context",
        "No relevant information found",
        "I cannot provide an answer",
        "The context provided does not define"
    ]
    
    if any(phrase in response_text for phrase in invalid_answers) or not response_text:
        st.write("### Generated Answer:")
        st.write("The answer to your question is not available in the provided documents.")
    else:
        st.write("### Generated Answer:")
        st.write(response_text)
        st.write("### Sources:")
        for doc in docs:
            st.write(f"- {doc.metadata.get('source', 'Unknown source')}")

def main():
    st.set_page_config("Diabetes Expert Assistant")
    st.header("Interactive Diabetes Knowledge Assistant 💁")

    user_question = st.text_input("Ask a Question (Diabetes Expert)")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                combined_text = get_pdf_text_and_metadata(pdf_docs)
                text_chunks = get_text_chunks_with_metadata(combined_text)
                get_vector_store(text_chunks)
                st.success("PDFs processed and vector store created!")

if __name__ == "__main__":
    main()
