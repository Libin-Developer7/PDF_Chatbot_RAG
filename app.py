import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


st.set_page_config(page_title="Local PDF Chatbot", page_icon="📚")
st.title("📚 100% Free Local PDF Chatbot")
st.write("Powered by Ollama (Llama 3 & Nomic Embeddings) running locally on your hardware.")

uploaded_file = st.file_uploader("Upload your PDF Book / Document", type="pdf")

if uploaded_file:
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:   
        f.write(uploaded_file.getbuffer())  

    st.info("Processing your PDF locally... (This might take a minute for large files)")

    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200     
    )
    splits = text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings
    )
    
 
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8}) 

    st.success("PDF processed successfully! Ask your questions below.")


    user_query = st.text_input("What would you like to know from this document?")

    if user_query:

        llm = ChatOllama(
            model="llama3.2", 
            temperature=0     
        )

        system_prompt = (
            "You are an expert assistant designed to answer questions strictly based on the provided documents.\n"
            "If you do not know the answer based on the context, say that you don't know. Do not hallucinate.\n\n"
            "Context:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        with st.spinner("Searching local vector database and running model inference..."):
            response = rag_chain.invoke({"input": user_query})

        st.markdown("### 🤖 Local AI Answer:")
        st.write(response["answer"])

        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


