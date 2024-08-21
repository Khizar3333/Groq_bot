import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
import time

load_dotenv()

# load groq apikey
groq_api_key=os.environ["GROQ_API_KEY"]

if "vector" not in st.session_state:
    st.session_state.embeddings = OpenAIEmbeddings()
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs=st.session_state.loader.load()
    st.session_state.text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
        
    )

    st.session_state.final_docs=st.session_state.text_splitter.split_documents(
        st.session_state.docs[:50]
    )
    st.session_state.vectors=FAISS.from_documents(
        st.session_state.final_docs,
        st.session_state.embeddings
    )

st.title("ChatGroq") 
llm = ChatGroq(groq_api_key=groq_api_key,model_name="mixtral-8x7b-32768")

prompt=ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question.
    if user ask question that is not in the context, please respond "I don't have access to this information do you have any question related to langsmith"

    <context>
    {context}
    <context>

    Question: {input}




  """
)

document_chain=create_stuff_documents_chain(llm,prompt)
retriever=st.session_state.vectors.as_retriever()
retrievel_chain=create_retrieval_chain(retriever,document_chain)

prompt=st.text_input("Input your prompt here")

if prompt:
    start=time.process_time()
    response=retrievel_chain.invoke({"input":prompt})
    print("response time",time.process_time()-start)
    st.write(response['answer'])

    # with streamlit expander
    with st.expander("Document similarity search"):
        for i,doc in enumerate(response['context']):
         st.write(doc.page_content)
         st.write("------------------")

