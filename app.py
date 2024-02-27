import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import faiss, chroma, tair
import getpass
import os
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
import openai
import chromadb
from llama_index.core import StorageContext
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template




file_path = file_path = os.path.join("data", "docs.txt")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)#creats pdf object that has pages which helps us to read from
        for page in pdf_reader.pages:
            text += page.extract_text()#this functions extracts raw text for us
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", #I want them seperated with new line
        chunk_size= 1000, #as 1000 characters
        chunk_overlap=200,
        length_function=len
    
    )
    chunks = text_splitter.split_text(text) #it is in a list format 
    return chunks

def write_to_txt_file(text_chunks, path):
    # Check if the file exists
    if os.path.exists(path):
        mode = 'a'  # Append mode
    else:
        mode = 'w'  # Create and write mode

    with open(path, mode, encoding='utf-8') as file:
        for chunk in text_chunks:
            file.write(chunk + '\n')


def get_vectorstore(text_chunks):
    write_to_txt_file(text_chunks, file_path)

    
    chroma_client = chromadb.EphemeralClient()

    if "quickstart" in chroma_client.list_collections():
        chroma_client.delete_collection("quickstart")

    # Create a new collection

    chroma_collection = chroma_client.create_collection("quickstart")
    embeddings = OpenAIEmbeddings()

    vectorstore = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vectorstore)
    documents = SimpleDirectoryReader("./data/").load_data()
    # Create the VectorStoreIndex with documents
    index = VectorStoreIndex.from_documents(documents, embed_model=embeddings, storage_context=storage_context)

    return index

def get_conversatoin_chain(vectorstore):
    llm =  ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        
        #retriever=vectorstore.as_retriever(),
        retriever=vectorstore.as_retriever(),
        memory=memory

    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.write(response)



def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = api_key
    st.set_page_config(page_title="Chat with your PDFs",
                        page_icon=":books:")
    
    st.write(css, unsafe_allow_html=True)

    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None


    st.header("Chat with your PDFs :books:")
    user_question = st.text_input("Ask a question about your documents: ")
    if user_question:
        handle_userinput(user_question)


    st.write(user_template.replace("{{MSG}}", "Hello Alferd"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello Master"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documnets")
         
        pdf_docs = st.file_uploader(
            "Upload your desired PDFs here and click on 'Process'", accept_multiple_files=True)#we can add mutiple files 
        
        if st.button("Process"):
            with st.spinner("Processing"): #adding this for form user friendly (to show the programme is actually running)
                #we need to get the PDFs first
                raw_text = get_pdf_text(pdf_docs)
            
                #getting the text chunks (converted)
                texts_chunks = get_text_chunks(raw_text)

                
                #creating vector store
                vectorstore = get_vectorstore(texts_chunks)

                #creating conversation chain
                st.session_state.conversation = get_conversatoin_chain(vectorstore) #session state helps for the vaiable to not get reinitialised







if __name__ == "__main__":
    main()