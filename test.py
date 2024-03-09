import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import fitz
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import faiss, chroma, tair
import getpass
import os
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, Document
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
import openai
import chromadb
from llama_index.core import StorageContext
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template




file_path = os.path.join("data", "docs.txt")
folder_path = os.path.join("data")

def pdf_to_text(pdf_content, folder_path):
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)
    for pdf in pdf_content:
        # Extract text from the PDF content
        pdf_reader = PdfReader(pdf)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()

        # Create a text file name based on the PDF file name
        pdf_file_name = os.path.basename(pdf.name)
        text_file_name = os.path.splitext(pdf_file_name)[0] + ".txt"

        # Check if the text file already exists in the provided folder
        text_file_path = os.path.join(folder_path, text_file_name)
        if not os.path.exists(text_file_path):
            # Save the text to the new text file
            with open(text_file_path, 'w', encoding='utf-8') as text_file:
                text_file.write(text)


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



@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on the Streamlit Python library and your job is to answer technical questions. Assume that all questions are related to the documents processed or provided to the streamlit input. Keep your answers technical and based on facts – do not hallucinate features."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index


def chat(index, query):
    
    chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
    response = chat_engine.chat(query)
    return response

def handle_userinput(user_question):
    vectorstore = load_data()
    query = user_question
    st.session_state.messages.append({"role": "user", "content": query})

    # Display the chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
        elif message["role"] == "assistant":
            with st.chat_message("assistant"):
                st.write(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat(vectorstore, query)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message)  # Add response to message history

                st.write(user_template.replace("{{MSG}}", query), unsafe_allow_html=True)
                st.write(bot_template.replace("{{MSG}}", response.response), unsafe_allow_html=True)



   


#"""def handle_userinput(user_question):
 #   response = st.session_state.conversation({'question': user_question})
  #  st.write(response)"""



def main():
    
    st.set_page_config(page_title="Chat with your PDFs",
                        page_icon=":books:")
    
    st.write(css, unsafe_allow_html=True)
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = api_key

    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None


    st.header("Chat with your PDFs :books:")
    if "messages" not in st.session_state.keys(): # Initialize the chat message history
        st.session_state.messages = [
            {"role": "assistant", "content": "This is Alferd, How may I assist you Young Master!"}
    ]
    user_question = st.text_input("Ask a question about your documents: ")
    if user_question:
        handle_userinput(user_question)

    

    with st.sidebar:
        st.subheader("Your documnets")
         
        pdf_docs = st.file_uploader(
            "Upload your desired PDFs here and click on 'Process'", accept_multiple_files=True)#we can add mutiple files 
        
        if st.button("Process"):
            with st.spinner("Processing"): #adding this for form user friendly (to show the programme is actually running)
                #we need to get the PDFs first
                #raw_text = get_pdf_text(pdf_docs)
            
                #getting the text chunks (converted)
                pdf_to_text(pdf_docs, folder_path)

                
                #creating vector store

            


               
                   



    #creating conversation chain
    #st.session_state.conversation = chat(vectorstore, query) #session state helps for the vaiable to not get reinitialised
        



if __name__ == "__main__":
    main()