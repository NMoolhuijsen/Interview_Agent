import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI, ChatHuggingFace
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplate import bot_template, user_template, css
from langchain_community.llms import HuggingFaceEndpoint

def get_pdf_text(pdf_docs):
    text = ""	# empty string
    for pdf in pdf_docs:	# iterate through pdfs
        pdf_reader = PdfReader(pdf)	# read pdf
        for page in pdf_reader.pages:	# iterate through pages
            text += page.extract_text()	# extract text from page
    
    return text	# return raw text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(	# create text splitter
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    )
    chunks = text_splitter.split_text(text)	# split text into chunks
    return chunks	# return text chunks

def get_vectorstore(text_chunks):
    """
    Create vector store from text chunks. Can be done via OpenAI or HuggingFaceInstructEmbeddings
    This variant uses model Instructor from HuggingFace.
    """	
    # embeddings = OpenAIEmbeddings()	# create embeddings in OpenAI
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")	# create embeddings in HuggingFace
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)    # create vector store
    return vectorstore	# return vector store

def get_conversation_chain(vectorstore):
    # llm = ChatHuggingFace()	# choose llm
    llm = HuggingFaceEndpoint(repo_id="google/gemma-7b") # choose llm from huggingface
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)	# create conversation chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain	# return conversation chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})	# get response
    st.session_state.chat_history = response['chat_history']	# save chat history
    # print(st.session_state.chat_history)

    for i, message in enumerate(st.session_state.chat_history):	# iterate through chat history
        # show user message
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)	
        # show bot message
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)	


def main():
    load_dotenv() # load environment variables
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:") # user interface
    st.write(css, unsafe_allow_html=True) # write css

    if "conversation" not in st.session_state: # if conversation is not in session state	
        st.session_state.conversation = None # set conversation to None. Else, use saved conversation
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:") # label for user input, hardcoded conv
    if user_question: # if user question is not empty
        handle_userinput(user_question) # handle user input

    st.write(user_template.replace("{{MSG}}", "hello bot"), unsafe_allow_html=True) # replace user template with hello bot
    st.write(bot_template.replace("{{MSG}}", "hello human"), unsafe_allow_html=True) 

    with st.sidebar:
        st.subheader("Upload PDFs")
        pdf_docs = st.file_uploader("Upload a PDF file", type="pdf", accept_multiple_files=True) # file uploader
        if st.button( "Upload"): # button to upload file
            st.spinner("Process") #
            with st.spinner("Processing"):	 # show processing message while processing
                # get pdf text
                raw_text = get_pdf_text(pdf_docs) # single string of text
                # st.write(raw_text) # display text
                # get text chunks
                text_chunks = get_text_chunks(raw_text) # list of text chunks
                # st.write(text_chunks)
                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                # use to keep the variable from being reset/reinitialized
                st.session_state.conversation = get_conversation_chain(vectorstore) 
                # conversation = get_conversation_chain(vectorstore) 


if __name__ == '__main__':
    main()
