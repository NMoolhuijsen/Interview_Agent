from dotenv import load_dotenv
import streamlit as st
import os

from langchain_community.vectorstores import VectorStore, FAISS, Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, PyPDFDirectoryLoader
from langchain_community.llms import HuggingFaceEndpoint

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains import QAGenerationChain, ConversationalRetrievalChain, LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser



# llm = HuggingFaceEndpoint(repo_id='mistralai/Mistral-7B-Instruct-v0.2')

# template
template = """
You are an interviewer for crime investigations. You are interviewing a witness to a crime regarding what they 
have witnessed. Which questions and the way you will ask these will be according to the following principles:
{context}

First you will give a brief overview of how the interview will go. Then you will ask the witness for their personal information:
1. What is your full name, date of birth, and place of birth?
2. What is your current address and city of residence?
3. What is your pasport number / ID number?

Only continue to the next question when a full answer in given to the previous question.

Before starting the interview, give the witness the instructions as specifified in {context}. Afterwards,
ask the witness to describe the events they witnessed:
1. What did you see and hear?
2. Where and when was this?
3. Ask the witness to describe the people involved.

For each question, the witness will respond with a response {response}. Ask the witness to elaborate if the response is not detailed enough.
At every stage, check for inconsistencies in the witness's story and ask for clarification if needed.

"""

# # prompt
# prompt = ChatPromptTemplate.from_template(template)
# chain = (
#     {"context": retriever, "response": RunnablePassthrough()} # "chat_history": chat_history,
#     | prompt
#     | llm 
#     | StrOutputParser() 
# )
class Interviewer:
    def __init__(self):
        self.retriever = None

    def set_retriever(self):
        # get documents from data folder
        loader = PyPDFDirectoryLoader('data')
        documents = loader.load()

        # split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        # create retriever
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vector_store = FAISS.from_documents(texts, embeddings)
        retriever = vector_store.as_retriever()
        self.retriever = retriever

    def get_conversation_chain(self):
        llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2") # choose llm from huggingface
        prompt_template = ChatPromptTemplate.from_template(template) # create prompt template
        # create conversation chain with memory
        # memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)	
        conversation_chain = create_history_aware_retriever(
            llm=llm,
            retriever=self.retriever,
            prompt=prompt_template
        )
        return conversation_chain
    

def main():
     # load environment variables
    load_dotenv()
    interviewer = Interviewer()
    interviewer.set_retriever()

    # set page config
    st.set_page_config(page_title="Interview") 
    st.header("Chat with an interview agent.")

    st.write(''' 
             Hello, I am an AI agent that will conduct the interview with you. In this interview you will make a statement
             regarding a crime you have witnessed and I will ask you questions about it. Do you want to proceed?
              ''' )
    
    if st.button( "Yes"):
        st.session_state.conversation = interviewer.get_conversation_chain()
        st.session_state.chat_history = []

        user_response = st.chat_message(st.session_state.conversation) # label for user input, hardcoded conv
        if user_response: # if user question is not empty
            handle_userinput(user_response, st.session_state.chat_history) # handle user input


def handle_userinput(user_response, chat_history):
    response = st.session_state.conversation.invoke({'response': user_response, 'chat_history': chat_history})	# get response
    st.session_state.chat_history = response['chat_history']	# save chat history
    st.chat_message(response)	# display response


if __name__ == "__main__":
    main()