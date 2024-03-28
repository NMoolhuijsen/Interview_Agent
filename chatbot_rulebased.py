from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
import os
from operator import itemgetter
from typing import List

from langchain_community.vectorstores import VectorStore, FAISS, Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, PyPDFDirectoryLoader
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import GPT4AllEmbeddings, HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langchain_core.output_parsers import ListOutputParser, CommaSeparatedListOutputParser, JsonOutputParser, StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains import QAGenerationChain, ConversationalRetrievalChain, LLMChain, ConstitutionalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel

class Interviewer:
    def __init__(self):
        self.llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2")

    def get_retriever(self, loader):
        # get documents from data folder
        documents = loader.load()

        # split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        # create retriever
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(texts, embeddings)
        retriever = vector_store.as_retriever()
        return retriever
    
    
    def get_response(self, messages, open_ended):
        user_messages = [HumanMessage(content=message["content"]) for message in messages if message["role"] == "user"]
        ai_messages = [AIMessage(content=message["content"]) for message in messages if message["role"] == "interview_agent"]

        if open_ended:
            examples_open = [
                {"input": "I saw a woman hit her child", "output": "What else can you tell me about it"},
                {"input": "A man robbed a woman outside of the supermarket", "output": "Can you walk me through what happened?"},
                {"input": "A child was aggresively playing with a dog and then kicked it", "output": "Let's go through the events step by step. Begin with the first thing you remember."},
                {"input": "I heard what sounded like a gunshot. From my window i saw people running away.", "output": "Please explain what you remember about that moment."},
                {"input": "Soldiers came into our house and started beating my husband. Then they arrested him, and now he is gone.", 
                 "output": "Start from the beginning and tell me everything in your own words."},
            ]
            example_prompt = ChatPromptTemplate.from_messages(
                [
                    ("human", "{input}"),
                    ("ai", "{output}"),
                ]
            )
            few_shot_prompt = FewShotChatMessagePromptTemplate(
                example_prompt=example_prompt,
                examples=examples_open,
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a interviewer for crime investigations. You ask open-ended questions."),
                    few_shot_prompt,
                    ("human", {input})
                ]
            )
            chain = prompt | self.llm | StrOutputParser()

        response = chain.invoke(user_messages[-1].content)

  
        return response

def main():
    load_dotenv()
    interviewer = Interviewer()
    st.set_page_config(page_title="Interview") 
    st.header("Report a crime here.")

    # create session state
    if "messages" not in st.session_state.keys():
        print('NEW SESSION')
        st.session_state.messages = [
            {"role": "interview_agent", "content": """Hello, I am an AI agent that will conduct the interview with you. 
             In this interview you will make a statement regarding a crime you have witnessed and I will ask you questions 
             about it. Please provide as much detail as possible. I will ask you questions and you will respond.
             """}]
        st.session_state.open_ended = True

    # display messages all previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # user reponse
    if prompt := st.chat_input("What did you experience?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # interview agent response
    if st.session_state.messages[-1]["role"] == "user":
        response = interviewer.get_response(st.session_state.messages, st.session_state.open_ended)
        st.session_state.messages.append({"role": "interview_agent", "content": response})
        with st.chat_message("interview_agent"):
            st.write(response)
    


 
if __name__ == "__main__":
    main()