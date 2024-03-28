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

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
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
        # initialize 
        self.llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2")

    def get_retriever(self, loader):
        # get documents from loader
        documents = loader.load()

        # split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        # create retriever
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(texts, embeddings)
        retriever = vector_store.as_retriever()
        return retriever

    def get_response_principles(self, user_response, chat_history):
        """"Get chatbot response using constitutional principles.
        Args: 
            user_response (str)
            chat_history (list)
        """
        Mendez = ConstitutionalPrinciple(
            name="Mendez Principles",
            critique_request="Follow the Mendez Principles for investigative interviewing.",
            revision_request="Rewrite the model's response to follow the Mendez Principles",
        )   

        # template
        template = """" You are an interviewer for crime investigations. 
        A witness has provided a statement {statement} about a crime they witnessed.
        Based on this statement and the whole conversation {conversation}, you ask questions to gather more information.
        Ask a single open-ended question.
        """
        # create prompt
        prompt = PromptTemplate(
            template=template,
            input_variables=["statement", "conversation"],
        )

        # create chain
        chain = LLMChain(llm=self.llm, prompt=prompt)

        constitutional_chain = ConstitutionalChain.from_llm(
            llm=self.llm,
            chain=chain,
            constitutional_principles=[Mendez],
            verbose=False,
            return_intermediate_steps=False,
        )
        response = constitutional_chain.invoke({"statement": user_response, "conversation": get_whole_conv(chat_history)})
        return response
    
    def get_response(self, user_response, chat_history):
        """"Get chatbot response"""

        # template
        template = """" You are interviewing a witness to a crime.
        The structure of the interview is as follows:
        1. Ask open-ended, non-leading question until the witness provides sufficient detail.
        2. Ask more specific questions to clarify details and fill in gaps.

        The witness has provided a statement {statement}; the whole conversation is {conversation}.
        Base your question on the statement and the whole conversation.
        Ask one question:
        """
        # create prompt
        prompt = PromptTemplate(
            template=template,
            input_variables=["statement", "conversation"],
        )

        # create chain
        chain = prompt | self.llm | StrOutputParser()

        response = chain.invoke({"statement": user_response, 
                                "conversation": get_whole_conv(chat_history)
                                })        
        return response
    
def get_whole_conv(chat_history):

    """Get the whole conversation as a single string."""

    concatenated_messages = ''
    for message in chat_history:
        concatenated_messages += message['content'] + ' '
    return concatenated_messages


def main():
    load_dotenv()
    # create interviewer object
    interviewer = Interviewer()
    # set page config
    st.set_page_config(page_title="Interview") 
    st.header("Report a crime here.")

    # create session state
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "interview_agent", "content": """Hello, I am an AI agent that will conduct the interview with you. 
             In this interview you will make a statement regarding a crime you have witnessed and I will ask you questions 
             about it. Please provide as much detail as possible. I will ask you questions and you will respond.
             """}]
        interviewer.chat_history = st.session_state.messages

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
        response = interviewer.get_response(prompt, st.session_state.messages) # if constitutional principles -> get_response_principles()
        st.session_state.messages.append({"role": "interview_agent", "content": response})
        with st.chat_message("interview_agent"):
            st.write(response)


 
if __name__ == "__main__":
    main()

################################### TEMPLATES + RESULTS ###################################

# one open-ended question | no Mendez
# NOTE: returned multiple, detailed/leading  questions
template = """" You are an interviewer for crime investigations. 
A witness has provided a statement {statement} about a crime they witnessed.
Based on this statement and the whole conversation {conversation}, you ask questions to gather more information.
Ask a single open-ended question.
"""

# one question | Mendez
# NOTE: returned  one detailed/leading question + much context
template = """" You are an interviewer for crime investigations. 
A witness has provided a statement {statement} about a crime they witnessed.
The whole conversation is {conversation}.

Based on this statement and the whole conversation, ask a follow up question that follows the Mendez Principles.
Ask one question.
"""

# Mendez+PEACE
# NOTE: returned multiple detailed/leading  questions, context and a user response
template = """" You are interviewing a witness to a crime.
The interview should follow the PEACE model and Mendez principles for investigative interviewing.
The witness has provided a statement {statement}; the whole conversation is {conversation}.

Ask a follow-up question that follows the PEACE model and Mendez principles.
"""

# Mendez + simple instructions  
# NOTE: multiple detailed/leading  questions
template = """" You are interviewing a witness to a crime.
The interview should follow the Mendez principles for investigative interviewing:
- First ask open-ended, non-suggestive, non-leading questions.
- Later on, ask more specific questions to clarify details.

The witness has provided a statement {statement}; the whole conversation is {conversation}.
Ask a follow-up question as instructed.
"""

# Mendez + simple instructions | ONE question ---> the only change is a question-> ONE question.
# NOTE:  Returned two detailed questions and a user response
template = """" You are interviewing a witness to a crime.
The interview should follow the Mendez principles for investigative interviewing:
- First ask open-ended, non-suggestive, non-leading questions.
- Later on, ask more specific questions to clarify details.

The witness has provided a statement {statement}; the whole conversation is {conversation}.
Ask ONE follow-up question as instructed.
"""

# interview structure instructions | one question
# NOTE:  Returned the same detailed question twice, + context.
# NOTE: according to model the question "Can you describe the man who was firing the gun, such as height, hair color, clothing, and any distinctive features?" is open-ended and non-leading.
template = """" You are interviewing a witness to a crime.
The structure of the interview is as follows:
1. Ask open-ended, non-leading question until the witness provides sufficient detail.
2. Ask more specific questions to clarify details and fill in gaps.

The witness has provided a statement {statement}; the whole conversation is {conversation}.
Base your question on the statement and the whole conversation.
Ask one question:
"""

