import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

st.title("Reading Documents Using RAG")

load_dotenv()

# Load the documents
filepath="/Users/la-venkata/Documents/Learning/Python/RAG/"
filepaths=[os.path.join(filepath,filename) for filename in os.listdir(filepath) if filename.endswith(".docx")]
all_docs=[]

for filename in filepaths:
    print(filename)
    loader=UnstructuredWordDocumentLoader(filename)
    docs=loader.load()
    all_docs.extend(docs)

# Split the documents for performance
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500)
all_docs_splitted=text_splitter.split_documents(all_docs)

# store the documents into the Vector Store
vector_store=Chroma.from_documents(documents=all_docs_splitted,embedding=OpenAIEmbeddings(),persist_directory=filepath)

vector_store=Chroma(persist_directory=filepath,embedding_function=OpenAIEmbeddings())

retriever=vector_store.as_retriever(search_type="similarity")

#Define the prompt templates
system_prompt=("You are a helpful assistant for question answering tasks"
               "Use the following context to answer the questions"
                "In the given context Developer Resource(team member) refers to role Developer"
               "In the given context, please note that 'Business System Analyst' or 'Functional Analyst' or 'Analyst' refer to the role Business System Analyst or BSA"
               "Please provide detail explaination whereever possible"
               "Say I dont know if you dont know the answer"
               "\n\n"
               "{context}"
)

prompt=ChatPromptTemplate([
    ("system",system_prompt),
    ("human","{input}")
])

#Define the LLM
llm=ChatGroq(temperature=0,model="llama-3.3-70b-versatile")

#Create the chain
chain=({"context":retriever,"input":RunnablePassthrough()}|prompt|llm|StrOutputParser())

user_input=st.chat_input("Say Something:")

if user_input:
     res=chain.invoke(input=user_input)
     st.write(res)
