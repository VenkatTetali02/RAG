from dotenv import load_dotenv
import streamlit as st
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader,UnstructuredPDFLoader, SeleniumURLLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
load_dotenv()

urls=["https://www.cnn.com/"]
#testurls=["https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-february-8-2023"]

#URLLoader=SeleniumURLLoader(urls)
URLLoader=UnstructuredURLLoader(urls)
data=URLLoader.load()
#print(data)

splitter=RecursiveCharacterTextSplitter(chunk_size=500)
docs=splitter.split_documents(data)
print(docs)

vector_store=Chroma.from_documents(documents=docs,embedding=OpenAIEmbeddings())

retriever=vector_store.as_retriever(search_type='similarity')

systemPrompt=("You are a helpful assistant for question answering tasks"
              "Use the following retrieved context to answer the question"
              "If you dont know say dont know"
              "\n\n"
              "{context}"
)
prompt=ChatPromptTemplate([("system",systemPrompt),("human","{input}")])

llm=ChatOpenAI(temperature=0.1)

chain=({"context": retriever, "input": RunnablePassthrough()}|prompt|llm|StrOutputParser())

res=chain.invoke(input="Summarize the news for me")

print(res)