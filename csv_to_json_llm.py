from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.title("CSV to JSON Converter")
# Define the prompt
prompt_message="""You are an expert reader in reading CSV files. Given the below.Produce only the output and no other supplemental information.
           Below are the few examples on how the information appears and how it should be extracted in json format
           
           Example1:
           "input_text": ["012636633,Test Name1,Computer Science,3.7,Advisor Name1,Test Contract","GM100,038136,2023-01-08,ZZ10,340000,Test Contract2"]

           "input_text": ["GM100,038135,2023-01-07,YY10,30000,Test Contract","GM100,038136,2023-01-08,ZZ10,340000,Test Contract2"]
           "output_text":"output={{
            {{"BU":"GM100",'Award':'038135','Start_Date':'2023-01-07','Amount':30000,'Description':'Test Contract'}},
            {{"BU":"GM100",'Award':'038136','Start_Date':'2023-01-08','Amount':340000,'Description':'Test Contract2'}}
           }}"

           Example2: 
           "input_text": ["GM100,038135,,YY10,40000,Test Contract1",]
           "output_text": "output={{
            {{"BU":"GM100",'Award':'038134','Start_Date':NULL,'Amount':40000,'Description':'Test Contract1'}}
            }}"

           Now generate for
           {input_text}\n
           "output_text":
         
           """

student_prompt_message="""You are an expert reader in reading CSV files. Given the below.Produce only the output and no other supplemental information.
           Below are the few examples on how the information appears and how it should be extracted in JSON format
           
           Example1:
           "input_text": ["012636633,Test Name1,Computer Science,3.7,Advisor Name1","733939,Test Name2,Bachelors in History,3.2,Advisor Name2"]
           "output_text":"output={{
            {{'StudentId':'012636633','Name':'Test Name1','Major':'Computer Science','GPA':'3.7','Advisor':'Advisor Name1'}},
            {{'StudentId':"733939",'Name':'Test Name2','Major':'Bachelors in History','GPA':'3.2','Advisor':'Advisor Name2'}}
           }}"

           Example2: 
           "input_text": ["123232,Test Name7,Bachelor in Economics,3.4,Advisor Name10"]
             "output_text":"output={{
            {{'StudentId':"123232",'Name':'Test Name7','Major':'Bachelor in Economics,'GPA':'3.4','Advisor':'Advisor Name10'}}
           }}"

           Now generate for
           {input_text}\n
           "output_text":
         
           """
prompt_template=PromptTemplate(input_variables=["input_text"],template=student_prompt_message)
llm=ChatGroq(temperature=0,model="llama-3.3-70b-versatile")

chain=prompt_template|llm|StrOutputParser()

# res=chain.invoke({'input_text':["GM100,0987773,,XX10,50000,Test Example8866","GM100,0987776,2023-09-07,ZZ10,70000,Test Ex"]})

user_input=st.chat_input("Input the CSV text to be converted to JSON")
if user_input:
    res=chain.invoke({'input_text':user_input})
    st.write(res)