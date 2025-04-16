from langchain_groq import ChatGroq
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.tools import Tool,StructuredTool
from langchain.agents import AgentType,initialize_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
import uuid
from pydantic import BaseModel
import streamlit as st
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv

load_dotenv()
llm=ChatGroq(model="llama-3.3-70b-versatile")
res=llm.invoke('Give me exactly one funny name for AI Agent')
print(res)
st.title(f'Hi..I am AI Agent {res.content}')


db=SQLDatabase.from_uri("sqlite:///bank.db")

class FlightModel(BaseModel):
    source_city:str
    dest_city:str
    num_of_passengers:int

class ConfirmationModel(BaseModel):
    confirmation: str

class Student(BaseModel):
     id: str

def book_flight_ticket(source_city,dest_city,num_of_passengers): 
    print(source_city,dest_city)
    # print(query["source"])
    # lst=query.sp
    # print(f"src is {src}")
    # print(f'dest  is {dest}')
    conf=str(uuid.uuid4())
    return f'Flight ticket booked with confirmation# {conf} from {source_city} to {dest_city} for {num_of_passengers} passengers'


def create_student(id,name):
    conf=str(uuid.uuid4())
    return f'Created Record1, Record2, Record 3\n\nCreated Student with Id {id} and name {name} and confirmation id is {conf}'

def send_email(confirmation):
    return f'Email sent for the confirmation# {confirmation}'

ft_tool=StructuredTool(name="Flight Ticket Booking Tool",func=book_flight_ticket,args_schema=FlightModel,description='Use this tool only to book a flight ticket from Source to destination using API')
email_tool=StructuredTool(name="Email Tool",func=send_email,args_schema=ConfirmationModel,description='Use this tool to send email related to the confirmation id received')
student_creation_tol=StructuredTool(name='Student Creation Tool',func=create_student,args_schema=Student,description="Use this tool to create Students in the database")

tools=SQLDatabaseToolkit(db=db,llm=llm).get_tools()

tools.append(ft_tool)
tools.append(email_tool)

for tool in tools:
    print(tool.name)

agent_prompt="""
            1.Use Flight Booking tool only for Ticket booking queries
            2.Use the Email tool for sending the confirmation emails. Confirmation ID is provided as Input
            3. Use the SQL Tools for performing database queries
            """
# AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION

# agent=initialize_agent(llm=llm,tools=tools,agent='zero-shot-react-description',verbose=True,agent_prompt=agent_prompt)
agent=initialize_agent(llm=llm,tools=tools,agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,verbose=True,agent_prompt=agent_prompt)

user_input=st.chat_input("Enter a query")
if user_input:
    res=agent.run(user_input)
    st.write(res)