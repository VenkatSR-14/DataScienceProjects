from pinecone import Pinecone as PC, ServerlessSpec
import pandas as pd
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.chains import LLMMathChain
from langchain_community.vectorstores import Pinecone
from io import StringIO
from settings import SETTINGS

pc = PC(api_key=SETTINGS["PINECONE_API_KEY"])
index = pc.Index("llmproject")
embed = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=SETTINGS['OPENAI_API_KEY'])

text_field = 'text'
vectorStore = Pinecone(
    index, embed.embed_query, text_field
)

llm = ChatOpenAI(
    temperature=0.0
)

timekeeping_policy = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorStore.as_retriever()
)

df = pd.read_csv("data/employee_data_new.csv")
python = PythonAstREPLTool(locals={'df': df})
calculator = LLMMathChain.from_llm(llm=llm, verbose=True)
user = 'Venkat Srinivasa Raghavan'
df_columns = df.columns

tools = [
    Tool(
        name="Employee Data",
        func=python.run(),
        description=f"""Useful for when you need to answer questions about employee data stored in pandas dataframe 'df'. 
        Run python pandas operations on 'df' to help you get the right answer.
        'df' has the following columns: {df_columns}
        
        <user>: How many Sick Leave do I have left?
        <assistant>: df[df['name'] == '{user}']['sick_leave']
        <assistant>: You have n sick leaves left.              
        """
    ),
    Tool(
        name="Calculator",
        func=calculator.run,
        description=f"""
                Useful when you need to do math operations or arithmetic.

                <user>: How much will I be paid if I encash my unused VLs?
                <assistant>: df[df['name'] == '{user}'][['basic_pay_in_php', 'vacation_leave']]
                <assistant>: You will be paid Php n if you encash your unused VLs.'
                """
    )
]

agent_kwargs = {
    'prefix': f'You are friendly HR assistant. You are tasked to assist the current user: {user} on questions related to HR. You have access to the following tools:'}

agent = initialize_agent(tools,
                         llm,
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                         verbose = True,
                         agent_kwargs=agent_kwargs,
                         handle_parsing_errors = True
                         )

def get_response(user_input):
    response = agent.run(user_input)
    return response

# <user>: How much will I be paid if I encash my unused VLs?
# <assistant>: df[df['name'] == '{user}'][['basic_pay_in_php', 'vacation_leave']]
# <assistant>: You will be paid Php n if you encash your unused VLs.
