from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Literal


class QryType(BaseModel):
    qtype: Literal["VS", "SA"] = Field(
        description="Type of the query. Can be either VS or SA"
    )


class SQLQuery(BaseModel):
    query: str = Field(description="SQL Query")


def classify_query(qry, llm):

    prompt_text = """
    You are given a text query. Your task is classify the query is better suited for a vector store (VS)  or a SQL agent (SA) using below rules:
    Instructions:
    - If query contains any of the below keywords then its a SQL agent
    - Look for all the table names
    - Look for all the column names
    - date related conditions like 3 months or 3 quarters data
    - can be derived from columns like meeting title, meeting datetime, meeting status etc
    - count related questions
    if none of the above rules are not met then its a Vector Store
    Use below table schema better decision:
    Table Name: meetings
    Columns:
        meeting_id INTEGER PRIMARY KEY AUTOINCREMENT,
        meeting_title TEXT NOT NULL,
        meeting_datetime TEXT NOT NULL,
        meeting_status TEXT NOT NULL,
        app_to_flag TEXT,
        opp_annualized_revenue REAL,
        budget_amount REAL,
        design_customer TEXT,
        end_customer TEXT,
        focus_project TEXT,
        lifetime_revenue REAL,
        opportunity_id TEXT,
        opportunity_stage TEXT,
        opportunity_type TEXT,
        project_name TEXT,
        traffic_light TEXT,
        canvas_app_test TEXT,
        meeting_notes_summary TEXT,
        modified_on TEXT,
        unit TEXT,
        created_on TEXT,
        source_appointment TEXT,
        state_code TEXT,
        highlights_lowlights TEXT

    Output the classification in JSON format with key "qtype" and value either "VS" or "SA". Dont include any explanation.
    Query: "{query}"
    Classification:
    """

    prompt = ChatPromptTemplate.from_template(prompt_text)

    chain = prompt | llm | JsonOutputParser(pydantic_object=QryType)

    resp = chain.invoke({"query": qry})
    return resp["qtype"]


def post_process_sql_qry(sql_qry, llm):

    prompt_text = """
    Your job to examine the given SQL query and regenerate if required based on below rules:
    Rules:
     - if query has status column, and value is 'In Progress' then replace it with 'In Progression'
     - If no change is required then return the same query as it is
     - If query has only one condition with company column name, regenerate for LIKE search
    
    output the query in markdown JSON format with key "query". Dont include any explanation.
    SQL Query: "{query}"
    New Query:
    """

    prompt = ChatPromptTemplate.from_template(prompt_text)

    chain = prompt | llm | JsonOutputParser(pydantic_object=SQLQuery)

    resp = chain.invoke({"query": sql_qry})
    return resp["query"]


llm = ChatGroq(temperature=0, model="mixtral-8x7b-32768")

sql_qryies = [
    "SELECT * FROM meetings WHERE status = 'In Progression' and company = 'GL'",
    "SELECT * FROM meetings WHERE company = 'GL'",
    "SELECT meeting_date_time WHERE status = 'In Progress'",
    "UPDATE meetings SET meeting_title is null WHERE status in ['In Progress','Completed']",
]

for qry in sql_qryies:
    print(post_process_sql_qry(qry, llm))
