from pydantic import BaseModel, Field
from typing import Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


class SearchFilter(BaseModel):
    """Search over a database of customer interaction meeting data."""

    query: str = Field(
        ..., description="Similarity search query applied to meeting notes search."
    )
    from_date: Optional[datetime] = Field(None, description="The start date")
    to_date: Optional[datetime] = Field(None, description="The end date")


llm = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", temperature=0)
# .with_structured_output(SearchFilter)

prompt = PromptTemplate.from_template(
    """
        Analyze the given input and extract the dates from the input. 
        For relative data calculations like next 3 quarters, past 3 months etc, always calculate from Today: {today}.
        rewrite the query for better semantic search results.
        Query: {input}
    """
)
prompt = prompt.partial(today=datetime.today().strftime("%Y-%m-%d"))
print(datetime.today().strftime("%Y-%m-%d"))
chain = prompt | llm


system = """You are an expert at converting user questions into database queries. 
You have access to a database of customer meeting interaction notes for building LLM-powered applications. 
Given a question, return a list of database queries optimized to retrieve the most relevant results.
For relative data calculations like next 3 quarters, past 3 months etc, always calculate from Today: {today}.

If there are acronyms or words you are not familiar with, do not try to rephrase them."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
prompt = prompt.partial(today=datetime.today().strftime("%Y-%m-%d"))
structured_llm = llm.with_structured_output(SearchFilter)
query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm


resp = query_analyzer.invoke("what are the actionable items for the LG")

print(resp)