from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from typing import Optional
from datetime import date


class QueryFilters(BaseModel):
    from_date: Optional[date] = Field(..., description="Start date of the meeting")
    to_date: Optional[date] = Field(..., description="End date of the meeting")


prompt_tmpl = """
Analyze the following query with following rules and extract the data.

Rule:
If query has any date duration, then expand it to from_date and to_date, otherwise leave it

Output in JSON format with keys: from_date, to_date
Only output JSON, no other text.


Current date is: {today}

The query is: {query}
JSON:
"""
prompt = PromptTemplate.from_template(prompt_tmpl)
prompt = prompt.partial(today=date.today().strftime("%Y-%m-%d"))
llm = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview")

chain = prompt | llm | JsonOutputParser(pydantic_object=QueryFilters)


questions = [
    "What are the critical issues for past 3 months",
    "what are the issues not resolved for more than one month",
    "what are tech support issues",
]

for question in questions:
    resp = chain.invoke({"query": question})
    print(resp)
