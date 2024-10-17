from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

from datetime import datetime


def convert_date_to_epoch(str_date):
    dt = datetime.strptime(str_date, "%d-%m-%Y")
    return dt.timestamp().strftime("%s")


def ingestion(embeddings_model):
    meeting_nodes = [
        (
            "New Year Party",
            "Meeting happened 2023-Jan-01 for parting new year",
            "2023-01-01",
        ),
        (
            "Abc Project",
            "Meeting happened 2023-Feb-20 on Project Abc, all are present",
            "2022-09-11",
        ),
        (
            "Customer meeting",
            "Meeting happened 2023-Jun-20 Where IFX participants Peter with customer contact Jang",
            "2024-04-12",
        ),
    ]

    from langchain.schema import Document

    docs = []
    from datetime import datetime

    for meeting in meeting_nodes:
        dt = datetime.strptime(meeting[2], "%Y-%m-%d")
        doc = Document(
            page_content=f"Meeting Title:{meeting[0]}\nMeeting Notes:{meeting[0]}",
            metadata={
                "year": dt.year,
                "month": dt.month,
                "day": dt.day,
                "meeting_date": dt.strftime("%s"),
                "meeting_title": meeting[0],
            },
        )
        docs.append(doc)

    chrome_vs = Chroma.from_documents(
        documents=docs,
        embedding=embeddings_model,
        persist_directory="vectordbs",
        collection_name="meetings_tmp",
    )


# ingestion(embeddings_model)


def query(qry):
    chroma_vs = Chroma(
        persist_directory="vectordbs",
        collection_name="meetings_tmp",
        embedding_function=embeddings_model,
    )
    # retriever = chroma_vs.as_retriever()

    metadata_field_info = [
        AttributeInfo(
            name="meeting_title",
            description="title of the meeting",
            type="string",
        ),
        AttributeInfo(
            name="year",
            description="the year the meeting happened",
            type="integer",
        ),
        AttributeInfo(
            name="month",
            description="the month the meting happened",
            type="integer",
        ),
        AttributeInfo(
            name="day",
            description="the day the meeting happened",
            type="integer",
        ),
        AttributeInfo(
            name="meeting_date",
            description="the date the meeting happened",
            type="integer",
        ),
    ]
    document_content_description = "customer Meeting and their notes"

    llm = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview")

    # prompt = PromptTemplate.from_template(
    #     """
    #     Analyze the given user query for any datetime related data, and expand it with year, month and date.
    #     Query: {input}
    #     """
    # )

    # chain = prompt | llm | StrOutputParser()
    # print(chain.invoke(qry))

    retriever = SelfQueryRetriever.from_llm(
        llm,
        chroma_vs,
        document_content_description,
        metadata_field_info,
        verbose=True,
    )

    rag_prompt = PromptTemplate.from_template(
        """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        Question: {question} 
        Context: {context} 
        Answer:
        """
    )
    from operator import itemgetter
    from langchain_core.runnables import RunnableLambda

    def format_docs(dict):
        print(dict["docs"])
        print(dict["que"])
        return "\n\n".join(doc.page_content for doc in dict["docs"])

    rag_chain = (
        {
            "context": {"docs": retriever, "que": lambda x: x}
            | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    resp = rag_chain.invoke(qry)
    print(resp)


#
query("give me meeting with title new year")
query("What are meetings happened between 01-04-2022 to 03-03-2024")
