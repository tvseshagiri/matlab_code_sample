from langchain.schema import Document
import faker
from datetime import datetime
import time
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def sample_data_gen():
    fake = faker.Faker()
    docs_list = []
    for i in range(50):
        meeting_date = (
            fake.date_between_dates(
                date_start=datetime(2023, 1, 1), date_end=datetime.today()
            ),
        )
        meeting_notes = {
            "meeting_id": fake.uuid4(),
            "meeting_date": meeting_date[0],
            "meeting_participants": [fake.name() for _ in range(5)],
            "meeting_agenda": fake.sentence(),
            "meeting_notes": fake.text(),
        }
        metadata = {
            "meeting_date": int(time.mktime(meeting_date[0].timetuple())),
        }
        docs_list.append(Document(page_content=str(meeting_notes), metadata=metadata))
    print(docs_list)
    return docs_list


def store_in_chroma():

    docs_list = sample_data_gen()
    Chroma.from_documents(
        docs_list,
        collection_name="fake_meetings_col",
        persist_directory="vectordbs",
        embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
    )


def retrieve():
    chroma_vs = Chroma(
        persist_directory="vectordbs",
        collection_name="fake_meetings_col",
        embedding_function=GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"
        ),
    )

    from_date = int(datetime(2024, 7, 1).timestamp())
    to_date = int(datetime(2024, 7, 30).timestamp())

    filter = {
        "$and": [
            {"meeting_date": {"$gte": from_date}},
            {"meeting_date": {"$lte": to_date}},
        ]
    }

    retriever = chroma_vs.as_retriever()

    resp = chroma_vs.similarity_search_with_score(
        query="Success whatever behin",
        filter=filter,
        k=1,
    )
    print(resp)


retrieve()
