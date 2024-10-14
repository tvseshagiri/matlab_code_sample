import requests
from typing import List
from langchain_core.embeddings import Embeddings


class DKAPIEmbeddings(Embeddings):
    def __init__(self, model_name: str, api_url: str):
        self.model_name = model_name
        self.api_url = api_url

    def embed_documents(self, texts: List[str]) -> List[List[float]]:

        embedding_list = []
        print(f"Got {len(texts)} texts")
        for text in texts:
            embeddings = generate_embeddings(text)
            embedding_list.append(embeddings)

        print(f"Returning {len(embedding_list)} vectors")

        return embedding_list

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def generate_embeddings(text):
    pass
