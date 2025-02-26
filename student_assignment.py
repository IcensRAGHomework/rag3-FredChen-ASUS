from datetime import datetime
import chromadb
import traceback

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = "text-embedding-ada-002"
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"


def generate_hw01():
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=gpt_emb_config["api_key"],
        api_base=gpt_emb_config["api_base"],
        api_type=gpt_emb_config["openai_type"],
        api_version=gpt_emb_config["api_version"],
        deployment_id=gpt_emb_config["deployment_name"],
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL", metadata={"hnsw:space": "cosine"}, embedding_function=openai_ef
    )

    return collection


def generate_hw02(
    question: str,
    city: list,
    store_type: list,
    start_date: datetime,
    end_date: datetime,
):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=gpt_emb_config["api_key"],
        api_base=gpt_emb_config["api_base"],
        api_type=gpt_emb_config["openai_type"],
        api_version=gpt_emb_config["api_version"],
        deployment_id=gpt_emb_config["deployment_name"],
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL", metadata={"hnsw:space": "cosine"}, embedding_function=openai_ef
    )

    where_list = []
    query_texts = [question] if question else ["search"]
    if city:
        where_list.append({"city": {"$in": city}})
    if store_type:
        where_list.append({"type": {"$in": store_type}})

    if start_date:
        where_list.append({"date": {"$gte": start_date.timestamp()}})
    if end_date:
        where_list.append({"date": {"$lte": end_date.timestamp()}})

    query_results = collection.query(
        query_texts=query_texts,
        n_results=10,
        where={"$and": where_list},
        include=["metadatas"],
    )

    return [metadata.get("name") for metadata in query_results["metadatas"][0]]


def generate_hw03(question, store_name, new_store_name, city, store_type):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=gpt_emb_config["api_key"],
        api_base=gpt_emb_config["api_base"],
        api_type=gpt_emb_config["openai_type"],
        api_version=gpt_emb_config["api_version"],
        deployment_id=gpt_emb_config["deployment_name"],
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL", metadata={"hnsw:space": "cosine"}, embedding_function=openai_ef
    )

    # get store id using store_name
    to_update_stores = collection.get(where={"name": store_name}, include=["metadatas"])
    for metadata in to_update_stores["metadatas"]:
        metadata["new_store_name"] = new_store_name

    # add new_store_name to the store's metadata
    collection.update(
        ids=to_update_stores["ids"],
        metadatas=to_update_stores["metadatas"],
    )

    # query the store using the question
    where_list = []
    query_texts = [question] if question else ["search"]
    if city:
        where_list.append({"city": {"$in": city}})
    if store_type:
        where_list.append({"type": {"$in": store_type}})

    query_results = collection.query(
        query_texts=query_texts,
        n_results=10,
        where={"$and": where_list},
        include=["metadatas"],
    )

    return [
        metadata.get("new_store_name") or metadata.get("name")
        for metadata in query_results["metadatas"][0]
    ]


def demo(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=gpt_emb_config["api_key"],
        api_base=gpt_emb_config["api_base"],
        api_type=gpt_emb_config["openai_type"],
        api_version=gpt_emb_config["api_version"],
        deployment_id=gpt_emb_config["deployment_name"],
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL", metadata={"hnsw:space": "cosine"}, embedding_function=openai_ef
    )

    return collection
