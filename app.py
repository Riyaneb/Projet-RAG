import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
from traitement import definition_base
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def app():
    load_dotenv()
    Settings.llm = Gemini(model="models/gemini-3.6-flash",api_key=os.getenv("GOOGLE_API_KEY"))
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

    vector_store = definition_base()
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    query_engine = index.as_query_engine(similarity_top_k=4,response_mode="refine")
    question = input("Posez votre question : ")
    reponse = query_engine.query(question)
    print("\n----------------------------\n")
    print(f"Chunk utilisés : {reponse.source_nodes}")
    print("\n----------------------------\n")
    print(f"Réponse : {reponse.response}")
