import chromadb
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.chroma import ChromaVectorStore

def definition_base():
    base = chromadb.PersistentClient(path="./chroma_db")
    collection = base.get_or_create_collection("cours_prepa")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    return vector_store

def chunk_documents():
    #On choisis l'outil pour extraire le texte des fichiers PDF
    extracteur = {".pdf": PyMuPDFReader()}
    #On charge les documents depuis le dossier "data" en utilisant l'extracteur
    documents = SimpleDirectoryReader("data", file_extractor=extracteur).load_data()
    for doc in documents:
        print("\n----------------------------")
        print(doc.metadata)
        print("----------------------------\n")

    print(f"Nombres de documents chargés: {len(documents)}")

    #On découpe les documents en chunks de 400 tokens avec un chevauchement de 100 tokens
    splitter = SentenceSplitter(chunk_size=400, chunk_overlap=100)
    chunk = splitter.get_nodes_from_documents(documents)
    return chunk
    
def traitement():
    chunk = chunk_documents()
    print(f"Nombres de chunks créés: {len(chunk)}")
    print("Test premier chunk")
    print(chunk[0].text)

    #On charge le modèle d'embedding depuis HuggingFace pour transformer les chunks en vecteurs
    modele_embedding = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

    #On crée une base de données ChromaDB pour stocker les vecteurs
    vector_store = definition_base()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    Settings.embed_model = modele_embedding

    #On transforme les chunks en vecteurs et on les stocke dans une base de données ChromaDB
    index = VectorStoreIndex(nodes=chunk,storage_context=storage_context)
    print("Vecteurs créés et stockés dans la base de données ChromaDB")
