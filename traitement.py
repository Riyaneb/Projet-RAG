from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PyMuPDFReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

#On choisis l'outil pour extraire le texte des fichiers PDF
extracteur = {"pdf": PyMuPDFReader()}

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

print(f"Nombres de chunks créés: {len(chunk)}")
print("Test premier chunk")
print(chunk[0].text)

#On charge le modèle d'embedding depuis HuggingFace pour transformer les chunks en vecteurs
modele_embedding = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

#On transforme les chunks en vecteurs et on les stocke dans une liste
liste_vecteur = []
for element in chunk:
    liste_vecteur.append(modele_embedding.get_text_embedding(element.text))

print(f"Nombre de vecteurs créés: {len(liste_vecteur)}")
print("Test premier vecteur")
print(liste_vecteur[0])
print("Test deuxième vecteur")
print(liste_vecteur[1])