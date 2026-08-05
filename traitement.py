from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PyMuPDFReader

#On choisis l'outil pour extraire le texte des fichiers PDF
extracteur = {"pdf": PyMuPDFReader()}

#On charge les documents depuis le dossier "data" en utilisant l'extracteur
documents = SimpleDirectoryReader("data", file_extractor=extracteur).load_data()
for doc in documents:
    print("\n----------------------------")
    print(doc.metadata)
    print("----------------------------\n")

print(f"Nombres de documents chargés: {len(documents)}")

#On découpe les documents en chunks de 1000 tokens avec un chevauchement de 200 tokens
splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)
chunk = splitter.get_nodes_from_documents(documents)

print(f"Nombres de chunks créés: {len(chunk)}")
print("Test premier chunk")
print(chunk[0].text)