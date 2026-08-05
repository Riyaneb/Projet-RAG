from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PyMuPDFReader

extracteur = {"pdf": PyMuPDFReader()}
documents = SimpleDirectoryReader("data", file_extractor=extracteur).load_data()
for doc in documents:
    print("\n----------------------------")
    print(doc.metadata)
    print("----------------------------\n")

print(f"Nombres de documents chargés: {len(documents)}")

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)
chunk = splitter.get_nodes_from_documents(documents)

print(f"Nombres de chunks créés: {len(chunk)}")
print("Test premier chunk")
print(chunk[0].text)