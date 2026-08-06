import os
from dotenv import load_dotenv
from llama_index.core import PromptTemplate, Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from traitement import definition_base
import streamlit as st

st.set_page_config(page_title="Assistant pédagogique pour la révision des cours de prépa", page_icon=":books:", layout="centered")
st.title("Assistant pédagogique pour la révision des cours de prépa")


#Chargement des vecteurs et du modèle LLM
@st.cache_resource
def chargement():
    load_dotenv()
    Settings.llm = Gemini(model="models/gemini-3.5-flash",api_key=os.getenv("GOOGLE_API_KEY"))
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

    vector_store = definition_base()
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    prompt = PromptTemplate("Tu es un assistant pédagogique qui aide un étudiant à réviser ses cours. Voici des extraits de cours pertinents : \n{context_str}\n\nEn te basant sur ces extraits, réponds à la question suivante : {query_str}\n\nSi les extraits ne contiennent pas suffisamment d'informations pour répondre à la question, réponds honnêtement que tu ne sais pas.")

    query_engine = index.as_query_engine(similarity_top_k=4,response_mode="compact",text_qa_template=prompt)
    return query_engine

with st.spinner("Chargement des modèles et des vecteurs"):
    query_engine = chargement()

question = st.chat_input("Posez votre question : ")
if question:
    st.chat_message("user").write(question)

    #Chargement de la réponse
    with st.spinner("Recherche dans les documents..."):
        reponse = query_engine.query(question)
    #Affichage de la réponse et des extraits de cours utilisés pour générer la réponse
    with st.chat_message("assistant"):
        st.markdown(f"Réponse : {reponse.response}")
        with st.expander("Cours utilisés : "):
            for i, source in enumerate(reponse.source_nodes):
                st.markdown(f"**Extrait {i+1} :** {source.node.get_text()}")
                st.caption(f"Source : {source.node.metadata.get('file_path', 'Inconnue')}")
                st.divider()


