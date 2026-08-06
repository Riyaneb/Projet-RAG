import time

from llama_index.core import Settings

from app import chargement
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

eval_questions = [
    {
        "question": "Qu'est-ce qu'une valeur propre d'un endomorphisme f ?",
        "reponse_attendue": "C'est un scalaire lambda s'il existe un vecteur non nul v tel que f(v) = lambda * v.",
    },
    {
        "question": "À quelle condition nécessaire et suffisante une matrice est-elle diagonalisable d'après son polynôme caractéristique ?",
        "reponse_attendue": "Une matrice est diagonalisable si et seulement si son polynôme caractéristique est scindé et que pour chaque valeur propre, la dimension du sous-espace propre est égale à sa multiplicité.",
    },
    {
        "question": "Quelle est la condition de convergence pour une intégrale de Riemann de la forme int(1 à +inf) dt / t^alpha ?",
        "reponse_attendue": "L'intégrale converge si et seulement si alpha est strictement supérieur à 1.",
    },
    {
        "question": "Que dit le théorème de König-Huygens pour le calcul de la variance d'une variable aléatoire X ?",
        "reponse_attendue": "La variance est égale à l'espérance du carré moins le carré de l'espérance : V(X) = E[X^2] - (E[X])^2.",
    },
]

def evaluation():
    print("Chargement du modèle\n")
    query_engine = chargement()
    print("Evaluation des questions :\n")

    faithfulness_evaluator = FaithfulnessEvaluator(llm=Settings.llm)
    relevancy_evaluator = RelevancyEvaluator(llm=Settings.llm)
 
    for element in eval_questions:
        question = element["question"]
        reponse = query_engine.query(question)
        eval_fidelite = faithfulness_evaluator.evaluate_response(response=reponse)
        eval_pertinence = relevancy_evaluator.evaluate_response(query=question,response=reponse)
        print(f"Question: {element['question']}, Fidélité: {eval_fidelite.passing}, Pertinence: {eval_pertinence.passing}")
        time.sleep(50)


if __name__ == "__main__":
    evaluation()