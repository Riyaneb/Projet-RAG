import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Modèle RAG pour la recherche de cours de prépa PDF")
    parser.add_argument("action", choices=["traitement","app","evaluation"], help="Action à effectuer: 'traitement' pour indexer les documents, 'app' pour lancer l'application, 'evaluation' pour évaluer le modèle")
    args = parser.parse_args()
    if args.action == "traitement":
        from traitement import traitement
        traitement()
    elif args.action == "app":
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    elif args.action == "evaluation":
        from evalutation import evaluation
        evaluation()


if __name__ == "__main__":
    main()