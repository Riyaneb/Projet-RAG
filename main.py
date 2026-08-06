import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Modèle RAG pour la recherche de cours de prépa PDF")
    parser.add_argument("action", choices=["traitement","app"], help="Action à effectuer: 'traitement' pour indexer les documents, 'app' pour lancer l'application")
    args = parser.parse_args()
    if args.action == "traitement":
        from traitement import traitement
        traitement()
    elif args.action == "app":
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])


if __name__ == "__main__":
    main()