from pathlib import Path
import mlflow
import numpy as np
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from time import sleep
import yaml

import models

from FlagEmbedding import FlagModel

with open("config.yml", encoding="utf-8") as _cfg_file:
    CONF = yaml.safe_load(_cfg_file)

FOLDER = Path("data") / "wiki"
FILENAMES = [
    FOLDER / title for title in [
        "Inception.md", "The Dark Knight.md", "Deadpool.md", "Fight Club.md", 
        "Pulp Fiction.md", "Titanic.md", "Avengers: Infinity War.md", "Seven Samurai.md"
    ]
]
DF = pd.read_csv("data/questions.csv", sep=";") 

# Modèle d'encodage sémantique pour mesurer la similarité
ENCODER = SentenceTransformer('all-MiniLM-L6-v2')

def _load_ml_flow(conf):
    """Initialise l'expérience MLflow pour la traçabilité des résultats."""
    mlflow.set_experiment("RAG_Movies_clean")

# Initialisation MLflow
_load_ml_flow(CONF)

def run_evaluate_retrieval(config, rag=None):
    """
    Lance l'évaluation du module de récupération (retrieval).
    Args:
        config: Configuration du modèle
        rag: Instance de modèle RAG (optionnel)
    Returns:
        Instance du modèle RAG après evaluation
    """
    rag = rag or models.get_model(config)
    score = evaluate_retrieval(rag, FILENAMES, DF.dropna())

    description = str(config["model"])
    _push_mlflow_result(score, config, description)
    
    return rag

def run_evaluate_reply(config, rag=None):
    """
    Lance l'évaluation de la génération de réponses du modèle RAG.
    Args:
        config: Configuration du modèle
        rag: Instance de modèle RAG (optionnel)
    Returns:
        Instance du modèle RAG après evaluation
    """
    rag = rag or models.get_model(config)
    # On prend un sous-échantillon des questions pour limiter le nombre de requêtes coûteuses
    indexes = range(2, len(DF), 10)
    score = evaluate_reply(rag, FILENAMES, DF.iloc[indexes])

    description = str(config["model"])
    _push_mlflow_result(score, config, description)
    
    return rag

def _push_mlflow_result(score, config, description=None):
    """
    Enregistre les résultats de l'évaluation dans MLflow.
    Args:
        score: Dictionnaire contenant les scores et le tableau résultat
        config: Configuration à logger (cachant les clés API)
        description: Description de la tentative
    """
    with mlflow.start_run(description=description):
        df = score.pop("df_result")
        mlflow.log_table(df, artifact_file="df.json")
        mlflow.log_metrics(score)

        # On n'enregistre pas les clés sensibles
        config_no_key = {
            key: val for key, val in config.items() if not key.endswith("_key")
        }

        mlflow.log_dict(config_no_key, "config.json")

def evaluate_reply(rag, filenames, df):
    """
    Évalue la qualité des réponses générées par le modèle RAG.
    Args:
        rag: Instance du modèle RAG
        filenames: Fichiers de contexte à charger
        df: DataFrame contenant les questions et les réponses attendues
    Returns:
        Dictionnaire avec les scores et tableau de résultats
    """
    rag.load_files(filenames)

    replies = []
    for question in tqdm(df["question"]):
        replies.append(rag.reply(question))
        # On dort pour ne pas surcharger l'API Groq/OpenAI
        sleep(2)

    # On ajoute les réponses générées au DataFrame
    df["reply"] = replies
    # Calcul de la similarité sémantique entre réponses attendues et générées
    df["sim"] = df.apply(lambda row: calc_semantic_similarity(row["reply"], row["expected_reply"]), axis=1)
    # Détermination de la correction à partir d'un seuil
    df["is_correct"] = df["sim"] > .7

    return {
        "reply_similarity": df["sim"].mean(),
        "percent_correct": df["is_correct"].mean(),
        "df_result": df[["question", "reply", "expected_reply", "sim", "is_correct"]],
    }

def evaluate_retrieval(rag, filenames, df_question):
    """
    Calcule la performance du modèle RAG sur la recherche d'informations de contexte (retrieval).
    Args:
        rag: Instance du modèle RAG
        filenames: Fichiers markdown utilisés pour le corpus
        df_question: DataFrame avec questions et textes à retrouver
    Returns:
        Dictionnaire avec le Mean Reciprocal Rank (MRR) et autres infos
    """
    rag.load_files(filenames)
    ranks = []
    for _, row in df_question.iterrows():
        # Sélection des chunks de contexte pour chaque question
        chunks = rag._get_context(row.question)
        try:
            # On cherche si le texte cible est bien dans l'un des chunks sélectionnés
            rank = next(i for i, c in enumerate(chunks) if row.text_answering in c)
        except StopIteration:
            # Si pas trouvé, rank = 0
            rank = 0

        ranks.append(rank)
        
    df_question["rank"] = ranks
            
    mrr = np.mean([0 if r == 0 else 1 / r for r in ranks])

    return {
        "mrr": mrr,
        "nb_chunks": len(rag.get_chunks()),
        "df_result": df_question[["question", "text_answering", "rank"]],
    }

def calc_acceptable_chunks(chunks, text_to_find):
    """
    Pour chaque réponse (texte à retrouver), construit la liste des indices de chunks qui la contiennent.
    Args:
        chunks: Liste de textes (chunks) du corpus
        text_to_find: Liste de réponses à retrouver dans les chunks
    Returns:
        Liste de sets d'indices de chunks acceptables pour chaque réponse
    """
    acceptable_chunks = []
    for answer in text_to_find:
        chunks_ok = set(i for i, chunk in enumerate(chunks) if answer in chunk)
        acceptable_chunks.append(chunks_ok)

    return acceptable_chunks

def calc_mrr(sim_score, acceptable_chunks, top_n=5):
    """
    Calcule le Mean Reciprocal Rank (MRR) pour des scores de similarité et des chunks acceptés.
    Args:
        sim_score: Matrice de scores de similarité (shape: nb questions x nb chunks)
        acceptable_chunks: Liste de sets d'indices de chunks acceptés par question
        top_n: On ne compte que les réponses parmi les top_n premiers (optionnel)
    Returns:
        Dictionnaire MRR et liste des rangs
    """
    ranks = []
    for this_score, this_acceptable_chunks in zip(sim_score, acceptable_chunks):
        # Trie les indices selon les scores décroissants
        indexes = reversed(np.argsort(this_score))
        try:
            # On cherche le rang du premier chunk acceptable
            rank = 1 + next(i for i, idx in enumerate(indexes) if idx in this_acceptable_chunks)
        except StopIteration:
            rank = len(this_score) + 1
        
        ranks.append(rank)
        
    return {
        "mrr": sum(1 / r if r < top_n + 1 else 0 for r in ranks) / len(ranks),
        "ranks": ranks,
    }

def calc_semantic_similarity(generated_answer: str, reference_answer: str) -> float:
    """
    Calcule la similarité sémantique entre une réponse générée et la vérité terrain.
    
    Args:
        generated_answer: La réponse produite par le système RAG
        reference_answer: La réponse attendue (ground-truth)
        
    Returns:
        Score de similarité cosinus entre 0 et 1
    """
    # Génère les embeddings pour les deux textes
    embeddings = ENCODER.encode([generated_answer, reference_answer])
    generated_embedding = embeddings[0].reshape(1, -1)
    reference_embedding = embeddings[1].reshape(1, -1)
    similarity = cosine_similarity(generated_embedding, reference_embedding)[0][0]
    return float(similarity)

if __name__ == "__main__":
    # Configuration du modèle pour le découpage des textes
    model_config = {"chunk_size": 512}
    # Lancer une évaluation retrieval ou reply selon besoin
    # run_evaluate_retrieval({"model": model_config})
    run_evaluate_reply({"model": model_config})

