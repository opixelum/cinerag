# Importation des bibliothèques
import numpy as np
import re
import tiktoken
import openai
import yaml

from FlagEmbedding import FlagModel

# Chargement de la configuration depuis le fichier YAML
CONF = yaml.safe_load(open("config.yml"))

# Initialisation du client OpenAI avec l'API Groq
CLIENT = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=CONF["groq_key"],
)

# Récupère l'encodeur de tokens pour le modèle spécifié
tokenizer = tiktoken.get_encoding("cl100k_base")


def get_model(config):
    # Instancie et retourne le modèle RAG avec la configuration donnée
    return RAG(**config["model"])


class RAG:
    def __init__(self, chunk_size=256):
        # chunk_size : taille de découpage des textes (en tokens)
        self._chunk_size = chunk_size
        self._embedder = None                   # Embedding model (sera chargé à la demande)
        self._loaded_files = set()              # Fichiers déjà chargés pour éviter le doublon
        self._texts = []                        # Liste des textes bruts déjà chargés
        self._chunks = []                       # Liste des chunks de texte découpés
        self._corpus_embedding = None           # Embeddings du corpus, sous forme de matrice numpy
        self._client = CLIENT                   # Client OpenAI/Groq utilisé pour la génération


    def load_files(self, filenames):
        # Charge les fichiers markdown spécifiés, les découpe et met à jour l'embedding du corpus
        texts = []
        for filename in filenames:
            if filename in self._loaded_files:
                continue  # Ignore les fichiers déjà traités

            with open(filename) as f:
                texts.append(f.read())
                self._loaded_files.add(filename)

        self._texts += texts

        # Découpe les nouveaux textes en chunks
        chunks_added = self._compute_chunks(texts)
        self._chunks += chunks_added

        # Calcule les embeddings pour les nouveaux chunks
        new_embedding = self.embed_corpus(chunks_added)
        if self._corpus_embedding is not None:
            # Ajoute les nouveaux embeddings à ceux déjà existants
            self._corpus_embedding = np.vstack([self._corpus_embedding, new_embedding])
        else:
            self._corpus_embedding = new_embedding


    def get_corpus_embedding(self):
        # Retourne la matrice d'embeddings du corpus
        return self._corpus_embedding


    def get_chunks(self):
        # Retourne la liste complète des chunks de texte
        return self._chunks


    def embed_questions(self, questions):
        # Encode une liste de questions en vecteurs d'embedding
        embedder = self.get_embedder()
        return embedder.encode(questions)


    def _compute_chunks(self, texts):
        # Découpe chaque texte en chunks de taille fixée (en tokens) et retourne la liste (applati)
        return sum(
            (chunk_markdown(txt, chunk_size=self._chunk_size) for txt in texts),
            [],
        )


    def embed_corpus(self, chunks):
        # Encode une liste de chunks en embeddings
        embedder = self.get_embedder()
        return embedder.encode(chunks)


    def get_embedder(self):
        # Instancie un modèle de vectorisation si nécessaire (FlagEmbedding)
        if not self._embedder:
            self._embedder = FlagModel(
                'BAAI/bge-base-en-v1.5',
                query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                use_fp16=True,
            )
        return self._embedder


    def reply(self, query):
        # Produit une réponse à la question "query" à partir du contexte extrait
        prompt = self._build_prompt(query)
        # Appelle le modèle de chat Groq/OpenAI avec le prompt généré
        res = self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-20b",
        )
        return res.choices[0].message.content
        

    def _build_prompt(self, query):
        # Construit le prompt à envoyer au modèle pour répondre à la question.
        context_str = "\n".join(self._get_context(query))

        return f"""Context information is below.
        ---------------------
        {context_str}
        ---------------------
        Given the context information and not prior knowledge, answer the query.
        If the answer is not in the context information, reply \"I cannot answer that question\".
        Query: {query}
        Answer:"""

    def _get_context(self, query):
        # Sélectionne les 5 chunks de contexte les plus similaires à la question (cosine similarity)
        query_embedding = self.embed_questions([query])
        sim_scores = query_embedding @ self._corpus_embedding.T
        indexes = list(np.argsort(sim_scores[0]))[-5:]  # Top 5 plus proches
        return [self._chunks[i] for i in indexes]
    

def count_tokens(text: str) -> int:
    # Retourne le nombre de tokens pour un texte donné (selon le tokenizer utilisé)
    return len(tokenizer.encode(text))


def parse_markdown_sections(md_text: str) -> list[dict[str, str]]:
    """
    Analyse du texte markdown pour retourner une liste de sections :
    chaque section contient la hiérarchie des headers et le contenu associé.
    Préserve toute la hiérarchie des headers (niveau).
    """
    pattern = re.compile(r"^(#{1,6})\s*(.+)$")
    lines = md_text.splitlines()

    sections = []
    header_stack = []
    current_section = {"headers": [], "content": ""}

    for line in lines:
        match = pattern.match(line)
        if match:
            level = len(match.group(1))    # Niveau du header '#', '##', ...
            title = match.group(2).strip()

            # Sauvegarde la section précédente si elle existe
            if current_section["content"]:
                sections.append(current_section)

            # Adapte la pile de headers selon la profondeur
            header_stack = header_stack[:level - 1]
            header_stack.append(title)

            current_section = {
                "headers": header_stack.copy(),
                "content": ""
            }
        else:
            current_section["content"] += line + "\n"

    if current_section["content"]:
        # Ajoute la dernière section à la liste
        sections.append(current_section)

    return sections


def chunk_markdown(md_text: str, chunk_size: int = 128) -> list[dict]:
    # Découpe le texte markdown analysé en chunks de taille maximum chunk_size (en tokens)
    parsed_sections = parse_markdown_sections(md_text)
    chunks = []

    for section in parsed_sections:
        tokens = tokenizer.encode(section["content"])
        # Créée des sous-listes de tokens d'une taille maximale chunk_size
        token_chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size) if tokens[i:i + chunk_size]]

        for token_chunk in token_chunks:
            # Décode chaque chunk de tokens en texte
            chunk_text = tokenizer.decode(token_chunk)
            chunks.append(chunk_text)

    return chunks
