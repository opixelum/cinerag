# Importation des bibliothèques
import numpy as np
import re
import tiktoken
import openai
import yaml
import os
from dotenv import load_dotenv

try:
    from FlagEmbedding import FlagModel
    _FLAG_AVAILABLE = True
except ModuleNotFoundError:
    FlagModel = None
    _FLAG_AVAILABLE = False
from sentence_transformers import SentenceTransformer

load_dotenv()
with open("config.yml", encoding="utf-8") as _cfg_file:
    _raw = os.path.expandvars(_cfg_file.read())
CONF = yaml.safe_load(_raw)

# Initialisation du client OpenAI avec l'API Groq
CLIENT = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=CONF["groq_key"],
)

# Récupère l'encodeur de tokens pour le modèle spécifié
tokenizer = tiktoken.get_encoding("cl100k_base")


def get_model(config):
    if config:
        return RAG(**config.get("model", {}))
    else:
        return RAG()


class RAG:
    def __init__(self, chunk_size=256, chunk_method="markdown", embedder_type="flag_bge", small_window=1):
        self._chunk_size = chunk_size
        self._chunk_method = chunk_method
        self._embedder_type = embedder_type
        self._embedder = None
        self._loaded_files = set()
        self._texts = []
        self._chunks = []
        self._corpus_embedding = None
        self._client = CLIENT
        self._small_window = small_window


    def load_files(self, filenames):
        # Charge les fichiers markdown spécifiés, les découpe et met à jour l'embedding du corpus
        texts = []
        for filename in filenames:
            if filename in self._loaded_files:
                continue  # Ignore les fichiers déjà traités

            # open markdown files using utf-8 and replace invalid bytes so reading never raises
            # UnicodeDecodeError on Windows when files contain odd bytes
            with open(filename, encoding="utf-8", errors="replace") as f:
                texts.append(f.read())
                self._loaded_files.add(filename)

        self._texts += texts

        chunks_added = self._compute_chunks(texts)
        if not chunks_added:
            return
        self._chunks += chunks_added

        new_embedding = self.embed_corpus(chunks_added)
        if new_embedding is None:
            return
        if self._corpus_embedding is not None:
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
        embedder = self.get_embedder()
        if hasattr(embedder, "encode_queries"):
            return embedder.encode_queries(questions)
        return embedder.encode(questions)


    def _compute_chunks(self, texts):
        builder = CHUNK_BUILDERS.get(self._chunk_method, chunk_markdown)
        return sum(
            (builder(txt, chunk_size=self._chunk_size) for txt in texts),
            [],
        )


    def embed_corpus(self, chunks):
        if not chunks:
            return None
        embedder = self.get_embedder()
        if hasattr(embedder, "encode_passages"):
            return embedder.encode_passages(chunks)
        return embedder.encode(chunks)


    def get_embedder(self):
        if self._embedder:
            return self._embedder
        if self._embedder_type == "e5_small":
            self._embedder = _E5Embedder("intfloat/multilingual-e5-small")
        elif _FLAG_AVAILABLE:
            self._embedder = FlagModel(
                'BAAI/bge-base-en-v1.5',
                query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                use_fp16=True,
            )
        else:
            self._embedder = _E5Embedder("intfloat/multilingual-e5-small")
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


    def small2big(self, chunks, idx, window=1):
        start = max(0, idx - window)
        end = min(len(chunks), idx + window + 1)
        return "\n\n".join(chunks[start:end])

    def _get_context(self, query):
        query_embedding = self.embed_questions([query])
        sim_scores = query_embedding @ self._corpus_embedding.T
        indexes = list(np.argsort(sim_scores[0]))[-5:]
        if self._small_window:
            return [self.small2big(self._chunks, idx, window=self._small_window) for idx in indexes]
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


def chunk_markdown(md_text: str, chunk_size: int = 128, overlap: int = 40) -> list[str]:
    parsed_sections = parse_markdown_sections(md_text)
    chunks = []

    for section in parsed_sections:
        tokens = tokenizer.encode(section["content"])
        i = 0

        while i < len(tokens):
            end = min(i + chunk_size, len(tokens))
            token_chunk = tokens[i:end]
            chunk_text = tokenizer.decode(token_chunk)
            chunks.append(chunk_text)

            if end == len(tokens):
                break
            i += chunk_size - overlap

    return chunks


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[\.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def chunk_markdown_hierarchy(md_text: str, chunk_size: int = 256, overlap: int = 80) -> list[str]:
    sections = parse_markdown_sections(md_text)
    chunks: list[str] = []
    
    doc_title = ""
    if sections and sections[0]["headers"]:
        doc_title = sections[0]["headers"][0]

    for section in sections:
        content = section["content"].strip()
        if not content:
            continue
        
        chunks.append(content)
        if doc_title:
            chunks.append(f"Dans {doc_title}, {content}")
        
        sentences = _split_sentences(content)
        n = len(sentences)
        
        for i, sent in enumerate(sentences):
            chunks.append(sent)
            if doc_title:
                chunks.append(f"{doc_title}: {sent}")
            if i + 1 < n:
                pair = f"{sent} {sentences[i+1]}"
                chunks.append(pair)
                if doc_title:
                    chunks.append(f"{doc_title}: {pair}")

    return list(dict.fromkeys(chunks))


def chunk_semantic_sentences(md_text: str, chunk_size: int = 200, overlap: int = 0) -> list[str]:
    sections = parse_markdown_sections(md_text)
    chunks: list[str] = []

    doc_title = ""
    if sections and sections[0]["headers"]:
        doc_title = sections[0]["headers"][0]

    all_content = []
    for section in sections:
        content = section["content"].strip()
        if content:
            all_content.append(content)

    full_text = " ".join(all_content)
    if full_text:
        chunks.append(full_text)
        if doc_title:
            chunks.append(f"{doc_title}. {full_text}")

    for content in all_content:
        chunks.append(content)
        if doc_title:
            chunks.append(f"{doc_title}. {content}")
        
        sentences = _split_sentences(content)
        n = len(sentences)
        
        for i, sent in enumerate(sentences):
            chunks.append(sent)
            if doc_title:
                chunks.append(f"{doc_title}. {sent}")
            if i + 1 < n:
                pair = f"{sent} {sentences[i+1]}"
                chunks.append(pair)
                if doc_title:
                    chunks.append(f"{doc_title}. {pair}")

    return list(dict.fromkeys(chunks))


class _E5Embedder:
    def __init__(self, model_name="intfloat/multilingual-e5-small", normalize_embeddings=True):
        self._model = SentenceTransformer(model_name)
        self._normalize = normalize_embeddings

    def encode_queries(self, texts: list[str]):
        formatted = [f"query: {txt}" for txt in texts]
        return self._encode(formatted)

    def encode_passages(self, texts: list[str]):
        formatted = [f"passage: {txt}" for txt in texts]
        return self._encode(formatted)

    def _encode(self, texts: list[str]):
        return self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=self._normalize)


CHUNK_BUILDERS = {
    "markdown": chunk_markdown,
    "hierarchy_cards": chunk_markdown_hierarchy,
    "semantic_sentences": chunk_semantic_sentences,
}
