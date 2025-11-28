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
except ModuleNotFoundError:  # pragma: no cover - optionnel
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
    def __init__(
        self,
        chunk_size=256,
        chunk_method="markdown",
        chunk_kwargs=None,
        embedder_type="flag_bge",
        embedder_params=None,
    ):
        # chunk_size : taille de découpage des textes (en tokens)
        self._chunk_size = chunk_size
        self._chunk_method = chunk_method
        self._chunk_kwargs = (chunk_kwargs or {}).copy()
        self._embedder_type = embedder_type
        self._embedder_params = embedder_params or {}
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
        # Encode une liste de questions en vecteurs d'embedding
        embedder = self.get_embedder()
        if hasattr(embedder, "encode_queries"):
            return embedder.encode_queries(questions)
        return embedder.encode(questions)


    def _compute_chunks(self, texts):
        # Découpe chaque texte en chunks de taille fixée (en tokens) et retourne la liste (applati)
        builder = CHUNK_BUILDERS.get(self._chunk_method, chunk_markdown)
        kwargs = {"chunk_size": self._chunk_size}
        kwargs.update(self._chunk_kwargs)
        return sum(
            (builder(txt, **kwargs) for txt in texts),
            [],
        )


    def embed_corpus(self, chunks):
        # Encode une liste de chunks en embeddings
        if not chunks:
            return None
        embedder = self.get_embedder()
        if hasattr(embedder, "encode_passages"):
            return embedder.encode_passages(chunks)
        return embedder.encode(chunks)


    def get_embedder(self):
        # Instancie un modèle de vectorisation si nécessaire (FlagEmbedding)
        if self._embedder:
            return self._embedder

        if self._embedder_type == "e5_small":
            params = {"model_name": "intfloat/multilingual-e5-small"}
            params.update(self._embedder_params)
            self._embedder = _E5Embedder(**params)
        else:
            params = {
                "model_name": "BAAI/bge-base-en-v1.5",
                "query_instruction_for_retrieval": "Represent this sentence for searching relevant passages:",
                "use_fp16": True,
            }
            params.update(self._embedder_params)
            model_name = params.pop("model_name")
            query_instruction = params.pop("query_instruction_for_retrieval", None)
            if not _FLAG_AVAILABLE:
                print("[RAG] FlagEmbedding indisponible, bascule automatique sur e5_small.")
                fallback_params = {"model_name": "intfloat/multilingual-e5-small"}
                fallback_params.update(self._embedder_params)
                self._embedder = _E5Embedder(**fallback_params)
            else:
                self._embedder = FlagModel(
                    model_name,
                    query_instruction_for_retrieval=query_instruction,
                    **params,
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


def chunk_markdown(md_text: str, chunk_size: int = 128, overlap: int = 40) -> list[str]:
    parsed_sections = parse_markdown_sections(md_text)
    chunks = []

    for section in parsed_sections:
        tokens = tokenizer.encode(section["content"])
        i = 0

        while i < len(tokens):
            # Calculate end index ensuring we don't exceed the total number of tokens
            end = min(i + chunk_size, len(tokens))
            token_chunk = tokens[i:end]
            chunk_text = tokenizer.decode(token_chunk)
            chunks.append(chunk_text)

            # Move chunk start forward (chunk_size - overlap) each time for overlap
            if end == len(tokens):
                break  # Avoid duplicating last bit if at end
            i += chunk_size - overlap  # In tokens

    return chunks


def chunk_markdown_hierarchy(
    md_text: str,
    chunk_size: int = 256,
    overlap: int = 80,
    min_sentences: int = 2,
) -> list[str]:
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


def chunk_coarse_to_fine(
    md_text: str,
    chunk_size: int = 320,
    overlap: int = 60,
    micro_size: int = 80,
    micro_overlap: int = 20,
) -> list[str]:
    tokens = tokenizer.encode(md_text)
    if not tokens:
        return []

    step = max(chunk_size - overlap, 1)
    chunks: list[str] = []
    macro_meta: list[tuple[int, str]] = []
    start = 0
    macro_id = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        macro_text = tokenizer.decode(tokens[start:end]).strip()
        if macro_text:
            chunks.append(f"[macro:{macro_id}]\n{macro_text}")
            macro_meta.append((macro_id, macro_text))
        macro_id += 1
        if end == len(tokens):
            break
        start += step

    for macro_id, macro_text in macro_meta:
        sentences = _split_sentences(macro_text)
        if not sentences:
            continue
        for window in _windows_from_sentences(sentences, micro_size, micro_overlap):
            chunks.append(f"[micro:{macro_id}]\n{window}")

    return chunks


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[\.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def _windows_from_sentences(
    sentences: list[str],
    chunk_size: int,
    overlap: int,
) -> list[str]:
    if chunk_size <= 0:
        return []

    keep_tokens = min(overlap, chunk_size - 1) if chunk_size > 1 else 0
    chunks: list[str] = []
    buffer: list[str] = []
    lengths: list[int] = []
    token_total = 0

    for sentence in sentences:
        clean = sentence.strip()
        if not clean:
            continue

        sent_tokens = len(tokenizer.encode(clean))
        if sent_tokens >= chunk_size:
            chunks.append(clean)
            continue

        if buffer and token_total + sent_tokens > chunk_size:
            chunks.append(" ".join(buffer))
            while lengths and token_total > keep_tokens:
                token_total -= lengths.pop(0)
                buffer.pop(0)

        buffer.append(clean)
        lengths.append(sent_tokens)
        token_total += sent_tokens

    if buffer:
        chunks.append(" ".join(buffer))

    return chunks


class _E5Embedder:
    def __init__(self, model_name="intfloat/multilingual-e5-small", normalize_embeddings=True, **kwargs):
        self._model = SentenceTransformer(model_name, **kwargs)
        self._normalize = normalize_embeddings

    def encode_queries(self, texts: list[str]):
        formatted = [f"query: {txt}" for txt in texts]
        return self._encode(formatted)

    def encode_passages(self, texts: list[str]):
        formatted = [f"passage: {txt}" for txt in texts]
        return self._encode(formatted)

    def _encode(self, texts: list[str]):
        return self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
        )


def chunk_semantic_sentences(
    md_text: str,
    chunk_size: int = 200,
    overlap: int = 0,
    max_sentences_per_chunk: int = 2,
) -> list[str]:
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


CHUNK_BUILDERS = {
    "markdown": chunk_markdown,
    "hierarchy_cards": chunk_markdown_hierarchy,
    "coarse_to_fine": chunk_coarse_to_fine,
    "semantic_sentences": chunk_semantic_sentences,
}
