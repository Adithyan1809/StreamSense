
"""StreamSense: YouTube transcript RAG service.

Transcript -> Chunking -> Local embeddings -> Chroma -> Gemini answer
"""

import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser


# 1. Configuration

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chroma_langchain_db"
CACHE_PATH = BASE_DIR / "transcript_cache"

load_dotenv(override=True)

# Replace with a YouTube video ID.
VIDEO_ID = "Y566_T-YlNQ"

COLLECTION_NAME = "transcripts_local_embeddings"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 4

CACHE_PATH.mkdir(parents=True, exist_ok=True)

# Gemini is used only for answer generation.
chat_model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0,
)

# Embeddings run locally on your CPU.
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 32,
    },
)


# 2. Transcript retrieval

def fetch_transcript(video_id: str) -> str:
    """Fetch a YouTube transcript."""

    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id)

        transcript = " ".join(
            segment.text for segment in transcript_list
        ).strip()

        if not transcript:
            raise ValueError("The transcript is empty.")

        print("Transcript fetched successfully!")
        return transcript

    except TranscriptsDisabled:
        print(f"Transcripts are disabled for video: {video_id}")
        return ""

    except Exception as exc:
        print(
            f"Transcript retrieval failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return ""


# 3. Chunking
 

def split_transcript(transcript: str) -> list[str]:
    """Split a transcript into overlapping chunks."""

    if not transcript.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_text(transcript)

    print(f"Total chunks created: {len(chunks)}")
    return chunks


def transcript_hash(transcript: str) -> str:
    """Return a stable hash for detecting transcript changes."""

    return hashlib.sha256(
        transcript.encode("utf-8")
    ).hexdigest()


# 4. Persistent vector store

def get_vector_store() -> Chroma:
    """Create or connect to the persistent Chroma collection."""

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(DB_PATH),
    )


def get_video_metadata(vector_store: Chroma, video_id: str):
    """Get metadata from an existing indexed video, if present."""

    results = vector_store.get(
        where={"video_id": video_id},
        include=["metadatas"],
    )

    metadatas = results.get("metadatas") or []

    if not metadatas:
        return None

    return metadatas[0]


def index_chunks(
    vector_store: Chroma,
    chunks: list[str],
    video_id: str,
    content_hash: str,
) -> None:
    """Replace a video's old chunks with its latest chunks."""

    if not chunks:
        raise ValueError("Cannot index an empty list of chunks.")

    existing = vector_store.get(
        where={"video_id": video_id},
        include=["metadatas"],
    )

    old_ids = existing.get("ids") or []

    # Remove only this video's previous records.
    if old_ids:
        vector_store.delete(ids=old_ids)

    ids = [
        f"{video_id}_chunk_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "video_id": video_id,
            "chunk_index": i,
            "content_hash": content_hash,
            "source": (
                f"https://www.youtube.com/watch?v={video_id}"
            ),
        }
        for i in range(len(chunks))
    ]

    # Chroma embeds the chunks using the local model.
    vector_store.add_texts(
        texts=chunks,
        ids=ids,
        metadatas=metadatas,
    )

    print(f"Indexed {len(chunks)} chunks locally.")
    print(
        "Total records in collection:",
        vector_store._collection.count(),
    )


def ensure_indexed(
    vector_store: Chroma,
    transcript: str,
    video_id: str,
) -> None:
    """Index only when this video's transcript has changed."""

    content_hash = transcript_hash(transcript)
    existing = get_video_metadata(vector_store, video_id)

    if (
        existing
        and existing.get("content_hash") == content_hash
    ):
        print("Transcript unchanged. Reusing existing index.")
        return

    print("Transcript is new or changed. Indexing...")

    chunks = split_transcript(transcript)

    index_chunks(
        vector_store=vector_store,
        chunks=chunks,
        video_id=video_id,
        content_hash=content_hash,
    )

# 5. RAG prompt

prompt = PromptTemplate(
    template="""
You are StreamSense, an assistant that answers questions
about a specific YouTube video.

Instructions:
- Answer using only the supplied transcript context.
- Do not invent facts or introduce unsupported outside knowledge.
- If the context does not contain enough information, say:
  "I don't know based on this transcript."
- Give a clear and direct answer.

Transcript context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"],
)


# 6. LCEL retrieval and generation chain


def format_docs(documents) -> str:
    """Combine retrieved document chunks into context."""

    return "\n\n".join(
        document.page_content for document in documents
    )


def build_rag_chain(
    vector_store: Chroma,
    video_id: str,
):
    """Build the LCEL RAG chain."""

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": RETRIEVAL_K,
            "filter": {"video_id": video_id},
        },
    )

    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )

    return parallel_chain | prompt | chat_model | StrOutputParser()


# 7. Interactive application

def main():
    print("=" * 55)
    print("Starting StreamSense RAG pipeline...")
    print("=" * 55)

    if not os.getenv("GOOGLE_API_KEY"):
        print(
            "ERROR: GOOGLE_API_KEY is missing. "
            "Add it to your .env file."
        )
        return

    transcript = fetch_transcript(VIDEO_ID)

    if not transcript:
        print(
            "No transcript was retrieved. "
            "Cannot build the RAG index."
        )
        return

    try:
        vector_store = get_vector_store()

        ensure_indexed(
            vector_store=vector_store,
            transcript=transcript,
            video_id=VIDEO_ID,
        )

        main_chain = build_rag_chain(
            vector_store=vector_store,
            video_id=VIDEO_ID,
        )

        print("\nStreamSense is ready!")
        print("Type 'exit' or 'quit' to stop.")

        while True:
            question = input(
                "\nAsk a question about the video: "
            ).strip()

            if question.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            if not question:
                print("Please enter a question.")
                continue

            try:
                answer = main_chain.invoke(question)

                print("\nAnswer:")
                print(answer)

            except Exception as exc:
                print(
                    f"\nQuestion answering failed: "
                    f"{type(exc).__name__}: {exc}"
                )

    except Exception as exc:
        print(
            f"\nPipeline failed: "
            f"{type(exc).__name__}: {exc}"
        )


if __name__ == "__main__":
    main()
