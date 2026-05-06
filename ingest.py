import zipfile
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from vectorstore import ingest_documents, VECTOR_BACKEND

DATA_DIR      = Path("data")
EMBED_MODEL   = "qwen3-embedding:4b"
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

def load_akos_zip(path: Path) -> list[Document]:
    docs = []
    with zipfile.ZipFile(path) as zf:
        txt_files = sorted(
            [n for n in zf.namelist() if n.endswith(".txt") and n != "manifest.json"],
            key=lambda x: int(Path(x).stem),
        )
        for name in txt_files:
            try:
                page_num = int(Path(name).stem)
                text = zf.read(name).decode("utf-8", errors="replace").strip()
                if text:
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": path.name, "page": page_num},
                    ))
            except Exception as e:
                print(f"Napaka na strani {name}: {e}")
    return docs

def load_documents(data_dir: Path) -> list[Document]:
    if not data_dir.exists():
        print(f"  ⚠️ Mapa '{data_dir}' ne obstaja. Ustvarjam jo...")
        data_dir.mkdir(parents=True)
        return []

    docs = []
    for path in sorted(data_dir.iterdir()):
        if path.suffix not in {".txt", ".pdf"}:
            continue

        print(f"  Loading: {path.name}")
        try:
            if zipfile.is_zipfile(path):
                loaded = load_akos_zip(path)
                print(f"    → AKOS zip-PDF: {len(loaded)} pages extracted")
            elif path.suffix == ".pdf":
                loader = PyPDFLoader(str(path))
                loaded = loader.load()
                print(f"    → Standard PDF: {len(loaded)} pages")
            else:
                loader = TextLoader(str(path), encoding="utf-8")
                loaded = loader.load()
                print(f"    → Text file: {len(loaded)} document(s)")
            docs.extend(loaded)
        except Exception as e:
            print(f" Napaka pri branju {path.name}: {e}")
    return docs

if __name__ == "__main__":
    print("\n AKOS RAG — Document Ingestion")
    print("=" * 40)
    
    docs = load_documents(DATA_DIR)
    if not docs:
        print(" Ni datotek za obdelavo v mapi /data. Prekinjam.")
    else:
        print("\n[2/3] Chunking...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        print(f"  → {len(chunks)} chunks created.")

        print("\n[3/3] Embedding & storing...")
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        ingest_documents(chunks, embeddings)
        print(f" Končano! Vector store: {VECTOR_BACKEND}")