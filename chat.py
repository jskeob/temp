from pathlib import Path
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from vectorstore import get_vectorstore, RETRIEVER_SETTINGS, VECTOR_BACKEND

EMBED_MODEL = "qwen3-embedding:4b"
LLM_MODEL   = "qwen3-vl:8b"

SYSTEM_PROMPT = (
    "Ste specializirani pomočnik Agencije za komunikacijska omrežja in storitve (AKOS).\n"
    "Vaše znanje temelji izključno na priloženih dokumentih\n\n"
    "Pravila:\n"
    "1. Odgovarjajte samo na podlagi podanega konteksta.\n"
    "2. Če informacije ni v dokumentu, recite: 'Podatek ni na voljo v dokumentu.'\n in koncaj"
    "3. Uporabljajte strokoven in uraden jezik.\n"
    "4. Če je mogoče, navedite številko strani iz metapodatkov.\n\n"
    "Kontekst:\n{context}"
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def build_chain(retriever, llm):
    return (
        RunnablePassthrough.assign(docs=lambda x: retriever.invoke(x["question"]))
        | RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | {
            "answer": PROMPT | llm | StrOutputParser(),
            "docs": lambda x: x["docs"],
        }
    )

def print_sources(docs):
    print("\n  Uporabljeni viri:")
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "unknown")
        page= doc.metadata.get("page", "")
        snippet = doc.page_content[:120].replace("\n", " ").strip()
        page_str = f" | stran {page}" if page != "" else ""
        print(f'   [{i}] {Path(src).name}{page_str}: "{snippet}..."')

def main():
    print("\n AKOS RAG — PoC Chatbot")
    print("#" * 40)
    print(f"  Backend:    {VECTOR_BACKEND}")
    print(f"  LLM:        {LLM_MODEL}")
    print(f"  Embeddings: {EMBED_MODEL}")
    print("  Napiši 'odnehaj' ali 'koncaj' ali 'q' za zaprtje.\n")

    llm = ChatOllama(model=LLM_MODEL, temperature=0, streaming=True, num_ctx=4096)
    embeddings  = OllamaEmbeddings(model=EMBED_MODEL)
    
    try:
        vectorstore = get_vectorstore(embeddings)
        retriever   = vectorstore.as_retriever(**RETRIEVER_SETTINGS)
        chain       = build_chain(retriever, llm)
        print(" Pripravljen!\n")
    except Exception as e:
        print(f" Napaka pri nalaganju baze: {e}")
        return
    
    while True:
        try:
            question = input("Uporabnik: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in {"odnehaj", "koncaj", "q"}:
            print("Adijo!")
            break
        print("\nOdgovor: ", end="", flush=True)
        source_docs = []
        try:
            for chunk in chain.stream({"question": question}):
                if "answer" in chunk:
                    print(chunk["answer"], end="", flush=True)
                if "docs" in chunk:
                    source_docs = chunk["docs"]
            print()
            if source_docs:
                print_sources(source_docs)
        except Exception as e:
            print(f"\n  Napaka: {e}")

        print()

if __name__ == "__main__":
    main()