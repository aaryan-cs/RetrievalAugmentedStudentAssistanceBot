from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os, shutil

DATA_PATH = "data"
CHROMA_PATH = "chroma"

def main():
    generate_data_store()

def generate_data_store():
    print("Starting data store generation")
    documents = load_documents()
    print("Loaded docs")
    chunks = split_text(documents)
    print("Split text into chunks")
    save_to_chroma(chunks)
    print("Saved to Chroma")

def load_documents():
    print("Loading directory loader")
    loader = DirectoryLoader(DATA_PATH, glob="*.txt")
    print("Loaded directory loader")
    documents = loader.load()
    print("Loaded documents")
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    if len(chunks) > 10:
        document = chunks[10]
        print(document.page_content)
        print(document.metadata)


    return chunks
def save_to_chroma(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    db = Chroma.from_documents(chunks, OllamaEmbeddings(model="nomic-embed-text"), persist_directory=CHROMA_PATH)
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()