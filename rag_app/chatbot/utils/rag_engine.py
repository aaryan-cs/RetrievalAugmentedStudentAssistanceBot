# rag_engine.py

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

CHROMA_PATH = "C:\\Users\\aarya\\Desktop\\Projects\\queryAssistance\\chroma"


PROMPT_TEMPLATE = """
You are Student Query Assistance Bot for helping students at NIT Andhra Pradesh. Use the following information to provide relevant , and concise answers which are based on the query and context:

{context}

{history}

User: {question}
Assistant:
"""

# Optional: tune threshold if you want filtering
# RELEVANCE_THRESHOLD = 0.4

# Maintain short conversation history
chat_history = []

def get_bot_response(query_text):
    global chat_history
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Retrieve top relevant chunks
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    # Filter based on threshold
    # relevant_chunks = [doc.page_content for doc, score in results if score >= RELEVANCE_THRESHOLD]

    # if not relevant_chunks:
    #     return "I’m sorry, I don’t know the answer to that."

    # context_text = "\n\n---\n\n".join(relevant_chunks)
    context_text = "\n\n---\n\n".join(doc.page_content for doc, score in results)

    # Include last 3 user+bot exchanges (6 lines)
    history_text = "\n".join(chat_history[-6:])

    # Prepare the prompt
    prompt = PROMPT_TEMPLATE.format(context=context_text, history=history_text, question=query_text)

    # Get response from model
    model = ChatOllama(model="llama3.1:latest")
    response = model.invoke(prompt).content.strip()

    # Update history
    chat_history.append(f"You: {query_text}")
    chat_history.append(f"Bot: {response}")
    chat_history[:] = chat_history[-6:]

    return response


# Optional: CLI mode for testing
if __name__ == "__main__":
    print("Hi, I am the NIT Andhra Pradesh Query Assistant Bot! (Type 'exit' to quit)")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat...")
            break
        answer = get_bot_response(user_input)
        print("\nBot:", answer)
