from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are Student Query Assistance Bot for helping students at NIT Andhra Pradesh. Use the following information to provide relevant , and concise answers which are based on the query and context:

{context}

{history}

User: {question}
Assistant:
"""
# Store conversation history
chat_history = []

def main():
    global chat_history
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    print("Hi I am Student Query Assistance Bot for NIT Andhra Pradesh! How can I help you today? (Type 'exit' or 'quit' to stop)")
    while True:
        query_text = input("\nYou: ")
        if query_text.lower() in ["exit", "quit"]:
            print("Exiting chat...")
            break

        results = db.similarity_search_with_relevance_scores(query_text, k=3)
        context_text = "\n\n---\n\n".join(doc.page_content for doc, score in results)
        
        # Format the chat history (last 3 exchanges)
        history_text = "\n".join(chat_history[-6:])  # Stores last 3 exchanges (6 lines: user & bot)

        prompt = PROMPT_TEMPLATE.format(context=context_text, history=history_text, question=query_text)

        model = ChatOllama(model="llama3.1:latest")
        response_text = model.invoke(prompt)

        # Print and store the response
        print("\nBot:", response_text.content)

        # Append user query and bot response to history
        chat_history.append(f"You: {query_text}")
        chat_history.append(f"Bot: {response_text.content}")

        # Limit history to last 3 exchanges
        chat_history = chat_history[-6:]  # Keeping last 3 interactions (each consisting of user + bot)

if __name__ == "__main__":
    main()
