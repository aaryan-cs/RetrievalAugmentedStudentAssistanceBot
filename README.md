# Retrieval-Augmented Student Assistance Bot

This repository contains the implementation of a Retrieval-Augmented Generation (RAG) chatbot designed to assist new students with queries about the admission process, department allocation, and other related information at NIT Andhra Pradesh. The chatbot uses AI agents to enhance responses based on a vector database and integrates with a Django backend.

## Features
- **Query Resolution**: Helps new students with doubts regarding admission and department allocation based on rank.
- **AI Agent**: Implements agentic AI to retrieve relevant data and generate human-like responses.
- **Vector Database**: Uses a Chroma database to store and retrieve student-related information efficiently.
  
## Setup

### Prerequisites
- Python 3.8+
- Django
- Chroma
- LangChain
- Ollama

### Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/aaryan-cs/RetrievalAugmentedStudentAssistanceBot.git
    cd RetrievalAugmentedStudentAssistanceBot
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up the environment for the Django backend:
    ```bash
    python manage.py migrate
    ```

4. Start the server:
    ```bash
    python manage.py runserver
    ```

## Usage

1. Navigate to `http://127.0.0.1:8000` in your browser.
2. Interact with the chatbot for query resolution.

## Technologies Used
- **Django**: Web framework for backend development.
- **LangChain**: Language models integration.
- **Chroma**: Vector database for efficient data retrieval.
- **Ollama**: Language model used for response generation.

