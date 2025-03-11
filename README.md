# DocuBot: AI-Powered Document Q&A

## Table of Contents

[Overview](#overview)

[Project Components](#project-components)

[Features](#features)

[Instructions](#instructions)

   -   [Requirements](#requirements)

   -   [Execution](#execution)

   -   [High-level Logic Contained in Notebooks](#high-level-logic-contained-in-notebooks)

[Presentation](#presentation)

[The Team](#the-team)

## Project Overview

### Overview

This project implements a Retrieval-Augmented Generation (RAG) system for AI-driven document processing and chatbot interactions. The system enhances traditional generative AI models by incorporating external knowledge retrieval, allowing for more accurate, contextual, and dynamic responses.

### Project Components

The project consists of two main Jupyter notebooks:

   1. database_rag.ipynb <br/>
      This notebook is responsible for setting up the Retrieval-Augmented Generation (RAG) pipeline using a document database. It includes:

         -   Data Ingestion: Loading and preprocessing documents for indexing.                
         -   Embedding Generation: Using pre-trained models (e.g., OpenAI’s text embeddings, 
         -   Sentence-BERT) to convert documents into vector representations.                
         -   Indexing: Storing embeddings in a vector database such as FAISS, Pinecone, or ChromaDB for efficient retrieval.            
         -   Retrieval Mechanism: Implementing semantic search to fetch relevant documents based on query similarity.            
         -   Testing Retrieval: Running sample queries to validate the retrieval process before integrating with the chatbot.

   3.  doc_chatbot.ipynb <br/>
        This notebook develops the chatbot that leverages the RAG framework, enabling intelligent responses based on document retrieval. It includes:

         -   Loading the Vector Database: Fetching the pre-indexed embeddings from database_rag.ipynb.            
         -   LLM Integration: Connecting a generative AI model (e.g., OpenAI's GPT-4, Hugging Face models) to generate responses.
         -   Query Processing: Taking user queries, retrieving the most relevant documents, and enhancing the generative model’s response using the retrieved context. 
         -   Interactive Chat Interface: Providing a basic interface for users to interact with the chatbot.    
         -   Evaluation & Fine-tuning: Assessing the chatbot’s responses for relevance and accuracy, with potential refinements to improve performance.

### Features
   -   Document Retrieval: Extracts relevant information from a document database using embeddings.
   -   LLM Integration: Utilizes a large language model (LLM) to generate responses.
   -   Semantic Search: Uses vector-based search to improve relevance.
   -   Chatbot Interface: Interactive chatbot designed for document-based queries.

## Instructions

### Requirements
   - Python 3.7+
   - Jupyter Notebook
   - Libraries
        -    os – For handling file paths and environment variables.
        -    dotenv – To manage environment variables from a .env file.
        -    time – For timing operations and delays.
        -    uuid – To generate unique identifiers.
        -    gradio – To create an interactive web-based chatbot interface.
        -    langchain – Core library for building LLM applications.
        -    langchain_community – Additional community-supported integrations for LangChain.
        -    langchain_chroma – For working with ChromaDB as a vector store.
        -    langchain_text_splitters – For handling text chunking and document processing.

### Execution

1. Clone the repository or download the notebooks.

2. Upload documents to the approriate path.

3. Open and execute the database_rag.ipynb notebook.

4. Open and execute the doc_chatbot.ipynb notebook.

    - ![User Interface](Images/userinterface.png)

### High-level logic contained in notebooks:

#### 1. database_rag.ipynb: Document Processing & Retrieval Setup

   - Load Environment Variables
        -   Read API keys and configurations using dotenv.

   - Document Processing
        -   Load raw documents (PDFs, text, or structured data).
        -   Use langchain_text_splitters to break large documents into smaller, meaningful chunks.

   - Generate Embeddings
       -   Use a pre-trained language model (e.g., OpenAI, Hugging Face) to convert document chunks into vector representations.

   - Store in a Vector Database
       -   Use langchain_chroma to store embeddings for efficient similarity search.
       -   Assign unique identifiers to each document using uuid.

   - Test Retrieval Mechanism
      -    Query the database with a sample input.
      -    Use similarity search to retrieve relevant document chunks.
      -    Print retrieved text snippets for validation.

#### 2. doc_chatbot.ipynb: Chatbot with RAG Integration

   - Load Required Dependencies
      -   Import necessary modules (os, dotenv, langchain, gradio).
      -   Load environment variables (e.g., API keys).

   - Initialize the Vector Database
      -   Load pre-stored document embeddings from database_rag.ipynb.

   - Process User Query
      -   Accept user input through a chatbot interface (gradio).
      -   Convert the input into an embedding using the same model as document embeddings.
      -   Retrieve the most relevant document chunks from the vector database.

   - Generate a Response with RAG
      -   Send the retrieved document context + user query to a large language model (e.g., OpenAI GPT-4).
      -   Generate a response using the LLM, enriched with retrieved document data.

   - Display Results in a Chat Interface
      -   Format and present the AI-generated response to the user.
      -   Allow users to continue the conversation iteratively.


## Presentation

## The Team

[cfleming22](https://github.com/cfleming22)

[avineets87](https://github.com/avineets87)

[rjf7q](https://github.com/rjf7q)

[GIBueno25](https://github.com/GIBueno25)


