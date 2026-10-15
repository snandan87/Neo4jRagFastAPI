
Indian Ecommerce GraphRAG Application

This repository contains a production-ready GraphRAG (Retrieval-Augmented Generation) application built with FastAPI, LangChain, Neo4j Graph Database, and OpenAI GPT models. It transforms unstructured plain text ecommerce logs into an interconnected knowledge graph and exposes an API endpoint to perform intelligent multi-hop graph question-answering.

Project Architecture

1. Unstructured Data: Merges and processes split textual logs representing entities and dynamic retail actions.
2. Knowledge Graph Transformation: Uses LangChain LLMGraphTransformer to map precise semantic entities and directional relationships.
3. Graph Database Storage: Stores vectorized structures natively inside Neo4j without using conflicting general entity labels.
4. Cypher Query Engine: Converts real-time human prompts into optimized Cypher graph language queries.
5. FastAPI Layer: Exposes a standardized REST interface serving responses directly generated via OpenAI models.

Project Structure

text
langchain-project/
│
├── data/
│   ├── ecommerce_customers.txt
│   ├── ecommerce_orders.txt
│   ├── ecommerce_products.txt
│   └── ecommerce_reviews.txt
│
├── source/
│   ├── api.py
│   └── loader2.py
│
├── .env
└── README.md


Prerequisites

• Python 3.10 or higher
• Docker Installed and Running
• OpenAI API Key

Installation & Setup

1. Clone the repository and navigate to the project directory:
bash
mkdir langchain-project && cd langchain-project

2. Initialize and activate the virtual environment using uv:
bash
uv venv
source .venv/bin/activate

For Windows Power Shell use: .venv\Scripts\Activate.ps1
3. Install all necessary development dependencies:
bash
uv pip install langchain langchain-neo4j langchain-openai fastapi uvicorn pydantic python-dotenv

