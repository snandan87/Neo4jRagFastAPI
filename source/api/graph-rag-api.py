import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain এবং Neo4j এর প্রয়োজনীয় লাইব্রেরি
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# .env ফাইল থেকে Credentials লোড করুন
load_dotenv()

# FastAPI অ্যাপ ইনিশিয়ালাইজ করুন
app = FastAPI(
    title="GraphRAG Ecommerce API",
    description="FastAPI that queries Neo4j Graph DB and answers using OpenAI",
    version="1.0.0"
)

# গ্লোাবল ভ্যারিয়েবল হিসেবে চেইনটি ডিফাইন করা হলো
graph_qa_chain = None

# 💡 এলএলএম-কে ডাটাবেজের রেজাল্ট নিখুঁতভাবে বোঝানোর জন্য কাস্টম QA প্রম্পট টেমপ্লেট
CYPHER_QA_TEMPLATE = """You are a helpful assistant that answers user questions based on the provided context extracted from a Neo4j graph database.

Context (Database Results):
{context}

User Question:
{question}

Instructions:
- Provide a clear, natural sounding, and direct answer based ONLY on the provided context.
- Do not manipulate, guess, or extrapolate information not present in the context.
- If the context contains names, locations, or IDs, use them exactly as they are.

Helpful Answer:"""

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=CYPHER_QA_TEMPLATE
)
# 💡 নোডের প্রপার্টি জটিলতা দূর করে সরাসরি নোডের ID/Name ধরে ট্রাভার্স করার নতুন প্রম্পট
CYPHER_GENERATION_TEMPLATE = """Task: Generate a Cypher statement to query a Neo4j graph database.

Available Nodes (Labels):
- Customer
- Location
- Order
- Product
- Company
- Review

Allowed Relationships (Multi-hop Paths):
- (:Customer)-[:LIVES_IN]->(:Location)
- (:Location)-[:LOCATED_IN]->(:Location)
- (:Customer)-[:PLACED]->(:Order)
- (:Order)-[:CONTAINS]->(:Product)
- (:Product)-[:MANUFACTURED_BY]->(:Company)
- (:Customer)-[:WROTE]->(:Review)
- (:Review)-[:FOR_PRODUCT]->(:Product)

Crucial Query Rules:
1. Every node in the database stores its core identifier (whether it is a Name like 'Priya Nair' or an ID like 'Cust-4005' or 'ORD-8805') in the `id` property.
2. When searching for a node by its identifier (Name or ID), ALWAYS filter using `{{id: '...'}}`. Do not use `name` property.
3. If the user mentions an ID like 'Cust-4005' or a name like 'Priya Nair', always write it as `(c:Customer {{id: 'Cust-4005'}})` or `(c:Customer {{id: 'Priya Nair'}})`.

Traversal Examples:
- Where does Rohan Das live:
  MATCH (c:Customer {{id: 'Rohan Das'}})-[:LIVES_IN]->(l:Location) RETURN l.id
- Which order did Cust-4005 place:
  MATCH (c:Customer {{id: 'Cust-4005'}})-[:PLACED]->(o:Order) RETURN o.id
- What product did Priya Nair order and which company made it:
  MATCH (c:Customer {{id: 'Priya Nair'}})-[:PLACED]->(o:Order)-[:CONTAINS]->(p:Product)-[:MANUFACTURED_BY]->(comp:Company) 
  RETURN p.id AS Product, comp.id AS Company

Question: {question}
Cypher Query:"""

CYPHER_GEN_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template=CYPHER_GENERATION_TEMPLATE
)



# API চালু হওয়ার সময় ডাটাবেজ এবং চেইন একবারেই কানেক্ট হবে
@app.on_event("startup")
def startup_event():
    global graph_qa_chain
    try:
        print("🔄 Connecting to Neo4j Graph Database...")
        graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD")
        )

        print("🤖 Initializing ChatOpenAI & Graph QA Chain...")
        llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

        # 💡 এখানে qa_prompt যুক্ত করা হয়েছে যা সরাসরি LLM এর ফাইনাল রেসপন্স নিশ্চিত করবে
        graph_qa_chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            verbose=True,
            allow_dangerous_requests=True,
            qa_prompt=QA_PROMPT  # কাস্টম প্রম্পট অ্যাসাইন করা হলো
            ,cypher_prompt=CYPHER_GEN_PROMPT
        )
        print("🎉 GraphRAG API System is fully initialized and ready!")
    except Exception as e:
        print(f"❌ Error during startup: {str(e)}")


# ইউজারের ইনপুট ফরম্যাট ডিফাইন করার জন্য Pydantic Model
class QueryRequest(BaseModel):
    prompt: str


# মূল API এন্ডপয়েন্ট (POST Request)
@app.post("/api/query")
def ask_graph_rag(request: QueryRequest):
    global graph_qa_chain

    if not graph_qa_chain:
        raise HTTPException(status_code=500, detail="Graph QA Chain is not initialized.")

    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    try:
        # ল্যাংচেইনের মাধ্যমে Neo4j থেকে রিলেটেড ডাটা তুলে এনে OpenAI দিয়ে উত্তর তৈরি
        response = graph_qa_chain.invoke({"query": request.prompt})

        # এলএলএম-এর জেনারেট করা ফাইনাল টেক্সট 'result' কী-তে থাকে
        llm_generated_answer = response.get("result")

        return {
            "status": "success",
            "prompt": request.prompt,
            "answer": llm_generated_answer  # এটি সরাসরি LLM এর তৈরি করা সঠিক বাক্য দেবে
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process request: {str(e)}")


# হেলথ চেক এন্ডপয়েন্ট
@app.get("/health")
def health_check():
    return {"status": "healthy", "database": "connected"}
