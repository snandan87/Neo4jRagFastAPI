import os
import glob
from dotenv import load_dotenv

# 💡 লেটেস্ট এবং সঠিক ইম্পোর্ট পাথ (langchain_neo4j)
from langchain_neo4j import Neo4jGraph
from langchain_neo4j import LLMGraphTransformer
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

# .env ফাইল থেকে Credentials লোড করুন
load_dotenv()


def main():
    print("🔄 Connecting to Neo4j Database...")
    # refresh_schema=False রাখছি যাতে কোনো প্লাগইন ট্রাবল ছাড়াই কানেক্ট হয়
    graph = Neo4jGraph(refresh_schema=False)

    print("🤖 Initializing LLM and Graph Transformer...")
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    # নোড ও রিলেশন ফিল্টার
    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=["Customer", "Product", "Company", "Location", "Order", "Review"],
        allowed_relationships=["MANUFACTURED_BY", "LIVES_IN", "LOCATED_IN", "PLACED", "CONTAINS", "WROTE",
                               "FOR_PRODUCT"]
    )

    # ডাটা ফোল্ডার স্ক্যান ও মার্জ লজিক
    data_folder = "../data"
    file_pattern = os.path.join(data_folder, "*.txt")
    text_files = glob.glob(file_pattern)

    if not text_files:
        print(f"❌ No text files found in '{data_folder}' folder.")
        return

    merged_text = ""
    for file_path in text_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                merged_text += f"\n\n--- Content from {os.path.basename(file_path)} ---\n\n"
                merged_text += f.read()
        except Exception as e:
            print(f"⚠️ Error reading file: {str(e)}")

    print("\n📦 Creating unified document...")
    merged_document = [Document(page_content=merged_text)]

    print("✨ Transforming unified text into graph format...")
    try:
        graph_documents = transformer.convert_to_graph_documents(merged_document)

        # 💡 নোডের ইন্টারনাল প্রপার্টি ম্যাপিং (id এর মান name ফিল্ডেও কপি করা হচ্ছে)
        for doc in graph_documents:
            for node in doc.nodes:
                node.properties["name"] = node.id
        print("✅ Graph structures generated.")
    except Exception as e:
        print(f"❌ Transformation failed: {str(e)}")
        return

    # 🚀 মূল ফিক্স: ডাটাবেজে সঠিক লেবেল সেভ করা
    if graph_documents:
        print("\n🚀 Writing to Neo4j (Enforcing Explicit Node Names)...")
        try:
            graph.add_graph_documents(
                graph_documents,
                baseEntityLabel=False,  # ❌ এটি বন্ধ করায় __Entity__ তৈরি হবে না
                include_source=False  # ❌ সোর্স ডকুমেন্ট লেবেল আইডি বন্ধ করা হলো
            )
            print("🎉 Success! Every node has been created with its proper clean labels (Customer, Order, etc.).")
        except Exception as e:
            print(f"❌ Failed to write data: {str(e)}")


if __name__ == "__main__":
    main()
