import os
import glob
from dotenv import load_dotenv

from langchain_community.graphs import Neo4jGraph
from langchain_community.document_loaders import TextLoader
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

load_dotenv()


def main():
    print("🔄 Connecting to Neo4j Database...")
    graph = Neo4jGraph(refresh_schema=False)

    print("🤖 Initializing LLM and Graph Transformer...")
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=["Customer", "Product", "Company", "Location", "Order", "Review"],
        allowed_relationships=["MANUFACTURED_BY", "LIVES_IN", "LOCATED_IN", "PLACED", "CONTAINS", "WROTE",
                               "FOR_PRODUCT"]
    )

    data_folder = "../data"
    file_pattern = os.path.join(data_folder, "*.txt")
    text_files = glob.glob(file_pattern)

    if not text_files:
        print(f"❌ No text files found in '{data_folder}' folder. Please check the path.")
        return

    print(f"📂 Found {len(text_files)} files to merge: {[os.path.basename(f) for f in text_files]}")

    # ৪. সব ফাইলের কন্টেন্ট একটি একক স্ট্রিং-এ মার্জ করুন
    merged_text = ""
    for file_path in text_files:
        file_name = os.path.basename(file_path)
        print(f"📖 Reading and merging: {file_name}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # প্রতিটি ফাইলের মাঝে একটু ফাঁকা জায়গা এবং ব্রেক দেওয়া হচ্ছে যাতে এলএলএম মিক্সড আপ না করে
                merged_text += f"\n\n--- Content from {file_name} ---\n\n"
                merged_text += f.read()
        except Exception as e:
            print(f"⚠️ Error reading file {file_name}: {str(e)}")

    # ৫. মার্জ করা টেক্সটকে একটি একক ল্যাংচেইন ডকুমেন্টে রূপান্তর করুন
    print("\n📦 Creating a single unified document for the entire dataset...")
    merged_document = [Document(page_content=merged_text)]

    # ৬. সম্পূর্ণ মার্জড ডাটাকে একসাথে গ্রাফে রূপান্তর করুন
    print("✨ Transforming unified text into graph format (Extracting cross-file relations)...")
    try:
        graph_documents = transformer.convert_to_graph_documents(merged_document)
        print(f"✅ Successfully extracted {len(graph_documents)} graph structure(s) with full context!")
    except Exception as e:
        print(f"❌ Transformation failed: {str(e)}")
        return

    # ৭. সম্পূর্ণ গ্রাফ ডাটা Neo4j ডাটাবেজে সেভ করুন
    if graph_documents:
        print(f"\n🚀 Writing graph documents to Neo4j...")
        try:
            # Neo4j স্বয়ংক্রিয়ভাবে ডুপ্লিকেট নোডগুলো মার্জ করে রিলেশন তৈরি করে দেবে
            graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=False)
            print(
                "🎉 Success! All Indian Ecommerce cross-file relations have been successfully loaded into Neo4j Graph DB.")
        except Exception as e:
            print(f"❌ Failed to write data to Neo4j: {str(e)}")
    else:
        print("\n🛑 No graph data was extracted. Database update skipped.")


if __name__ == "__main__":
    main()
