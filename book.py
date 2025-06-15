import os
import random
from dotenv import load_dotenv

from langchain import hub
from langchain.agents import AgentExecutor, create_json_chat_agent
from langchain_core.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

print("Loading environment variables...")
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Error: GOOGLE_API_KEY environment variable not set.")
if not os.getenv("PINECONE_API_KEY"):
    raise ValueError("Error: PINECONE_API_KEY environment variable not set.")

print("Initializing models and connections...")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

PINECONE_INDEX_NAME = "langchain-books-pure-v1"
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    raise ValueError(
        f"Pinecone index '{PINECONE_INDEX_NAME}' not found. "
        "Please run the data ingestion script first."
    )

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings
)
print("Successfully connected to Pinecone index.")

def search_book_database(query: str) -> str:
    """
    Performs a similarity search on the book vector database.
    Returns a formatted string of the top 3 results.
    """
    print(f"\n[Tool Action: Searching database for '{query}']")
    results = vectorstore.similarity_search(query, k=3)
    if not results:
        return "No relevant books found in the database."

    response_string = "Found the following relevant books in the database:\n"
    for doc in results:
        title = doc.metadata.get('title', 'N/A')
        authors = doc.metadata.get('authors', 'N/A')
        response_string += f"- Title: {title}, Authors: {authors}\n"
    return response_string

book_search_tool = Tool(
    name="book_database_search",
    func=search_book_database,
    description="Searches a vector database of books for relevant titles and authors based on a query. Use this to find inspiration, check for similar plot points, or understand character archetypes from existing works."
)

tools = [book_search_tool]

SYSTEM_PROMPT_TEMPLATE = """
You are "Muse," a friendly and brilliant AI creative partner for a book author.
Your goal is to help the user brainstorm, develop characters, outline plots, and overcome writer's block.
You have access to a special tool called `book_database_search`.
Use it when the user asks for inspiration from existing books, wants to know about common tropes, or wants to find examples of specific plot devices.
Before suggesting a completely new idea, you can quickly search the database to see if similar concepts exist, which can enrich your answer.
Be encouraging, collaborative, and inspiring. Ask clarifying questions to better understand the user's vision.
"""

prompt = hub.pull("hwchase17/react-chat-json")
prompt.messages[0].prompt.template = SYSTEM_PROMPT_TEMPLATE

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

agent = create_json_chat_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)

def run_creative_agent():
    """Runs the main loop for the Creative Workflow Agent."""
    print("\n🤖 Hello! I'm Muse, your creative partner. How can I help you with your book today?")
    print("Type 'quit' to exit.\n")

    exit_phrases = ["quit", "exit", "goodbye", "thanks", "thank you", "bye"]
    goodbye_messages = [
        "It was a pleasure helping you. Happy writing! 👋",
        "Take care and let your creativity flow! ✨",
        "Looking forward to your next masterpiece. Goodbye! 📚",
        "Until next time, author! 🎩"
    ]

    while True:
        try:
            user_input = input("You: ").strip().lower()

            if any(phrase in user_input for phrase in exit_phrases):
                print(f"\nMuse: {random.choice(goodbye_messages)}")
                break

            response = agent_executor.invoke({"input": user_input})
            print(f"Muse: {response['output']}")

        except Exception as e:
            print(f"\n[Error: An unexpected error occurred: {e}]")

if __name__ == "__main__":
    run_creative_agent()
