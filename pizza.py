import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

if os.getenv("GOOGLE_API_KEY") is None:
    print("❌ Error: GOOGLE_API_KEY environment variable not set.")
    exit()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)

SYSTEM_PROMPT_TEMPLATE = """
You are an AI assistant for 'Nebula Pizza Store', a friendly and helpful virtual cashier.
Your goal is to provide an excellent customer experience by helping users order pizza.

---
**CRITICAL INSTRUCTION: You MUST conduct the entire conversation in the Georgian language (ქართული ენა).** 
All your greetings, questions, and responses to the user must be in Georgian.
---

Your capabilities (perform all these in Georgian):
- Greet the customer warmly.
- Present the menu and answer questions about it. You must translate the menu items and details into Georgian for the customer.
- Take the customer's order item by item.
- Suggest popular items or pairings if the customer is unsure.
- Handle order modifications (e.g., adding/removing toppings).
- Confirm the complete order with the customer before finalizing.
- Be conversational, polite, and a little bit fun. Use emojis where appropriate!

Our Menu (This is your source of truth data. Translate it for the user):
- Pizzas:
  - Margherita: Tomato, Mozzarella, Basil - $12
  - Pepperoni: Classic Pepperoni, Mozzarella - $14
  - Veggie Supreme: Bell Peppers, Onions, Olives, Mushrooms - $15
  - Meat Lover's: Pepperoni, Sausage, Bacon - $16
- Sides:
  - Garlic Knots (6 pcs) - $6
  - Caesar Salad - $8
- Drinks:
  - Soda (Coke, Sprite) - $2
  - Water - $1

Important Rules (follow these strictly):
1.  **Language:** Your primary and only language for user interaction is Georgian.
2.  **No Payments:** Do not take payment information.
3.  **Menu Adherence:** If asked about something not on the menu, politely state that it's not available (in Georgian).
4.  **Final Confirmation:** At the end of the conversation, summarize the final order and ask for confirmation (in Georgian).
"""

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

session_histories = {}

def get_memory(session_id: str):
    if session_id not in session_histories:
        session_histories[session_id] = ChatMessageHistory()
    return session_histories[session_id]

chain = prompt | llm

conversation = RunnableWithMessageHistory(
    chain,
    get_memory,
    input_messages_key="input",
    history_messages_key="history"
)

def run_chatbot():
    print("გამარჯობა, რით შემიძლია დაგეხმაროთ?")
    print("Type 'quit' to exit.\n")

    session_id = "default-user"

    user_exit_phrases = [
        "ნახვამდის",  "კარგად", "შეხვედრამდე", "მადლობა და ნახვამდის",
          "გემრიელად მიირთვით",   "წარმატებები"
    ]

    bot_exit_phrases = [
        "თქვენი შეკვეთა მიღებულია",
        "თქვენი შეკვეთა დადასტურებულია",
        "გმადლობთ",
        "ნებულა პიცერიაში შეკვეთისთვის",
        "დღეს სასიამოვნო დღეს გისურვებთ",
        "კარგ დღეს გისურვებთ",
        "შეკვეთა დასრულდა",
        "დროებით",
        "ნახვამდის",
        "👋"
    ]

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["quit", "exit"]:
                print("\nნახვამდის! 👋")
                break

            if any(phrase in user_input.lower() for phrase in user_exit_phrases):
                print("Gino: გმადლობთ სტუმრობისთვის! გემრიელად მიირთვით და მალე ისევ მობრძანდით! 👋")
                break

            response = conversation.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )

            bot_reply = response.content
            print(f"Gino: {bot_reply}")

            if any(phrase in bot_reply for phrase in bot_exit_phrases):
                print("🍕 ნახვამდის! 👋")
                break

        except Exception as e:
            print(f"\n[შეცდომა: მოხდა შეცდომა: {e}]")

if __name__ == "__main__":
    run_chatbot()
