from dotenv import load_dotenv

load_dotenv()

from src.graph import app

if __name__ == "__main__":
    print("Hello Advanced RAG")
    print(app.invoke(input={"question": "How to make an IDS?"}))