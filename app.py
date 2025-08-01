import asyncio
from mcptools import callmcp
from dotenv import load_dotenv
load_dotenv()
       
async def main():
    while True:
        input_text = input("Ask/Exit: ")
        if input_text.lower() == "exit":
            break
        elif input_text.lower() == "ask":
            question = input("Enter your question: ")
            await callmcp(question)
if __name__ == "__main__":
        asyncio.run(main())