import asyncio
import pandas as pd
from mcptools import callmcp
from dotenv import load_dotenv
from rag import generate_answer_with_feedback

load_dotenv()

async def process_questions_from_csv(csv_path):
    df = pd.read_csv(csv_path)  # Make sure your file is saved as .csv

    for index, row in df.iterrows():
        question = str(row['question']).strip()
        full_question = question + " Choose the best answer from the context and if you found transliteration word you can use the original word for example ออร์โธปิดิกส์ or you can use the word Orthopedics instead."

        print(f"\n📌 Processing Question ID {row['id']}:\n{question}")

        try:
            respond = await callmcp(full_question)
            prompt = full_question + " this is the information from mcp " + respond
            answer = generate_answer_with_feedback(prompt)
        except Exception as e:
            answer = f"❌ ERROR: {str(e)}"

        print(f"✅ Answer:\n{answer}")

if __name__ == "__main__":
    csv_path = "test.csv"  # Update with actual CSV path
    asyncio.run(process_questions_from_csv(csv_path))