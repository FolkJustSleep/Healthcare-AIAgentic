import asyncio
import pandas as pd
from mcptools import callmcp
from dotenv import load_dotenv
from rag import generate_answer_with_feedback
import time

load_dotenv()

async def process_questions(input_question_csv, output_submission_csv):
    # Read the CSV with questions
    df = pd.read_csv(input_question_csv)

    output_data = []  # List of dicts to store {id, answer}

    for _, row in df.iterrows():
        qid = row['id']
        question = str(row['question']).strip()
        full_question = question + " Choose the best answer from the context and if you found transliteration word you can use the original word for example ออร์โธปิดิกส์ or you can use the word Orthopedics instead."

        print(f"\n📌 Processing Question ID {qid}:\n{question}")

        try:
            respond = await callmcp(full_question)
            prompt = full_question + " this is the information from mcp " + respond
            answer = generate_answer_with_feedback(prompt)
            time.sleep(5)
        except Exception as e:
            answer = f"❌ ERROR: {str(e)}"

        output_data.append({'id': qid, 'answer': answer})

    # Convert to DataFrame and write only id + answer
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_submission_csv, index=False)
    print(f"\n✅ Submission saved to: {output_submission_csv}")

if __name__ == "__main__":
    input_csv = "test copy.csv"             # This CSV must have 'id', 'question'
    output_csv = "submission.csv"           # This will be updated with 'id', 'answer'
    asyncio.run(process_questions(input_csv, output_csv))