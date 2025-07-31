import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
from mcp_tools import lookup_patient
from pydantic import BaseModel, Field

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=gemini_api_key)

lookup_patient_function = {
    "name": "lookup_patient",
    "description": "Look up patient information by ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "patient_id": {
                "type": "string",
                "description": "The ID of the patient to look up.",
            },
        },
        "required": ["patient_id"],
    },
}

systemprompt = "You are a helpful medical assistant. You can look up patient information using the MCP server only. and if you don't know the answer, just say 'I don't know'."
userprompt = "I want information for patient with id POO1."
message = {
            "role": "system",
            "parts": [
                {"text": systemprompt},
            ],
            "role": "user",
            "parts": [
                {"text": userprompt},
            ],
        }

tools = types.Tool(function_declarations=[lookup_patient_function])
config = types.GenerateContentConfig(tools=[tools])
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=message,
    config=config,
)
if response.candidates[0].content.parts[0].function_call:
    function_call = response.candidates[0].content.parts[0].function_call
    print(f"Function to call: {function_call.name}")
    print(f"Arguments: {function_call.args}")
    #  In a real app, you would call your function here:
    result = lookup_patient(**function_call.args)
    print(f"Result: {result}")
else:
    print("No function call found in the response.")
    print(response.text)