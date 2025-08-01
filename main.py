import os
import asyncio
from langchain.agents import Tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_community.chat_models import ChatOpenAI
from mcptools import callmcp
from dotenv import load_dotenv
load_dotenv()
       
async def main():
    client = MultiServerMCPClient({
        "cmkl": {
            "url": "https://mcp-hackathon.cmkl.ai/mcp",
            "transport": "streamable_http"
        }
    })
    tools = await client.get_tools()
    print("Discovered tools:", [tool.name for tool in tools])

    llm = ChatOpenAI(model="typhoon-v2-70b-instruct", temperature=0.1, api_key=os.getenv("OPENAI_API_KEY_MCP"), base_url="https://api.opentyphoon.ai/v1")
     # Create system message with tools information
    tool_descriptions = []
    for tool in tools:
        tool_descriptions.append(f"- {tool.name}: {tool.description}")
    while True:
        input_text = input("Ask/Exit: ")
        if input_text.lower() == "exit":
            break
        elif input_text.lower() == "ask":
            question = input("Enter your question: ")
            await callmcp(client, llm, tools, question)
            
if __name__ == "__main__":
        asyncio.run(main())