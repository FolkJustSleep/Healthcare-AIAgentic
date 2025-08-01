import os
import asyncio
from langchain.agents import Tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_community.chat_models import ChatOpenAI
from mcptools import callmcp
from dotenv import load_dotenv
from rag_tools import generate_answer_with_feedback
load_dotenv()

def create_rag_tool():
    """Create a RAG tool for the AI agent"""
    async def rag_wrapper(query: str) -> str:
        """
        Use RAG (Retrieval Augmented Generation) to answer questions based on stored knowledge.
        
        Args:
            query: The question or query to search for and answer
            
        Returns:
            Generated answer based on retrieved relevant documents
        """
        try:
            # Call your RAG function - use await if it's async
            # If generate_answer_with_feedback is async, uncomment the next line:
            # answer = await generate_answer_with_feedback(query)
            answer = generate_answer_with_feedback(query)
            return f"RAG Answer: {answer}"
        except Exception as e:
            return f"RAG Error: Unable to generate answer - {str(e)}"
    
    return Tool(
        name="rag_search",
        description="Search and answer questions using Retrieval Augmented Generation (RAG). Use this tool when you need to find information from stored documents or knowledge base.",
        func=rag_wrapper
    )
async def main():
    client = MultiServerMCPClient({
        "cmkl": {
            "url": "https://mcp-hackathon.cmkl.ai/mcp",
            "transport": "streamable_http"
        }
    })
    mcp_tools = await client.get_tools()
    ragtools = create_rag_tool()
    tools = [*mcp_tools, ragtools]
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
            response = llm.invoke([{"role": "user", "content": question}])
            print("LLM Response:", response.content)
            
if __name__ == "__main__":
        asyncio.run(main())