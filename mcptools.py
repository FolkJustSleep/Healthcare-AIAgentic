from langchain.schema import HumanMessage, AIMessage
import json
from langchain.agents import Tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_community.chat_models import ChatOpenAI
import os
async def callmcp(question):
    client = MultiServerMCPClient({
        "cmkl": {
            "url": os.getenv("MCP_CMKL_URL"),
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
    tool_descriptions = []
    for tool in tools:
        tool_descriptions.append(f"- {tool.name}: {tool.description}")

    system_message = f"""You are a general assistant with access to the following tools to help answer and if you don't know the answer, you can call the tools to get the information.:

{chr(10).join(tool_descriptions)}
"""
    # Initialize conversation
    messages = [{"role": "system", "content": system_message}]
    
    # Example user message
    user_message = question + "if you found transliteration word you can use the original word for example ออร์โธปิดิกส์ or  you can use the word Orthopedics instead."
    messages.append({"role": "user", "content": user_message})
    
    # Get response from LLM
    response = await llm.ainvoke([HumanMessage(content=user_message)])
    
    # Check if response contains tool call
    response_content = response.content
    
    try:
        # Try to parse as JSON (tool call)
        tool_call = json.loads(response_content)
        if "tool" in tool_call and "arguments" in tool_call:
            # Find the tool
            tool_name = tool_call["tool"]
            tool_args = tool_call["arguments"]
            
            # Find the tool in our tools list
            selected_tool = None
            for tool in tools:
                if tool.name == tool_name:
                    selected_tool = tool
                    break
            
            if selected_tool:
                print(f"Using tool: {tool_name}")
                print(f"Arguments: {tool_args}")
                
                # Execute the tool
                result = await client.call_tool(tool_name, tool_args)
                print(f"Tool result: {result}")
                
                # Add tool result to conversation and get final response
                messages.append({"role": "assistant", "content": response_content})
                messages.append({"role": "user", "content": f"Tool result: {result}"})
                
                final_response = await llm.ainvoke([HumanMessage(content=f"Tool result: {result}")])
                return final_response.content
            else:
                return f"Tool {tool_name} not found"
        else:
            return response_content
    except json.JSONDecodeError:
        # Not a tool call, just a regular response
        return response_content
