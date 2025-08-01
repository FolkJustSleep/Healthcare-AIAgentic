from langchain.schema import HumanMessage, AIMessage
import json
async def callmcp(client, llm, tools, question):
    tool_descriptions = []
    for tool in tools:
        tool_descriptions.append(f"- {tool.name}: {tool.description}")

    system_message = f"""You are a general assistant with access to the following tools to help answer and if you don't know the answer, you can call the tools to get the information.:

{chr(10).join(tool_descriptions)}
"""
    # Initialize conversation
    messages = [{"role": "system", "content": system_message}]
    
    # Example user message
    user_message = question
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
                print("Final response:", final_response.content)
            else:
                print(f"Tool {tool_name} not found")
        else:
            print("Agent response:", response_content)
    except json.JSONDecodeError:
        # Not a tool call, just a regular response
        print("Agent response:", response_content)
 