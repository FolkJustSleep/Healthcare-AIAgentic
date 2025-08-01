from langchain.schema import HumanMessage, AIMessage
from rag_tools import generate_answer_with_feedback
import json
async def callmcp(client, llm, tools, question) -> str:
    tool_descriptions = []
    for tool in tools:
        tool_descriptions.append(f"- {tool.name}: {tool.description}")
    

#     system_message = f"""You are a general assistant with access to the following tools to help answer and if you don't know the answer, you can call the tools to get the information.if question is about health insurance or medical information that need to use an advance knowledge you can use rag_search tool. if the question is thai language you need to translate the data to thai language before you answer the question.
# {chr(10).join(tool_descriptions)}
# """
    system_message = f"""คุณคือผู้ช่วยที่จะตอบคำถามเกี่ยวกับสุขภาพและการประกันสุขภาพ คุณสามารถใช้เครื่องมือที่มีเพื่อช่วยในการตอบคำถาม หากคุณไม่แน่ใจในคำตอบ คุณสามารถเรียกใช้เครื่องมือเพื่อค้นหาข้อมูลเพิ่มเติมได้ หากคำถามเป็นภาษาไทย คุณควรแปลข้อมูลเป็นภาษาไทยก่อนที่จะตอบคำถาม คุณสามารถใช้หลายเครื่องมือประกอบกันได้ทั้ง rag_search และเครื่องมืออื่น ๆ ที่มีอยู่ในระบบ
{chr(10).join(tool_descriptions)}
"""
    # Initialize conversation
    messages = [{"role": "system", "content": system_message}]
    
    # Example user message
    # user_message = question + "if you found transliteration word you can use the original word for example ออร์โธปิดิกส์ or  you can use the word Orthopedics instead."
    user_message = question + "หากคุณพบคำที่ต้องการการถอดเสียง คุณสามารถใช้คำเดิมได้ เช่น ออร์โธปิดิกส์ หรือคุณสามารถใช้คำว่า Orthopedics แทนได้ให้คุณลองใช้ tool rag_search เพื่อค้นหาข้อมูลเพิ่มเติมเกี่ยวกับคำถามนี้ก่อนหากไม่เจอให้ใช้เครื่องมืออื่น ๆ"
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
                return f"Tool {tool_name} not found"
        else:
            return response_content
    except json.JSONDecodeError:
        # Not a tool call, just a regular response
        return response_content
