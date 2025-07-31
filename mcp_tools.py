# mcp_tools.p
import requests

def lookup_patient(patient_id: str) -> str:
    """lookup_patient ที่เชื่อมกับ MCP server"""
    response = requests.get("https://mcp-hackathon.cmkl.ai/mcp",
    json={
            "method": "tools/call",
            "params": {
                "name": "lookup_patient",
                "arguments": {
                    "patient_id": patient_id
                }
            }
        }   
        )
    return response.json().get("result", "ไม่พบข้อมูล")