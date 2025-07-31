import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=gemini_api_key)
def classification(question):
    prompt = f""""
    บทบาทของคุณคือ: "ผู้ช่วย AI ด้านการแพทย์"
    งานของคุณคือ "จำแนกคำถามจากผู้ใช้ให้ตรงกับหมวดหมู่ที่เหมาะสม"

    ประเภทที่ใช้ได้:
    - อาการ/ฉุกเฉิน
    - ยา/เวชภัณฑ์
    - สิทธิรักษาพยาบาล/ประกันสุขภาพ
    - ทันตกรรม
    - การวินิจฉัยโรค
    - การรักษาเฉพาะทาง
    - ความรู้ทั่วไปทางการแพทย์
    - ข้อสอบหรือคำถามแนววิชาการ
    จงจำแนกคำถามต่อไปนี้ให้ตรงกับหมวดหมู่ที่กำหนด พร้อมระบุประเภทเท่านั้น (ห้ามอธิบายเพิ่ม):
    คำถาม: {question}
    """
    # เรียกใช้โมเดล Gemini
    response = client.models.generate_content( model="gemini-2.5-flash",contents=prompt)

    return response.text.strip()
   
question= "อัตราค่าบริการตรวจคัดกรองพยาธิใบไม้ตับด้วยการตรวจปัสสาวะมีอัตราเท่าใดต่อครั้ง"
category=classification(question)
print(category)

"""client = genai.Client(api_key=gemini_api_key)
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how AI works ",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
    ),
)
print(response.text)"""