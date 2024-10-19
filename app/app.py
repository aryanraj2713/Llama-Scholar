from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
from groq import Groq
import PyPDF2
from pptx import Presentation
import io
import boto3
from botocore.exceptions import ClientError
from PIL import Image
import tempfile
import base64
from pydantic import BaseModel
import httpx
from mangum import Mangum

class Task(BaseModel):
    epoch_time: int
    name: str
    email: str
    task_name: str

SUPABASE_URL = "https://jpoqergdaandvpxyirtq.supabase.co"
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Impwb3FlcmdkYWFuZHZweHlpcnRxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjkzMTc0NjIsImV4cCI6MjA0NDg5MzQ2Mn0.OZTinuAeJObdhIv9AfzMXWZcs0aofaUAn611HpEKWfs"

app = FastAPI()
handler = Mangum(app)
# Initialize Groq client
client = Groq(
    api_key="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",
)


bedrock_client = boto3.client("bedrock-runtime", region_name="us-west-2")
model_id = "meta.llama3-1-70b-instruct-v1:0"


@app.get("/")
async def root():
    return {"message": "Welcome"}

@app.get("/greeting")
async def greeting():
    return {"message": "Hello", "details": "You are using LAamAScholar Bot"}

@app.post("/summarize")
async def summarize_file(file: UploadFile = File(...)):
    content = await file.read()
    
    if file.filename.endswith('.pdf'):
        text = extract_text_from_pdf(content)
    elif file.filename.endswith('.pptx'):
        text = extract_text_from_pptx(content)
    else:
        return JSONResponse(status_code=400, content={"error": "Unsupported file format. Please upload a PDF or PPTX file."})
    
    summary = summarize_text(text)
    return {"summary": summary}

@app.post("/generate_questions")
async def generate_questions(file: UploadFile = File(...)):
    content = await file.read()
    
    if file.filename.endswith('.pdf'):
        text = extract_text_from_pdf(content)
    elif file.filename.endswith('.pptx'):
        text = extract_text_from_pptx(content)
    else:
        return JSONResponse(status_code=400, content={"error": "Unsupported file format. Please upload a PDF or PPTX file."})
    
    questions = generate_important_questions(text)
    return {"questions": questions}

@app.post("/voice-notes")
async def transcribe_voice_notes(file: UploadFile = File(...)):
    if not file.filename.endswith('.mp3'):
        return JSONResponse(status_code=400, content={"error": "Unsupported file format. Please upload an MP3 file."})
    
    # Create a temporary file to store the uploaded content
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name

    try:
        with open(temp_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(file.filename, audio_file),
                model="whisper-large-v3",
                response_format="verbose_json",
            )
        
        return {"transcription": transcription.text}
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)
def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

@app.post("/ocr/")
async def perform_ocr(file: UploadFile = File(...)):
    # Read and encode the image
    contents = await file.read()
    base64_image = encode_image(io.BytesIO(contents))

    # Prepare the message for Groq
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": "Explain what is written in the image from an educational Point of View."
                }
            ]
        }
    ]

    # Make the API call to Groq
    completion = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=messages,
        temperature=0,  # Set to 0 for more deterministic results
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )

    # Extract the OCR result from the response
    ocr_result = completion.choices[0].message.content

    return {"ocr_result": ocr_result}

def extract_text_from_pdf(content):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_pptx(content):
    prs = Presentation(io.BytesIO(content))
    text = ""
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, 'text'):
                text += shape.text + "\n"
    return text

def summarize_text(text):
    prompt = f"Summarize the following text:\n\n{text[:4000]}..."  # Truncate to 4000 characters to fit within token limit
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",
    )
    
    return chat_completion.choices[0].message.content
def generate_important_questions(text):
    prompt = f"Generate 5 important questions based on the following text:\n\n{text[:4000]}..."  # Truncate to 4000 characters to fit within token limit
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",
    )
    
    return chat_completion.choices[0].message.content

'''
    task service for storing task and sending out as reminders
'''
@app.post("/tasks")
async def create_task(task: Task):
    # Prepare the data to send to Supabase
    data = {
        "epoch_time": task.epoch_time,
        "name": task.name,
        "email": task.email,
        "task_name": task.task_name,
    }

    # Send a request to Supabase
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SUPABASE_URL}/rest/v1/tasks",
            json=data,
            headers={
                "Authorization": f"Bearer {SUPABASE_API_KEY}",
                "Content-Type": "application/json",
            },
        )

    # Check if the request was successful
    if response.status_code == 201:
        return {"message": "Task created successfully", "data": data}
    else:
        raise HTTPException(status_code=response.status_code, detail=response.json())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)