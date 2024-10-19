from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
from pywa import WhatsApp
from pywa.types import Message
import json
import logging
from manish import MaNish
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from typing import Optional
from fastapi.responses import JSONResponse, PlainTextResponse
import uvicorn
import PyPDF2
from pptx import Presentation
import io
import boto3
from botocore.exceptions import ClientError
import json
from pydantic import BaseModel
import httpx
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid
import logging


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger("app")

app = FastAPI()
manish = MaNish('EAAPZCdgo11ucBOxmxq3h4XlhzL7IituEG4L5yXnQbMiD0LtyakeLD0bZB0ZBdczo18hIJhW6ZAQ8aPRkw127wlXnD7Cks7IWO3KELZBHfa1jIwGZAiX9dL6ftEk4Oxng3gVdZCRgG8QlGtsEH0Hcf4j62ZAKOLZALdcuZCZC4HIp1ZBhFi65hzr2ACuGnAqCGZAS0abrfQP7KHydrvKMSwAPHfZBqnXUtZBZAHcZD',  phone_number_id='436207996245933')

#secrets
aws_access_key_id = "AKIAQ3EGUNCQ5QJVNYMJ"
aws_secret_access_key = "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"
aws_region = "us-west-2"

TUNE_AI_API_URL = "https://proxy.tune.app/chat/completions"
TUNE_AI_API_KEY = "sk-tune-IeArLOH0xZIE5cTHuNzrVGQlATFTMNDZqzh" 

# clients
qdrant_client = QdrantClient(
    url="https://13f743e4-bf99-4752-a59a-fb717c1b8d52.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key="16WN21MOBd-WI1JNKD7KTQ8Pn378xUv9PY4MzQQiwGNPf3ZUf3gZhg",
)


bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=aws_region,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)







@app.post("/webhook", include_in_schema=False)
async def webhook(request: Request):
    data = await request.json()
    changed_field = manish.changed_field(data)
    if changed_field == "messages":
        new_message = manish.get_mobile(data)
        if new_message:
            mobile = manish.get_mobile(data)
            name = manish.get_name(data)
            message_type = manish.get_message_type(data)
            logger.info(
                f"New Message; sender:{mobile} name:{name} type:{message_type}"
            )
            if message_type == "text":
                message = manish.get_message(data)
                name = manish.get_name(data)
                logger.info("Message: %s", message)
                manish.send_message(f"Hi {name}, nice to connect with you", mobile)
            elif message_type == "interactive":
                message_response = manish.get_interactive_response(data)
                intractive_type = message_response.get("type")
                message_id = message_response[intractive_type]["id"]
                message_text = message_response[intractive_type]["title"]
                logger.info(f"Interactive Message; {message_id}: {message_text}")
                manish.send_message(message_text, mobile)
            elif message_type == "location":
                message_location = manish.get_location(data)
                message_latitude = message_location["latitude"]
                message_longitude = message_location["longitude"]
                logger.info("Location: %s, %s", message_latitude, message_longitude)
                manish.send_message(f"Latitude: {message_latitude}, Longitude: {message_longitude}", mobile)
            elif message_type == "image":
                image = manish.get_image(data)
                image_id, mime_type = image["id"], image["mime_type"]
                image_url = manish.query_media_url(image_id)
                image_filename = manish.download_media(image_url, mime_type)
                logger.info(f"{mobile} sent image {image_filename}")
                manish.send_message(f"Image: {image_filename}", mobile)
            elif message_type == "video":
                video = manish.get_video(data)
                video_id, mime_type = video["id"], video["mime_type"]
                video_url = manish.query_media_url(video_id)
                video_filename = manish.download_media(video_url, mime_type)
                logger.info(f"{mobile} sent video {video_filename}")
                manish.send_message(f"Video: {video_filename}", mobile)
            elif message_type == "audio":
                audio = manish.get_audio(data)
                audio_id, mime_type = audio["id"], audio["mime_type"]
                audio_url = manish.query_media_url(audio_id)
                audio_filename = manish.download_media(audio_url, mime_type)
                logger.info(f"{mobile} sent audio {audio_filename}")
                manish.send_message(f"Audio: {audio_filename}", mobile)
            elif message_type == "file":
                file = manish.get_file(data)
                logger.info(file)
                file_id, mime_type = file["id"], file["mime_type"]
                file_url = manish.query_media_url(file_id)
                file_filename = manish.download_media(file_url, mime_type)
                logger.info(f"{mobile} sent file {file_filename}")
                manish.send_message(f"File: {file_filename}", mobile)
            elif message_type == "document":
                document = manish.get_document(data)
                document_id, mime_type = document["id"], document["mime_type"]
                document_url = manish.query_media_url(document_id)
                document_filename = manish.download_media(document_url, mime_type)
                logger.info(f"{mobile} sent document {document_filename}")
                manish.send_message(f"Document: {document_filename}", mobile)
            else:
                logger.info(f"{mobile} sent {message_type} ")
                manish.send_message(f"Sorry , I don't know how to handle {message_type}", mobile)
                logger.info(data)
        else:
            delivery = manish.get_delivery(data)
            if delivery:
                logger.info(f"Message : {delivery}")
            else:
                logger.info("No new message")
    return "ok"


VERIFY_TOKEN = "meatyhamhock"

@app.get("/webhook", include_in_schema=False)
async def verify(request: Request):
    if request.query_params.get('hub.mode') == "subscribe" and request.query_params.get("hub.challenge"):
        if not request.query_params.get('hub.verify_token') == VERIFY_TOKEN: #os.environ["VERIFY_TOKEN"]:
           return "Verification token mismatch", 403
        return int(request.query_params.get('hub.challenge'))
    return "Hello world", 200

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")