# Llama Scholar : Transform WhatsApp into Your Smart Study Assistant with Llama models

## Problem Statement

Students commonly store and share study materials via WhatsApp, yet the platform is underutilized for educational innovation. Despite its frequent use, WhatsApp lacks integrated features like summarization, content extraction, and personalized learning tools, leaving students with static, disjointed information. This presents a missed opportunity to enhance productivity and streamline learning within the platform they already rely on.

## Our Solution

LlamaScholar is an AI-driven WhatsApp bot that boosts student productivity by offering dynamic study tools. It provides real-time summaries, smart searches, and interactive learning aids—all within WhatsApp. The solution utilizes Meta's LLama models for:
- Image content extraction via its vision model
- Text generation and summarization , preparing questionnaires 
- Retrieval-Augmented Generation (RAG) for intelligent, context-aware learning, making study sessions more efficient and effective.
-  Extraction of Content from media forms like Audio 

## Implementation

The implementation of a WhatsApp bot involves utilizing Qdrant for efficient data storage and retrieval, crucial for enhancing the learning experience. By tuning AI models hosted on AWS Bedrock, the bot can handle multimodal inputs—images, texts, files, and audio—to extract necessary information seamlessly. Integration with the WhatsApp API, facilitated by wrappers like Ma-nish, allows for dynamic features such as OCR for images and RAG features, delivering a personalized and interactive learning experience for students.

![WhatsApp Image 2024-10-20 at 09 07 09](https://github.com/user-attachments/assets/743bb59f-080c-4de3-8970-c3d15aa60337)

## Tech Stack


