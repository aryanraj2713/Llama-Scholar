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
BackEnd: 
- WhatsApp Business API
- Ma-nish whatsapp wrapper
- Python 
- FastAPI
- Llama3 models 
- AWS BedRock 
- Tune AI 
- Qdrant

FrontEnd: 
- RemixJS
- Tailwind CSS
- Aceternity UI

## Data Sources

The project uses a combination of:
- User-provided data (images, notes, presentations)
- Stored educational content in the vector database
- Generated content from Llama models
- Data is collected through user interactions on WhatsApp and processed using Llama models. The system emphasizes data privacy and secure storage in the vector database.

## Video Demo


https://github.com/user-attachments/assets/e3f5d3d5-1120-45fd-93f2-5351c6cbb0f9


## USP of Our Solution

The solution's USP integrates cutting edge tools like  Qdrant for fast, personalized content retrieval, Tune AI  for optimized model performance, and a multimodal approach using LLaMA models. Leveraging WhatsApp, a platform widely used by students for real-time messaging, group chats, and multimedia sharing, it delivers dynamic content processing—text, images, and voice.
Offering real-time summarization, image extraction, and personalized study tools, it enhances productivity and streamlines learning, providing an efficient, AI-powered support system within a familiar and accessible communication platform.

## Future Scope 

The solution has significant potential for scaling:
- Adding more varied range of features into the bot 
- Integration with more educational platforms and resources
- Development of subject-specific modules for targeted learning
- Potential for adaptation to professional training and lifelong learning contexts



