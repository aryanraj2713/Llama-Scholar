import React, { useState } from "react";
import { Card } from "./Card";
import Popup from "./Popup";

const CardsContainer = () => {
  const cardsData = [
    {
      title: "ocr",
      description: " extract key content from images.",
      popupDescription:
        "This feature allows users to capture images using their device's camera or upload them, converting the images into text notes through an OCR process. The extracted text is automatically organized . ",
    },
    //     {
    //       title: "/task",
    //       description: "creates task lists.",
    //       popupDescription: "Task management helps you organize and prioritize your daily tasks."
    //     },
    {
      title: "ppt summarization",
      description: "condenses presentations,.",
      popupDescription:
        "This feature enables users to automatically summarize presentations by extracting key points and highlights from the content. Utilizing advanced natural language processing techniques, it distills lengthy presentations into concise summaries.",
    },
    {
      title: "generate imp questions",
      description: "creates critical questions.",
      popupDescription:
        "This feature generates important questions based on the content provided, helping users engage more deeply with the material. By analyzing the key themes and concepts, it formulates relevant questions .",
    },
    //       title: "/time-table",
    //       description: "retrieve the most relevant questions from stored content.",
    //       popupDescription: "Create and manage timetables for your classes or personal tasks."
    //     },
    {
      title: "voice notes",
      description: " to convert memos into text,",
      popupDescription:
        "This feature allows users to record and store voice notes, providing a convenient way to capture important information quickly. Users can easily organize their voice notes by categories or tags for efficient recall later. The recorded notes can be played back at any time.",
    },
  ];

  const [selectedCard, setSelectedCard] = useState(null);
  const handleCardClick = (card) => {
    setSelectedCard(card);
  };

  const handleClosePopup = () => {
    setSelectedCard(null);
  };

  return (
    <div className="bg-black min-h-screen p-8">
           {" "}
      <h1 className="text-white text-5xl font-bold mb-12 text-center">
        Endpoints
      </h1>
           {" "}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-8">
               {" "}
        {cardsData.map((card, index) => (
          <Card
            key={index}
            className="h-60 w-full bg-gray-950 rounded-lg shadow-lg cursor-pointer"
            onClick={() => handleCardClick(card)}
          >
                       {" "}
            <div className="text-white text-center">
                            <h2 className="text-2xl font-bold">{card.title}</h2>
                            <p className="mt-2">{card.description}</p>         
               {" "}
            </div>
                     {" "}
          </Card>
        ))}
             {" "}
      </div>
           {" "}
      {selectedCard && (
        <Popup
          title={selectedCard.title}
          description={selectedCard.popupDescription}
          onClose={handleClosePopup}
        />
      )}
         {" "}
    </div>
  );
};

export default CardsContainer;
