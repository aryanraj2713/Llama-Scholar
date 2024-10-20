// Popup.js
import React, { useEffect } from "react";

const Popup = ({ title, description, onClose }) => {
  useEffect(() => {
    // Close the popup when Esc key is pressed
    const handleEsc = (e) => {
      if (e.key === "Escape") {
        onClose();
      }
    };

    document.addEventListener("keydown", handleEsc);

    // Cleanup event listener on unmount
    return () => {
      document.removeEventListener("keydown", handleEsc);
    };
  }, [onClose]);

  return (
    <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 backdrop-blur-md transition-opacity duration-300">
      <div className="bg-gray-800 bg-opacity-70 rounded-lg p-8 shadow-lg w-[90%] sm:w-[400px] h-[400px] transform transition-transform duration-300 scale-0 opacity-0 animate-popup-enter">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-bold text-white">{title}</h2>
          <button onClick={onClose} className="text-white text-2xl">
            &times; {/* Close button */}
          </button>
        </div>
        <p className="text-white">{description}</p>
      </div>
    </div>
  );
};

export default Popup;
