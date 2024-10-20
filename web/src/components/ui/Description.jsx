import React from "react";
import { motion, useScroll, useTransform } from "framer-motion";

const Description = () => {
  // Track scroll progress
  const { scrollYProgress } = useScroll();

  // Map scroll progress to rotation (start rotating after scrolling down a bit)
  const rotate = useTransform(scrollYProgress, [0.2, 1], [0, 360]); // Starts rotating after 20% scroll

  return (
    <div className="flex flex-col md:flex-row items-center justify-between min-h-[80vh] p-4 md:p-8 bg-black rounded-lg shadow-lg mt-12 md:mt-16">
      {/* Text Section */}
      <div className="md:w-1/2 w-full text-white mb-8 md:mb-0 flex flex-col justify-center space-y-4">
        <h2 className="text-3xl md:text-5xl font-bold mb-4 md:mb-6 text-center md:text-left">
          Llama Scholar
        </h2>
        <p className="text-base md:text-lg text-center md:text-left">
          Llama Scholar is a WhatsApp bot designed for scholars to help them with
          academic tasks, research, and time management. With easy integration
          into your daily schedule, Llama Scholar enhances productivity and allows
          users to focus on their studies efficiently.
        </p>
        <p className="text-base md:text-lg text-center md:text-left">
          It leverages cutting-edge technologies and is your intelligent academic
          assistant, ensuring smooth communication and helping you stay ahead in
          your scholarly pursuits.
        </p>
      </div>

      {/* Image Section with rotation */}
      <div className="md:w-1/2 w-full flex justify-center md:justify-end">
        <motion.img
          src="/llama-bg.png" // Use relative path as per your project structure
          alt="Llama Scholar"
          className="rounded-lg shadow-lg w-64 h-64 md:w-80 md:h-80 lg:w-96 lg:h-96 object-cover"
          style={{ rotate }} // Apply rotation based on scroll progress
        />
      </div>
    </div>
  );
};

export default Description;
