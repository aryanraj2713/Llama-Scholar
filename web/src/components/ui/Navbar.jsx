import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "../../lib/utils"; 

export const Navbar = ({ navItems, className }) => {
  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 1, y: 0 }}
        animate={{ y: 0, opacity: 1 }} 
        transition={{ duration: 0.3, ease: "easeInOut" }}
        className={cn(
          "w-full fixed top-0 z-50 py-4 px-8 bg-black text-white shadow-lg flex justify-center items-center",
          className,
          "hidden md:flex" 
        )}
      >
        <div className="flex justify-between items-center w-full max-w-7xl">
          
          <div className="text-xl font-bold">
            <a href="#" className="hover:text-gray-300">
              Llama Scholar
            </a>
          </div>

         
          <div className="flex space-x-6">
            {navItems.map((navItem, idx) => (
              <a
                key={`link-${idx}`}
                href={navItem.link}
                className={cn(
                  "text-sm hover:text-gray-300 transition-colors duration-200"
                )}
              >
                {navItem.name}
              </a>
            ))}
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
};
