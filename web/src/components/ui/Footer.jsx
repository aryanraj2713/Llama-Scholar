import React from "react";

export const profiles = [
  {
    id: 1,
    name: "Trisha",
    designation: "Frontend Dev",
    image: "https://avatars.githubusercontent.com/u/107291403?v=4",
    github: "https://github.com/trisha-thakur",
  },
  {
    id: 2,
    name: "Divyanshu",
    designation: "Backend Dev",
    image: "https://avatars.githubusercontent.com/u/91051053?v=4",
    github: "https://github.com/divyanshu1810",
  },
  {
    id: 3,
    name: "Mathangy",
    designation: "ML Dev",
    image: "https://avatars.githubusercontent.com/u/34826479?v=4",
    github: "https://github.com/mkswagger",
  },
  {
    id: 4,
    name: "Aryan",
    designation: "ML Dev",
    image: "https://avatars.githubusercontent.com/u/75358720?v=4",
    github: "https://github.com/aryanraj2713",
  },
];

const Footer = () => {
  return (
    <footer className="bg-black-900 py-4 flex justify-center">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 max-w-4xl w-full px-2">
        {profiles.map((profile) => (
          <a
            key={profile.id}
            href={profile.github}
            target="_blank"
            rel="noopener noreferrer"
            className="flex flex-col items-center text-center bg-black-900 shadow-lg rounded-lg p-2 transition-transform hover:scale-105"
          >
            <img
              src={profile.image}
              alt={profile.name}
              className="w-20 h-20 rounded-full object-cover mb-2 border-2 border-gray-700"
            />
            <h4 className="text-sm font-semibold text-white">{profile.name}</h4>
            <p className="text-xs text-gray-400">{profile.designation}</p>
          </a>
        ))}
      </div>
    </footer>
  );
};

export default Footer;
