"use client";
import { cn } from "../../lib/utils";
import React, { useEffect, useRef, useState } from "react";
import { createNoise3D } from "simplex-noise";

export const Home = ({
  children,
  className,
  containerClassName,
  colors,
  waveWidth,
  blur = 10,
  speed = "fast",
  waveOpacity = 0.5,
  ...props
}) => {
  const noise = createNoise3D();
  let w, h, nt, i, x, ctx, canvas;
  const canvasRef = useRef(null);
  const [animationId, setAnimationId] = useState(null);

  const getSpeed = () => {
    switch (speed) {
      case "slow":
        return 0.001;
      case "fast":
        return 0.002;
      default:
        return 0.001;
    }
  };

  const init = () => {
    canvas = canvasRef.current;
    ctx = canvas.getContext("2d");
    w = ctx.canvas.width = window.innerWidth;
    h = ctx.canvas.height = window.innerHeight;
    ctx.filter = `blur(${blur}px)`;
    nt = 0;

    // Set initial background
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, w, h);

    window.onresize = function () {
      w = ctx.canvas.width = window.innerWidth;
      h = ctx.canvas.height = window.innerHeight;
      ctx.filter = `blur(${blur}px)`;
      // Ensure background remains black on resize
      ctx.fillStyle = "black";
      ctx.fillRect(0, 0, w, h);
    };
    render();
  };

  const waveColors = colors ?? [
    "#38bdf8",
    "#818cf8",
    "#c084fc",
    "#e879f9",
    "#22d3ee",
  ];

  const drawWave = (n) => {
    nt += getSpeed();
    for (i = 0; i < n; i++) {
      ctx.beginPath();
      ctx.lineWidth = waveWidth || 50;
      ctx.strokeStyle = waveColors[i % waveColors.length];
      for (x = 0; x < w; x += 5) {
        var y = noise(x / 800, 0.3 * i, nt) * 100;
        ctx.lineTo(x, y + h * 0.5);
      }
      ctx.stroke();
      ctx.closePath();
    }
  };

  const render = () => {
    // Clear with black background
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, w, h);
    
    // Set wave opacity
    ctx.globalAlpha = waveOpacity || 0.5;
    
    drawWave(5);
    const id = requestAnimationFrame(render);
    setAnimationId(id);
  };

  useEffect(() => {
    init();
    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, []);

  const [isSafari, setIsSafari] = useState(false);
  useEffect(() => {
    setIsSafari(
      typeof window !== "undefined" &&
        navigator.userAgent.includes("Safari") &&
        !navigator.userAgent.includes("Chrome")
    );
  }, []);

  return (
    <div
      className={cn(
        "h-screen flex flex-col items-center justify-center bg-black",
        containerClassName,
        "animate-fadeIn"
      )}
      style={{
        animationDuration: "0.8s",
        animationTimingFunction: "ease-out",
      }}
    >
      <canvas
        className="absolute inset-0 z-0 animate-canvasFadeIn"
        ref={canvasRef}
        id="canvas"
        style={{
          ...(isSafari ? { filter: `blur(${blur}px)` } : {}),
          backgroundColor: "black",
          animationDuration: "1s",
          animationTimingFunction: "ease-out",
        }}
      ></canvas>
      <div
        className={cn("relative z-10 text-center", className)}
        style={{
          animation: "slideUp 1s ease-out",
        }}
        {...props}
      >
        <h1 className="text-white text-7xl font-bold mb-4 animate-fadeIn">
          Llama Scholar
        </h1>
        <p className="text-2xl font-bold text-white mb-8 text-center mx-auto px-4 animate-fadeIn">
          WhatsApp bot for Scholars
        </p>
        {children}
      </div>
    </div>
  );
};
