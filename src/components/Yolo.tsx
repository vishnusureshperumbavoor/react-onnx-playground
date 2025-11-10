import React, { useState, useEffect, useRef } from "react";
import busImage from "../assets/bus.jpg";
import { renderBoxes } from "./renderBox";

type Box = {
  classId: number;
  probability: number;
  box: [number, number, number, number];
};

// --- Configuration for the model ---
const MODEL_URL =
  "https://huggingface.co/vishnusureshperumbavoor/yolov11/resolve/main/yolo11n.onnx?download=true";
const CACHE_NAME = "yolo-model-cache-v2"; // Changed version to ensure fresh cache

// --- Helper function to fetch and cache the model ---
async function fetchAndCacheModel(
  url: string,
  cacheName: string,
  setLoadingText: React.Dispatch<React.SetStateAction<string>>
): Promise<Blob> {
  const cache = await caches.open(cacheName);
  const cachedResponse = await cache.match(url);

  if (cachedResponse) {
    console.log("Model loaded from browser cache.");
    return cachedResponse.blob();
  } else {
    setLoadingText("Downloading AI model (once-off)... Please wait.");
    console.log("Model not in cache. Downloading from URL...");
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Network response was not ok: ${response.statusText}`);
    }
    await cache.put(url, response.clone()); // Store a clone in the cache
    console.log("Model downloaded and stored in cache.");
    return response.blob();
  }
}

export const Yolo = () => {
  const [loadingText, setLoadingText] = useState<string>("Initializing...");
  const [isModelReady, setIsModelReady] = useState<boolean>(false);
  const [predictions, setPredictions] = useState<Box[]>([]);
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const workerRef = useRef<Worker | null>(null);
  // REMOVED: modelBlobRef is no longer needed in the component.

  // Initialize Worker and Download/Cache the Model
  useEffect(() => {
    workerRef.current = new Worker(
      new URL("./yolo.worker.ts", import.meta.url),
      { type: "module" }
    );

    workerRef.current.onmessage = (event) => {
      const { status, boxes, error } = event.data;
      switch (status) {
        case "model-loaded":
          console.log("Model successfully loaded in worker.");
          setIsModelReady(true); // This is now the single source of truth for readiness
          setLoadingText("");
          break;
        case "complete":
          setPredictions(boxes);
          setLoadingText("");
          break;
        case "error":
          console.error("Worker Error:", error);
          setLoadingText("Error in worker.");
          break;
      }
    };

    // Fetch the model when the component mounts
    fetchAndCacheModel(MODEL_URL, CACHE_NAME, setLoadingText)
      .then((blob) => {
        setLoadingText("Model downloaded. Initializing worker...");
        // Send the model to the worker for initialization.
        // We no longer need to store the blob in the component.
        workerRef.current?.postMessage({
          type: "init",
          modelBlob: blob,
        });
      })
      .catch((error) => {
        console.error("Failed to download or cache model:", error);
        setLoadingText("Error: Could not load the model.");
      });

    return () => workerRef.current?.terminate();
  }, []);

  // Modified handlePredict to use isModelReady
  const handlePredict = async () => {
    const imageElement = imageRef.current;
    if (!isModelReady || !imageElement || !workerRef.current) {
      console.log(
        "Aborting prediction: model is not ready or image is missing."
      );
      return;
    }

    setLoadingText("Analyzing image...");
    const imageBitmap = await createImageBitmap(imageElement);

    workerRef.current.postMessage(
      {
        type: "predict",
        imageBitmap,
      },
      [imageBitmap]
    );
  };

  // --- (clearPredictions and the drawing useEffect are unchanged) ---
  const clearPredictions = () => {
    setPredictions([]);
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx?.clearRect(0, 0, canvas.width, canvas.height);
    }
  };

  useEffect(() => {
    const image = imageRef.current;
    const canvas = canvasRef.current;
    if (predictions.length > 0 && image && canvas) {
      canvas.width = image.clientWidth;
      canvas.height = image.clientHeight;
      renderBoxes(canvas, predictions);
    }
  }, [predictions]);

  return (
    <div style={{ padding: 20 }}>
      <h2>YOLO: Object Detection</h2>
      <div style={{ position: "relative", maxWidth: "700px" }}>
        <img
          ref={imageRef}
          src={busImage}
          alt="Preview"
          style={{ width: "100%", height: "auto" }}
          onLoad={clearPredictions}
        />
        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
          }}
        />
      </div>

      <div
        style={{
          marginTop: "15px",
          display: "flex",
          gap: "10px",
          alignItems: "center",
        }}
      >
        {/* Disable the button until the model is fully loaded in the worker */}
        <button
          onClick={handlePredict}
          disabled={!isModelReady || !!loadingText}
        >
          {loadingText
            ? loadingText
            : isModelReady
            ? "Predict"
            : "Initializing Model..."}
        </button>
        <button onClick={clearPredictions} disabled={!predictions.length}>
          Clear
        </button>
        {loadingText && <p>{loadingText}</p>}
      </div>
    </div>
  );
};
