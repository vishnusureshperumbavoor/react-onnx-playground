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
      // Set canvas to match the displayed size
      canvas.width = image.clientWidth;
      canvas.height = image.clientHeight;
      
      // Calculate scale factors from natural to displayed size
      const scaleX = image.clientWidth / image.naturalWidth;
      const scaleY = image.clientHeight / image.naturalHeight;
      
      // Scale the predictions to match displayed size
      const scaledPredictions = predictions.map(pred => ({
        ...pred,
        box: [
          pred.box[0] * scaleX,
          pred.box[1] * scaleY,
          pred.box[2] * scaleX,
          pred.box[3] * scaleY,
        ] as [number, number, number, number],
      }));
      
      renderBoxes(canvas, scaledPredictions);
    }
  }, [predictions]);

  return (
    <div style={{ padding: "20px", maxWidth: "1200px", margin: "0 auto" }}>
      <div style={{ textAlign: "center", marginBottom: "24px" }}>
        <h2>YOLOv11 Object Detection</h2>
      </div>
      
      <div style={{ 
        background: "rgba(30, 30, 46, 0.6)",
        backdropFilter: "blur(10px)",
        border: "1px solid rgba(129, 140, 248, 0.2)",
        borderRadius: "16px",
        padding: "24px",
        boxShadow: "0 8px 32px rgba(0, 0, 0, 0.3)",
        display: "flex",
        gap: "24px",
        alignItems: "flex-start",
      }}>
        {/* Left side - Image */}
        <div style={{ 
          flex: "1",
          minWidth: 0,
          maxWidth: "600px",
        }}>
          <div style={{ 
            position: "relative", 
            width: "100%",
            maxHeight: "600px",
            borderRadius: "12px",
            overflow: "hidden",
            boxShadow: "0 4px 12px rgba(99, 102, 241, 0.2)",
          }}>
            <img
              ref={imageRef}
              src={busImage}
              alt="Preview"
              style={{ 
                width: "100%", 
                height: "auto",
                maxHeight: "600px",
                objectFit: "contain",
                display: "block",
              }}
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
        </div>

        {/* Right side - Controls and Output */}
        <div style={{
          flex: "0 0 300px",
          display: "flex",
          flexDirection: "column",
          gap: "16px",
        }}>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "10px",
            }}
          >
            <button
              onClick={handlePredict}
              disabled={!isModelReady || !!loadingText}
              style={{ width: "100%" }}
            >
              <span>
                {loadingText
                  ? loadingText
                  : isModelReady
                  ? "🎯 Detect Objects"
                  : "⏳ Initializing..."}
              </span>
            </button>
            <button 
              onClick={clearPredictions} 
              disabled={!predictions.length}
              style={{
                background: "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)",
                width: "100%",
              }}
            >
              <span>Clear</span>
            </button>
          </div>
          
          {loadingText && (
            <div style={{ 
              padding: "12px 20px",
              background: "rgba(99, 102, 241, 0.1)",
              borderRadius: "10px",
              border: "1px solid rgba(129, 140, 248, 0.2)",
            }}>
              <p style={{ margin: 0, color: "#818cf8", fontSize: "0.9em" }}>
                {loadingText}
              </p>
            </div>
          )}
          
          {predictions.length > 0 && (
            <div style={{ 
              padding: "12px 20px",
              background: "linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%)",
              borderRadius: "10px",
              border: "1px solid rgba(129, 140, 248, 0.3)",
            }}>
              <p style={{ margin: 0, color: "#e4e4e7", fontSize: "0.95em" }}>
                ✅ Detected <b style={{ color: "#818cf8" }}>{predictions.length}</b> object(s)
              </p>
            </div>
          )}

          {!loadingText && !predictions.length && isModelReady && (
            <div style={{ 
              padding: "12px 20px",
              background: "rgba(161, 161, 170, 0.1)",
              borderRadius: "10px",
              border: "1px solid rgba(161, 161, 170, 0.2)",
            }}>
              <p style={{ margin: 0, color: "#a1a1aa", fontSize: "0.9em" }}>
                Click "Detect Objects" to analyze the image
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
