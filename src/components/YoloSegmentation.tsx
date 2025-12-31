import React, { useState, useEffect, useRef } from "react";
import busImage from "../assets/bus.jpg";
import { renderBoxes } from "./renderBox";

type Box = {
  classId: number;
  probability: number;
  box: [number, number, number, number];
  mask?: Float32Array; // Segmentation mask data
};

// --- Configuration for the segmentation model ---
const MODEL_URL =
  "https://huggingface.co/vishnusureshperumbavoor/yolov11/resolve/main/yolo11n-seg.onnx?download=true";
const CACHE_NAME = "yolo-seg-model-cache-v1"; // Changed version for segmentation model

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

export const YoloSegmentation = () => {
  const [loadingText, setLoadingText] = useState<string>("Initializing...");
  const [isModelReady, setIsModelReady] = useState<boolean>(false);
  const [predictions, setPredictions] = useState<Box[]>([]);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [uploadedVideo, setUploadedVideo] = useState<string | null>(null);
  const [isVideo, setIsVideo] = useState<boolean>(false);
  const [isDetecting, setIsDetecting] = useState<boolean>(false);
  const imageRef = useRef<HTMLImageElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const workerRef = useRef<Worker | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);
  const animationFrameRef = useRef<number | null>(null);
  const isDetectingRef = useRef<boolean>(false);
  const isProcessingRef = useRef<boolean>(false);

  // Initialize Worker and Download/Cache the Model
  useEffect(() => {
    workerRef.current = new Worker(
      new URL("./yolo-seg.worker.ts", import.meta.url),
      { type: "module" }
    );

    workerRef.current.onmessage = (event) => {
      const { status, boxes, masks, error } = event.data;
      switch (status) {
        case "model-loaded":
          console.log("Segmentation model successfully loaded in worker.");
          setIsModelReady(true);
          setLoadingText("");
          break;
        case "complete":
          // Combine boxes with their corresponding masks
          const predictionsWithMasks = boxes.map((box: Box, index: number) => ({
            ...box,
            mask: masks && masks[index] ? masks[index] : undefined,
          }));
          setPredictions(predictionsWithMasks);
          isProcessingRef.current = false;
          if (!isDetectingRef.current) {
            setLoadingText("");
          }
          break;
        case "error":
          console.error("Worker Error:", error);
          setLoadingText("Error in worker.");
          stopVideoDetection();
          break;
      }
    };

    // Fetch the model when the component mounts
    fetchAndCacheModel(MODEL_URL, CACHE_NAME, setLoadingText)
      .then((blob) => {
        setLoadingText("Model downloaded. Initializing worker...");
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

  // Continuous video detection function
  const detectVideoFrame = async () => {
    const videoElement = videoRef.current;
    if (!isModelReady || !videoElement || !workerRef.current || !isDetectingRef.current) {
      return;
    }

    if (videoElement.paused || videoElement.ended || 
        videoElement.videoWidth === 0 || videoElement.videoHeight === 0) {
      return;
    }

    if (isProcessingRef.current) {
      animationFrameRef.current = requestAnimationFrame(detectVideoFrame);
      return;
    }

    try {
      isProcessingRef.current = true;
      const imageBitmap = await createImageBitmap(videoElement);
      workerRef.current.postMessage(
        {
          type: "predict",
          imageBitmap,
        },
        [imageBitmap]
      );
    } catch (error) {
      console.error("Error creating image bitmap:", error);
      isProcessingRef.current = false;
    }

    if (isDetectingRef.current) {
      animationFrameRef.current = requestAnimationFrame(detectVideoFrame);
    }
  };

  const handlePredict = async () => {
    if (isVideo) {
      if (isDetecting) {
        return;
      }

      const videoElement = videoRef.current;
      if (!isModelReady || !videoElement || !workerRef.current) {
        return;
      }

      if (videoElement.videoWidth === 0 || videoElement.videoHeight === 0) {
        return;
      }

      isDetectingRef.current = true;
      setIsDetecting(true);
      setLoadingText("");
      detectVideoFrame();
    } else {
      const imageElement = imageRef.current;
      if (!isModelReady || !imageElement || !workerRef.current) {
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
    }
  };

  const stopVideoDetection = () => {
    isDetectingRef.current = false;
    setIsDetecting(false);
    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
  };

  useEffect(() => {
    return () => {
      stopVideoDetection();
      if (uploadedVideo) {
        URL.revokeObjectURL(uploadedVideo);
      }
    };
  }, [uploadedVideo]);

  useEffect(() => {
    const videoElement = videoRef.current;
    if (!videoElement || !isVideo) return;

    const handlePause = () => {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };

    const handlePlay = () => {
      if (isDetectingRef.current) {
        detectVideoFrame();
      }
    };

    const handleEnded = () => {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };

    videoElement.addEventListener('pause', handlePause);
    videoElement.addEventListener('play', handlePlay);
    videoElement.addEventListener('ended', handleEnded);

    return () => {
      videoElement.removeEventListener('pause', handlePause);
      videoElement.removeEventListener('play', handlePlay);
      videoElement.removeEventListener('ended', handleEnded);
    };
  }, [isVideo]);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setUploadedImage(event.target?.result as string);
        setUploadedVideo(null);
        setIsVideo(false);
        clearPredictions();
      };
      reader.readAsDataURL(file);
    }
  };

  const handleVideoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith("video/")) {
      const url = URL.createObjectURL(file);
      setUploadedVideo(url);
      setUploadedImage(null);
      setIsVideo(true);
      clearPredictions();
    }
  };

  const clearPredictions = () => {
    stopVideoDetection();
    setPredictions([]);
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx?.clearRect(0, 0, canvas.width, canvas.height);
    }
  };

  useEffect(() => {
    const image = imageRef.current;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    if (!canvas) return;
    
    if (predictions.length > 0) {
      if (isVideo && video && video.videoWidth > 0 && video.videoHeight > 0) {
        canvas.width = video.clientWidth;
        canvas.height = video.clientHeight;
        
        const videoAspect = video.videoWidth / video.videoHeight;
        const displayAspect = video.clientWidth / video.clientHeight;
        
        let displayWidth, displayHeight, offsetX, offsetY;
        
        if (displayAspect > videoAspect) {
          displayHeight = video.clientHeight;
          displayWidth = video.clientHeight * videoAspect;
          offsetX = (video.clientWidth - displayWidth) / 2;
          offsetY = 0;
        } else {
          displayWidth = video.clientWidth;
          displayHeight = video.clientWidth / videoAspect;
          offsetX = 0;
          offsetY = (video.clientHeight - displayHeight) / 2;
        }
        
        const scaleX = displayWidth / video.videoWidth;
        const scaleY = displayHeight / video.videoHeight;
        
        const scaledPredictions = predictions.map(pred => ({
          ...pred,
          box: [
            pred.box[0] * scaleX + offsetX,
            pred.box[1] * scaleY + offsetY,
            pred.box[2] * scaleX + offsetX,
            pred.box[3] * scaleY + offsetY,
          ] as [number, number, number, number],
        }));
        
        renderBoxes(canvas, scaledPredictions, video);
      } else if (!isVideo && image && image.naturalWidth > 0) {
        canvas.width = image.clientWidth;
        canvas.height = image.clientHeight;
        
        const imageAspect = image.naturalWidth / image.naturalHeight;
        const displayAspect = image.clientWidth / image.clientHeight;
        
        let displayWidth, displayHeight, offsetX, offsetY;
        
        if (displayAspect > imageAspect) {
          displayHeight = image.clientHeight;
          displayWidth = image.clientHeight * imageAspect;
          offsetX = (image.clientWidth - displayWidth) / 2;
          offsetY = 0;
        } else {
          displayWidth = image.clientWidth;
          displayHeight = image.clientWidth / imageAspect;
          offsetX = 0;
          offsetY = (image.clientHeight - displayHeight) / 2;
        }
        
        const scaleX = displayWidth / image.naturalWidth;
        const scaleY = displayHeight / image.naturalHeight;
        
        const scaledPredictions = predictions.map(pred => ({
          ...pred,
          box: [
            pred.box[0] * scaleX + offsetX,
            pred.box[1] * scaleY + offsetY,
            pred.box[2] * scaleX + offsetX,
            pred.box[3] * scaleY + offsetY,
          ] as [number, number, number, number],
        }));
        
        renderBoxes(canvas, scaledPredictions, image);
      }
    } else {
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
    }
  }, [predictions, isVideo]);

  return (
    <div style={{ padding: "20px", maxWidth: "1000px", margin: "0 auto", height: "100%" }}>
      <div style={{ textAlign: "center", marginBottom: "20px" }}>
        <h2 style={{ fontSize: "1.5em", margin: "0" }}>YOLOv11 Instance Segmentation</h2>
      </div>
      
      <div style={{ 
        background: "rgba(30, 30, 46, 0.6)",
        backdropFilter: "blur(10px)",
        border: "1px solid rgba(129, 140, 248, 0.2)",
        borderRadius: "16px",
        padding: "20px",
        boxShadow: "0 8px 32px rgba(0, 0, 0, 0.3)",
        display: "flex",
        gap: "20px",
        alignItems: "flex-start",
      }}>
        <div style={{ 
          flex: "1",
          minWidth: 0,
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
        }}>
          <div style={{ 
            position: "relative", 
            width: "100%",
            maxWidth: "600px",
            maxHeight: "450px",
            borderRadius: "12px",
            overflow: "hidden",
            boxShadow: "0 4px 12px rgba(99, 102, 241, 0.2)",
          }}>
            {!isVideo ? (
              <img
                ref={imageRef}
                src={uploadedImage || busImage}
                alt="Preview"
                style={{ 
                  width: "100%", 
                  height: "auto",
                  maxHeight: "450px",
                  objectFit: "contain",
                  display: "block",
                  position: "relative",
                  zIndex: 1,
                  borderRadius: "12px",
                }}
                onLoad={clearPredictions}
              />
            ) : (
              <video
                ref={videoRef}
                src={uploadedVideo || ""}
                style={{ 
                  width: "100%", 
                  height: "auto",
                  maxHeight: "450px",
                  objectFit: "contain",
                  display: "block",
                  backgroundColor: "#000",
                  position: "relative",
                  zIndex: 1,
                  borderRadius: "12px",
                }}
                controls
                autoPlay
                loop
                muted
                playsInline
                onLoadedMetadata={() => {
                  const canvas = canvasRef.current;
                  const video = videoRef.current;
                  if (canvas && video) {
                    canvas.width = video.clientWidth;
                    canvas.height = video.clientHeight;
                  }
                }}
              />
            )}
            <canvas
              ref={canvasRef}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                height: "100%",
                pointerEvents: "none",
                zIndex: 10,
                borderRadius: "12px",
              }}
            />
          </div>
        </div>

        <div style={{
          flex: "0 0 280px",
          display: "flex",
          flexDirection: "column",
          gap: "12px",
        }}>
          <div style={{
            padding: "14px",
            background: "rgba(99, 102, 241, 0.1)",
            borderRadius: "12px",
            border: "1px solid rgba(129, 140, 248, 0.2)",
            display: "flex",
            flexDirection: "column",
            gap: "8px",
          }}>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              style={{ display: "none" }}
            />
            <input
              ref={videoInputRef}
              type="file"
              accept="video/*"
              onChange={handleVideoUpload}
              style={{ display: "none" }}
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={!isModelReady}
              style={{
                width: "100%",
                background: "linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%)",
              }}
            >
              <span>📁 Upload Image</span>
            </button>
            <button
              onClick={() => videoInputRef.current?.click()}
              disabled={!isModelReady}
              style={{
                width: "100%",
                background: "linear-gradient(135deg, #8b5cf6 0%, #c026d3 100%)",
              }}
            >
              <span>🎥 Upload Video</span>
            </button>
            {uploadedImage && (
              <p style={{ 
                margin: "4px 0 0 0", 
                color: "#4ade80", 
                fontSize: "0.85em",
                textAlign: "center",
              }}>
                ✓ Custom image loaded
              </p>
            )}
            {uploadedVideo && (
              <p style={{ 
                margin: "4px 0 0 0", 
                color: "#c084fc", 
                fontSize: "0.85em",
                textAlign: "center",
              }}>
                ✓ Video loaded
              </p>
            )}
          </div>

          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "8px",
            }}
          >
            <button
              onClick={handlePredict}
              disabled={!isModelReady || (!!loadingText && !isVideo) || (isVideo && isDetecting)}
              style={{ 
                width: "100%",
              }}
            >
              <span>
                {loadingText && !isVideo
                  ? loadingText
                  : isVideo && isDetecting
                  ? "✅ Detecting..."
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
              padding: "10px 14px",
              background: "rgba(99, 102, 241, 0.1)",
              borderRadius: "8px",
              border: "1px solid rgba(129, 140, 248, 0.2)",
            }}>
              <p style={{ margin: 0, color: "#818cf8", fontSize: "0.85em" }}>
                {loadingText}
              </p>
            </div>
          )}
          
          {isVideo && isDetecting && (
            <div style={{ 
              padding: "10px 14px",
              background: "linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(22, 163, 74, 0.15) 100%)",
              borderRadius: "8px",
              border: "1px solid rgba(34, 197, 94, 0.3)",
            }}>
              <p style={{ margin: 0, color: "#4ade80", fontSize: "0.85em" }}>
                🔄 Live detection active
              </p>
            </div>
          )}

          {predictions.length > 0 && (
            <div style={{ 
              padding: "10px 14px",
              background: "linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%)",
              borderRadius: "8px",
              border: "1px solid rgba(129, 140, 248, 0.3)",
            }}>
              <p style={{ margin: 0, color: "#e4e4e7", fontSize: "0.85em" }}>
                ✅ Detected <b style={{ color: "#818cf8" }}>{predictions.length}</b> object(s)
              </p>
            </div>
          )}

          {!loadingText && !predictions.length && isModelReady && (
            <div style={{ 
              padding: "10px 14px",
              background: "rgba(161, 161, 170, 0.1)",
              borderRadius: "8px",
              border: "1px solid rgba(161, 161, 170, 0.2)",
            }}>
              <p style={{ margin: 0, color: "#a1a1aa", fontSize: "0.85em" }}>
                Click "Detect Objects" to analyze the image
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

