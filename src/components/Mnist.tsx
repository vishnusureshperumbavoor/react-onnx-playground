import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

export function Mnist() {
  const [output, setOutput] = useState<Float32Array | null>(null);
  const [loading, setLoading] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);

  const sessionRef = useRef<ort.InferenceSession | null>(null);
  useEffect(() => {
    async function loadModel() {
      setLoading(true);
      try {
        sessionRef.current = await ort.InferenceSession.create("mnist-8.onnx");
        console.log("ONNX model loaded:", sessionRef.current);
      } catch (e) {
        console.error("ONNX model load failed:", e);
      } finally {
        setLoading(false);
      }
    }
    loadModel();
  }, []);

  function clearCanvas() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    setOutput(null);

    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.lineWidth = 16;
    ctx.lineCap = "round";
    ctx.strokeStyle = "black";
  }

  useEffect(() => {
    clearCanvas();
  }, []);

  function getPointerPosition(e: React.MouseEvent | React.TouchEvent) {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    let clientX: number, clientY: number;
    if ("touches" in e && e.touches.length > 0) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    } else if ("changedTouches" in e && e.changedTouches.length > 0) {
      clientX = e.changedTouches[0].clientX;
      clientY = e.changedTouches[0].clientY;
    } else {
      if ("clientX" in e && "clientY" in e) {
        clientX = (e as React.MouseEvent).clientX;
        clientY = (e as React.MouseEvent).clientY;
      } else {
        clientX = 0;
        clientY = 0;
      }
    }
    return {
      x: clientX - rect.left,
      y: clientY - rect.top,
    };
  }

  function startDrawing(e: React.MouseEvent | React.TouchEvent) {
    setIsDrawing(true);
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const { x, y } = getPointerPosition(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
  }

  function draw(e: React.MouseEvent | React.TouchEvent) {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const { x, y } = getPointerPosition(e);
    ctx.lineTo(x, y);
    ctx.stroke();
  }

  function endDrawing() {
    setIsDrawing(false);
    runMnistModel();
  }

  function getImageData() {
    const canvas = canvasRef.current;
    if (!canvas) return new Float32Array(1 * 1 * 28 * 28).fill(0);
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext("2d");
    if (!tempCtx) return new Float32Array(1 * 1 * 28 * 28).fill(0);
    tempCtx.drawImage(canvas, 0, 0, 28, 28);
    const imgData = tempCtx.getImageData(0, 0, 28, 28).data;
    const arr = new Float32Array(1 * 1 * 28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
      arr[i] = (255 - imgData[i * 4]) / 255;
    }
    return arr;
  }

  async function runMnistModel() {
    setLoading(true);
    setOutput(null);
    try {
      const session = sessionRef.current;
      if (!session) throw new Error("Model not loaded");
      const inputType = "float32";
      const imgData = getImageData();
      const inputShape = [1, 1, 28, 28];
      const inputTensor = new ort.Tensor(inputType, imgData, inputShape);
      const feeds: Record<string, ort.Tensor> = {};
      feeds[session.inputNames[0]] = inputTensor;
      const results = await session.run(feeds);
      const outputTensor = results[session.outputNames[0]];
      if (outputTensor.data instanceof Float32Array) {
        setOutput(outputTensor.data);
      } else {
        setOutput(null);
        console.error("Model output is not a Float32Array:", outputTensor.data);
      }
    } catch (e) {
      console.error("ONNX inference failed:", e);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ padding: "30px 20px", maxWidth: "600px", margin: "0 auto" }}>
      <div style={{ textAlign: "center", marginBottom: "24px" }}>
        <h2>MNIST Recognition</h2>
        <p style={{ fontSize: "0.95em", color: "#a1a1aa", margin: "8px 0 0 0" }}>
          Draw a digit and let AI recognize it
        </p>
      </div>
      
      <div style={{ 
        background: "rgba(30, 30, 46, 0.6)",
        backdropFilter: "blur(10px)",
        border: "1px solid rgba(129, 140, 248, 0.2)",
        borderRadius: "16px",
        padding: "24px",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        boxShadow: "0 8px 32px rgba(0, 0, 0, 0.3)",
      }}>
        <canvas
          ref={canvasRef}
          width={280}
          height={280}
          style={{
            border: "2px solid rgba(129, 140, 248, 0.3)",
            borderRadius: "12px",
            background: "white",
            touchAction: "none",
            boxShadow: "0 4px 12px rgba(99, 102, 241, 0.2)",
          }}
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={endDrawing}
          onMouseLeave={endDrawing}
          onTouchStart={startDrawing}
          onTouchMove={draw}
          onTouchEnd={endDrawing}
        />
        <div style={{ marginTop: 16 }}>
          <button onClick={clearCanvas}>
            <span>Clear Canvas</span>
          </button>
        </div>
        
        {loading && (
          <div style={{ 
            marginTop: 20,
            padding: "12px 20px",
            background: "rgba(99, 102, 241, 0.1)",
            borderRadius: "10px",
            border: "1px solid rgba(129, 140, 248, 0.2)",
          }}>
            <p style={{ margin: 0, color: "#818cf8", fontSize: "0.9em" }}>
              🔄 Processing...
            </p>
          </div>
        )}
        
        {!loading && output && (
          <div style={{ 
            marginTop: 20,
            padding: "20px 28px",
            background: "linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%)",
            borderRadius: "12px",
            border: "1px solid rgba(129, 140, 248, 0.3)",
            textAlign: "center",
            width: "100%",
            maxWidth: "280px",
          }}>
            {(() => {
              const exp = Array.from(output).map((v) => Math.exp(v));
              const sumExp = exp.reduce((a, b) => a + b, 0);
              const probs = exp.map((v) => v / sumExp);
              const predIdx = probs.indexOf(Math.max(...probs));
              const predProb = probs[predIdx];
              return (
                <>
                  <div style={{ fontSize: "3em", marginBottom: "8px", fontWeight: "bold" }}>
                    {predIdx}
                  </div>
                  <p style={{ 
                    fontSize: "1em", 
                    color: "#e4e4e7",
                    margin: 0,
                  }}>
                    Predicted: <b style={{ color: "#818cf8" }}>{predIdx}</b>
                  </p>
                  <p style={{ 
                    fontSize: "0.9em",
                    color: "#a1a1aa",
                    marginTop: "4px",
                    marginBottom: 0,
                  }}>
                    {(predProb * 100).toFixed(1)}% confidence
                  </p>
                </>
              );
            })()}
          </div>
        )}
        
        {!loading && !output && (
          <div style={{ 
            marginTop: 20,
            padding: "12px 20px",
            background: "rgba(161, 161, 170, 0.1)",
            borderRadius: "10px",
            border: "1px solid rgba(161, 161, 170, 0.2)",
          }}>
            <p style={{ margin: 0, color: "#a1a1aa", fontSize: "0.9em" }}>
              Draw a digit to see predictions
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
