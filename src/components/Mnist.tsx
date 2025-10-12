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

  function clearCanvas() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  // Convert canvas to 28x28 grayscale Float32Array
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
      const imgData = getImageData(); // Float32Array of shape [1*1*28*28]
      const inputShape = [1, 1, 28, 28]; // batch, channel, height, width
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

  // Initialize canvas background
  useEffect(() => {
    clearCanvas();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.lineWidth = 16;
    ctx.lineCap = "round";
    ctx.strokeStyle = "black";
  }, []);

  return (
    <div style={{ padding: 20 }}>
      <h1>MNIST ONNX Runtime Web</h1>
      <div>
        <canvas
          ref={canvasRef}
          width={280}
          height={280}
          style={{
            border: "1px solid #ccc",
            background: "white",
            touchAction: "none",
          }}
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={endDrawing}
          onMouseLeave={endDrawing}
          onTouchStart={startDrawing}
          onTouchMove={draw}
          onTouchEnd={endDrawing}
        />
        <div style={{ marginTop: 10 }}>
          <button onClick={clearCanvas}>Clear</button>
        </div>
      </div>
      {loading && <p>Loading model or running inference...</p>}
      {!loading && output && (
        <>
          <h2>Model output (digit class probabilities):</h2>
          <p>
            {(() => {
              // Softmax calculation
              const exp = Array.from(output).map((v) => Math.exp(v));
              const sumExp = exp.reduce((a, b) => a + b, 0);
              const probs = exp.map((v) => v / sumExp);
              const predIdx = probs.indexOf(Math.max(...probs));
              const predProb = probs[predIdx];
              return (
                <>
                  Predicted digit: <b>{predIdx}</b> (
                  {(predProb * 100).toFixed(2)}% sure)
                </>
              );
            })()}
          </p>
        </>
      )}
      {!loading && !output && <p>No output from model.</p>}
    </div>
  );
}
