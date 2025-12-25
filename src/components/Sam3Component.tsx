import { useEffect, useState, useRef } from "react";
import * as ort from "onnxruntime-web";
import { fetchAndCacheModel } from "../utils/fetchAndCacheModel";

const models = {
  encoder: {
    name: "sam-b-encoder-quant",
    url: "https://huggingface.co/Xenova/sam-vit-base/resolve/main/onnx/vision_encoder_quantized.onnx",
    size: 95,
    key: "encoder",
  },
  decoder: {
    name: "sam-b-decoder-quant",
    url: "https://huggingface.co/Xenova/sam-vit-base/resolve/main/onnx/prompt_encoder_mask_decoder_quantized.onnx",
    size: 4,
    key: "decoder",
  },
};

const MODEL_INPUT_SIZE = 1024;

export const Sam3Component = () => {
  const [encoderSession, setEncoderSession] =
    useState<ort.InferenceSession | null>(null);
  const [decoderSession, setDecoderSession] =
    useState<ort.InferenceSession | null>(null);

  const [loading, setLoading] = useState(false);
  const [processingEncoder, setProcessingEncoder] = useState(false);
  const [encoderProgress, setEncoderProgress] = useState(0);
  const [decoderProgress, setDecoderProgress] = useState(0);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageUrl] = useState<string>(
    "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg"
  );

  // --- Tensors & Data ---
  const [imageEmbeddings, setImageEmbeddings] = useState<ort.Tensor | null>(
    null
  );
  const [imagePositionalEmbeddings, setImagePositionalEmbeddings] =
    useState<ort.Tensor | null>(null);
  const [imageTensor, setImageTensor] = useState<ort.Tensor | null>(null);
  const [modelScale, setModelScale] = useState<{ x: number; y: number } | null>(
    null
  );

  // --- Refs ---
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // 1. Load Models
  useEffect(() => {
    async function loadSamModels() {
      setLoading(true);
      try {
        // Enable WebAssembly multi-threading if available
        ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;

        const encoderBuffer = await fetchAndCacheModel(
          models.encoder,
          setEncoderProgress
        );
        const decoderBuffer = await fetchAndCacheModel(
          models.decoder,
          setDecoderProgress
        );

        const encoder = await ort.InferenceSession.create(encoderBuffer);
        setEncoderSession(encoder);

        const decoder = await ort.InferenceSession.create(decoderBuffer);
        setDecoderSession(decoder);
      } catch (e) {
        console.error("Failed to load SAM models:", e);
      } finally {
        setLoading(false);
      }
    }
    loadSamModels();
  }, []);

  // 2. Pre-process Image (Resize -> Normalize -> Tensor)
  const processImage = () => {
    if (!imageRef.current) return;
    const img = imageRef.current;

    // Calculate scale factors to map clicks later
    const scaleX = MODEL_INPUT_SIZE / img.naturalWidth;
    const scaleY = MODEL_INPUT_SIZE / img.naturalHeight;
    setModelScale({ x: scaleX, y: scaleY });

    // Draw image to 1024x1024 canvas
    const canvas = document.createElement("canvas");
    canvas.width = MODEL_INPUT_SIZE;
    canvas.height = MODEL_INPUT_SIZE;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(img, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
    const imageData = ctx.getImageData(
      0,
      0,
      MODEL_INPUT_SIZE,
      MODEL_INPUT_SIZE
    );
    const { data } = imageData;

    // Create Float32Array for the tensor (NCHW format)
    const input = new Float32Array(3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE);

    // Normalize (Standard ImageNet statistics)
    // Mean: [123.675, 116.28, 103.53], Std: [58.395, 57.12, 57.375]
    for (let i = 0; i < MODEL_INPUT_SIZE * MODEL_INPUT_SIZE; i++) {
      const r = data[i * 4];
      const g = data[i * 4 + 1];
      const b = data[i * 4 + 2];

      input[i] = (r - 123.675) / 58.395; // R channel
      input[i + MODEL_INPUT_SIZE * MODEL_INPUT_SIZE] = (g - 116.28) / 57.12; // G channel
      input[i + 2 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE] =
        (b - 103.53) / 57.375; // B channel
    }

    const tensor = new ort.Tensor("float32", input, [
      1,
      3,
      MODEL_INPUT_SIZE,
      MODEL_INPUT_SIZE,
    ]);
    setImageTensor(tensor);
  };

  const handleImageLoad = () => {
    setImageLoaded(true);
    processImage();
  };

  // 3. Run Encoder (Once per image)
  useEffect(() => {
    if (
      encoderSession &&
      imageTensor &&
      !imageEmbeddings &&
      !processingEncoder
    ) {
      runEncoder();
    }
  }, [encoderSession, imageTensor, imageEmbeddings, processingEncoder]);

  const runEncoder = async () => {
    if (!encoderSession || !imageTensor) return;
    setProcessingEncoder(true);

    try {
      // For Xenova/standard SAM models, input name is usually 'pixel_values'
      const feeds = { pixel_values: imageTensor };
      console.log("vsp encoder feeds", feeds);
      const results = await encoderSession.run(feeds);
      const embeddings = results["image_embeddings"];
      const positionalEmbeddings = results["image_positional_embeddings"];
      setImageEmbeddings(embeddings);
      setImagePositionalEmbeddings(positionalEmbeddings);
      console.log("Encoder run complete. Embeddings ready.");
    } catch (e) {
      console.error("Encoder failed:", e);
    } finally {
      setProcessingEncoder(false);
    }
  };

  // 4. Run Decoder (Interactive - on click)
  const handleSegmentClick = async (e: React.MouseEvent<HTMLImageElement>) => {
    if (
      !decoderSession ||
      !imageEmbeddings ||
      !modelScale ||
      !imageRef.current ||
      !imagePositionalEmbeddings
    )
      return;

    // 1. Get click coordinates
    const rect = e.currentTarget.getBoundingClientRect();
    // Calculate x,y relative to natural image size
    const x =
      (e.clientX - rect.left) * (e.currentTarget.naturalWidth / rect.width);
    const y =
      (e.clientY - rect.top) * (e.currentTarget.naturalHeight / rect.height);

    // 2. Map to Model Coordinates (1024x1024)
    const pointX = x * modelScale.x;
    const pointY = y * modelScale.y;

    try {
      // 3. Prepare Inputs for SAM Decoder

      // Points: shape (1, 1, 2)
      const pointCoords = new ort.Tensor(
        "float32",
        new Float32Array([pointX, pointY]),
        [1, 1, 1, 2]
      );

      // Labels: shape (1, 1, 1)
      const pointLabels = new ort.Tensor(
        "int64",
        BigInt64Array.from([1n]),
        [1, 1, 1]
      );

      const feeds = {
        input_points: pointCoords,
        input_labels: pointLabels,
        image_embeddings: imageEmbeddings,
        image_positional_embeddings: imagePositionalEmbeddings,
      };
      console.log("vsp decoder feeds", feeds);
      console.log("vsp Session Input Names:", decoderSession.inputNames);

      // 4. Run Inference
      const results = await decoderSession.run(feeds);
      console.log("Decoder run complete. Results:", results);

      // 5. Draw Result
      const masks = results["pred_masks"]; // Output shape usually (1, 1, H, W)
      console.log("Decoder masks:", masks);
      if (masks) {
        drawMask(masks.data as Float32Array);
      }
    } catch (e) {
      console.error("Decoder failed:", e);
    }
  };

  // 5. Visualization
  const drawMask = (maskData: Float32Array) => {
    if (!canvasRef.current || !imageRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Match canvas to displayed image size
    canvas.width = imageRef.current.width;
    canvas.height = imageRef.current.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Use an offscreen canvas to render the raw 1024x1024 mask
    const maskCanvas = document.createElement("canvas");
    maskCanvas.width = MODEL_INPUT_SIZE;
    maskCanvas.height = MODEL_INPUT_SIZE;
    const maskCtx = maskCanvas.getContext("2d");
    if (!maskCtx) return;

    const imgData = maskCtx.createImageData(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);

    // maskData is logits. Positive > 0 is foreground.
    // We only render the first mask (if multiple returned, they are usually stacked).
    const pixelCount = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE;

    for (let i = 0; i < pixelCount; i++) {
      if (maskData[i] > 0.0) {
        imgData.data[i * 4] = 0; // R
        imgData.data[i * 4 + 1] = 114; // G
        imgData.data[i * 4 + 2] = 255; // B
        imgData.data[i * 4 + 3] = 160; // Alpha
      }
    }
    maskCtx.putImageData(imgData, 0, 0);

    // Draw the 1024x1024 mask onto the display canvas (browser handles scaling)
    ctx.drawImage(maskCanvas, 0, 0, canvas.width, canvas.height);
  };

  function formatPercent(percent: number) {
    return percent % 1 === 0 ? percent : percent.toFixed(2);
  }

  return (
    <div style={{ padding: "40px 20px", maxWidth: "1000px", margin: "0 auto" }}>
      <div style={{ textAlign: "center", marginBottom: "40px" }}>
        <h1>SAM3 Interactive Segmentation</h1>
        <p style={{ fontSize: "1.1em", color: "#a1a1aa" }}>
          Segment Anything Model v3 - Click to segment objects interactively
        </p>
      </div>

      <div style={{ 
        background: "rgba(30, 30, 46, 0.6)",
        backdropFilter: "blur(10px)",
        border: "1px solid rgba(129, 140, 248, 0.2)",
        borderRadius: "20px",
        padding: "32px",
        boxShadow: "0 8px 32px rgba(0, 0, 0, 0.3)",
      }}>
        {/* Status Panel */}
        <div style={{ marginBottom: 30 }}>
          <div style={{ marginBottom: 20 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <span style={{ color: "#e4e4e7", fontWeight: 600 }}>Encoder Model</span>
              <span style={{ color: "#818cf8", fontWeight: 600 }}>{formatPercent(encoderProgress)}%</span>
            </div>
            <div
              style={{
                background: "rgba(30, 30, 46, 0.8)",
                height: 8,
                borderRadius: 8,
                overflow: "hidden",
                border: "1px solid rgba(129, 140, 248, 0.2)",
              }}
            >
              <div
                style={{
                  background: "linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%)",
                  width: `${encoderProgress}%`,
                  height: "100%",
                  transition: "width 0.3s",
                  boxShadow: encoderProgress > 0 ? "0 0 10px rgba(99, 102, 241, 0.5)" : "none",
                }}
              />
            </div>
          </div>
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <span style={{ color: "#e4e4e7", fontWeight: 600 }}>Decoder Model</span>
              <span style={{ color: "#818cf8", fontWeight: 600 }}>{formatPercent(decoderProgress)}%</span>
            </div>
            <div
              style={{
                background: "rgba(30, 30, 46, 0.8)",
                height: 8,
                borderRadius: 8,
                overflow: "hidden",
                border: "1px solid rgba(129, 140, 248, 0.2)",
              }}
            >
              <div
                style={{
                  background: "linear-gradient(90deg, #06b6d4 0%, #3b82f6 100%)",
                  width: `${decoderProgress}%`,
                  height: "100%",
                  transition: "width 0.3s",
                  boxShadow: decoderProgress > 0 ? "0 0 10px rgba(59, 130, 246, 0.5)" : "none",
                }}
              />
            </div>
          </div>
        </div>

        {/* Messages */}
        <div style={{ minHeight: 50, marginBottom: 20 }}>
          {processingEncoder && (
            <div style={{ 
              padding: "16px 24px",
              background: "rgba(251, 146, 60, 0.1)",
              borderRadius: "12px",
              border: "1px solid rgba(251, 146, 60, 0.3)",
              textAlign: "center",
            }}>
              <p style={{ margin: 0, color: "#fb923c", fontSize: "1.1em" }}>
                ⚡ Processing image embeddings... (One-time process)
              </p>
            </div>
          )}
          {!processingEncoder && imageEmbeddings && (
            <div style={{ 
              padding: "16px 24px",
              background: "linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(16, 185, 129, 0.15) 100%)",
              borderRadius: "12px",
              border: "1px solid rgba(34, 197, 94, 0.3)",
              textAlign: "center",
            }}>
              <p style={{ margin: 0, color: "#4ade80", fontSize: "1.1em" }}>
                🎯 <strong>Ready!</strong> Click anywhere on the image to segment objects
              </p>
            </div>
          )}
        </div>

        {/* Image Container */}
        <div
          style={{ 
            position: "relative", 
            display: "inline-block",
            borderRadius: "12px",
            overflow: "hidden",
            boxShadow: "0 8px 20px rgba(99, 102, 241, 0.2)",
          }}
        >
          <img
            ref={imageRef}
            src={imageUrl}
            alt="Test"
            crossOrigin="anonymous"
            style={{
              maxWidth: "100%",
              width: 600,
              display: "block",
              cursor: imageEmbeddings ? "crosshair" : "wait",
            }}
            onLoad={handleImageLoad}
            onClick={handleSegmentClick}
          />
          <canvas
            ref={canvasRef}
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              pointerEvents: "none",
              width: "100%",
              height: "100%",
            }}
          />
        </div>
      </div>
    </div>
  );
};
