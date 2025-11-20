import { useEffect, useState, useRef } from "react";
import * as ort from "onnxruntime-web";
import { fetchAndCacheModel } from "../utils/fetchAndCacheModel";

// --- Configuration ---
// We use Xenova's quantized models because they are single-file and web-optimized.
// The 'sam3' repo models are multi-file and will crash without complex VFS handling.
const models = {
  encoder: {
    name: "sam-b-encoder-quant",
    url: "https://huggingface.co/Xenova/sam-vit-base/resolve/main/onnx/vision_encoder_quantized.onnx",
    size: 95, // ~95MB
    key: "encoder",
  },
  decoder: {
    name: "sam-b-decoder-quant",
    url: "https://huggingface.co/Xenova/sam-vit-base/resolve/main/onnx/prompt_encoder_mask_decoder_quantized.onnx",
    size: 4, // ~4MB
    key: "decoder",
  },
};

const MODEL_INPUT_SIZE = 1024;

export const Sam3Component = () => {
  // --- Sessions ---
  const [encoderSession, setEncoderSession] =
    useState<ort.InferenceSession | null>(null);
  const [decoderSession, setDecoderSession] =
    useState<ort.InferenceSession | null>(null);

  // --- State ---
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
    <div style={{ padding: 20, fontFamily: "sans-serif" }}>
      <h1>Segment Anything (Web Optimized)</h1>

      {/* Status Panel */}
      <div style={{ marginBottom: 20 }}>
        <div>
          <strong>Encoder:</strong> {formatPercent(encoderProgress)}%
          <div
            style={{
              background: "#eee",
              height: 6,
              width: 200,
              borderRadius: 4,
            }}
          >
            <div
              style={{
                background: "#4caf50",
                width: `${encoderProgress}%`,
                height: "100%",
                borderRadius: 4,
                transition: "width 0.2s",
              }}
            />
          </div>
        </div>
        <div style={{ marginTop: 10 }}>
          <strong>Decoder:</strong> {formatPercent(decoderProgress)}%
          <div
            style={{
              background: "#eee",
              height: 6,
              width: 200,
              borderRadius: 4,
            }}
          >
            <div
              style={{
                background: "#2196f3",
                width: `${decoderProgress}%`,
                height: "100%",
                borderRadius: 4,
                transition: "width 0.2s",
              }}
            />
          </div>
        </div>
      </div>

      {/* Messages */}
      <div style={{ minHeight: 30 }}>
        {processingEncoder && (
          <span style={{ color: "#e65100" }}>
            Processing Image... (This runs once)
          </span>
        )}
        {!processingEncoder && imageEmbeddings && (
          <span style={{ color: "green" }}>
            <strong>Ready! Click the image to segment.</strong>
          </span>
        )}
      </div>

      {/* Image Container */}
      <div
        style={{ position: "relative", display: "inline-block", marginTop: 10 }}
      >
        <img
          ref={imageRef}
          src={imageUrl}
          alt="Test"
          crossOrigin="anonymous"
          style={{
            maxWidth: 500,
            display: "block",
            border: "1px solid #ccc",
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
            pointerEvents: "none", // Let clicks pass through to image
            width: "100%",
            height: "100%",
          }}
        />
      </div>
    </div>
  );
};
