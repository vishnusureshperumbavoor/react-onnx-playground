import { useEffect, useState } from "react";
import * as ort from "onnxruntime-web";
import { get, set } from "idb-keyval";
import type { ModelInfo } from "../types/common";

const models: { sam_b: ModelInfo[] } = {
  sam_b: [
    {
      name: "sam-b-encoder",
      url: "https://huggingface.co/schmuell/sam-b-fp16/resolve/main/sam_vit_b_01ec64.encoder-fp16.onnx",
      size: 180,
      key: "encoder",
    },
    {
      name: "sam-b-decoder",
      url: "https://huggingface.co/schmuell/sam-b-fp16/resolve/main/sam_vit_b_01ec64.decoder.onnx",
      size: 17,
      key: "decoder",
    },
  ],
};

async function fetchAndCacheModel(model: ModelInfo): Promise<ArrayBuffer> {
  const cacheKey = `sam2-model-${model.key}`;
  const cached = await get<ArrayBuffer>(cacheKey);
  if (cached) {
    return cached;
  } else {
    const response = await fetch(model.url);
    const buffer = await response.arrayBuffer();
    await set(cacheKey, buffer);
    return buffer;
  }
}

export const Sam2Component = () => {
  const [encoderSession, setEncoderSession] =
    useState<ort.InferenceSession | null>(null);
  const [decoderSession, setDecoderSession] =
    useState<ort.InferenceSession | null>(null);
  const [loading, setLoading] = useState(false);
  const [imageUrl] = useState<string>(
    "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg"
  );
  const [imageLoaded, setImageLoaded] = useState(false);

  useEffect(() => {
    async function loadSamModels() {
      setLoading(true);
      try {
        const encoderBuffer = await fetchAndCacheModel(models.sam_b[0]);
        const encoder = await ort.InferenceSession.create(encoderBuffer);
        setEncoderSession(encoder);

        const decoderBuffer = await fetchAndCacheModel(models.sam_b[1]);
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

  const handleImageLoad = () => setImageLoaded(true);

  const handleSegment = () => {
    if (!encoderSession || !decoderSession || !imageLoaded) return;
    alert("Segmentation triggered! (Implement inference logic)");
  };

  return (
    <div style={{ padding: "40px 20px", maxWidth: "1000px", margin: "0 auto" }}>
      <div style={{ textAlign: "center", marginBottom: "40px" }}>
        <h1>SAM2 Segmentation</h1>
        <p style={{ fontSize: "1.1em", color: "#a1a1aa" }}>
          Segment Anything Model v2 - Advanced image segmentation
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
        {loading && (
          <div style={{ 
            padding: "24px",
            background: "rgba(99, 102, 241, 0.1)",
            borderRadius: "12px",
            border: "1px solid rgba(129, 140, 248, 0.2)",
            textAlign: "center",
          }}>
            <p style={{ margin: 0, color: "#818cf8", fontSize: "1.1em" }}>
              🔄 Loading SAM models... Please wait.
            </p>
          </div>
        )}
        
        {!loading && encoderSession && decoderSession && (
          <div style={{ 
            padding: "16px 24px",
            background: "linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(16, 185, 129, 0.15) 100%)",
            borderRadius: "12px",
            border: "1px solid rgba(34, 197, 94, 0.3)",
            textAlign: "center",
            marginBottom: "24px",
          }}>
            <p style={{ margin: 0, color: "#4ade80", fontSize: "1.1em" }}>
              ✅ Models loaded successfully! Ready for inference.
            </p>
          </div>
        )}
        
        {!loading && (!encoderSession || !decoderSession) && (
          <div style={{ 
            padding: "16px 24px",
            background: "rgba(239, 68, 68, 0.1)",
            borderRadius: "12px",
            border: "1px solid rgba(239, 68, 68, 0.3)",
            textAlign: "center",
            marginBottom: "24px",
          }}>
            <p style={{ margin: 0, color: "#f87171", fontSize: "1.1em" }}>
              ❌ Failed to load one or both models.
            </p>
          </div>
        )}
        
        <div style={{ 
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: "20px",
        }}>
          <div style={{
            borderRadius: "12px",
            overflow: "hidden",
            boxShadow: "0 8px 20px rgba(99, 102, 241, 0.2)",
          }}>
            <img
              src={imageUrl}
              alt="Test"
              style={{ 
                maxWidth: "100%",
                width: 500,
                display: "block",
              }}
              onLoad={handleImageLoad}
            />
          </div>
          
          <button
            onClick={handleSegment}
            disabled={!encoderSession || !decoderSession || !imageLoaded}
          >
            <span>✂️ Run Segmentation</span>
          </button>
        </div>
      </div>
    </div>
  );
};
