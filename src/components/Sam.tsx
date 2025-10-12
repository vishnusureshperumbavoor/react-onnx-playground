import React, { useEffect, useState } from "react";
import * as ort from "onnxruntime-web";
import { get, set } from "idb-keyval"; // Add this import

type ModelInfo = {
  name: string;
  url: string;
  size: number;
  key: string;
};

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

export const Sam = () => {
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
    <div style={{ padding: 20 }}>
      <h1>SAM2 ONNX Runtime Web</h1>
      {loading && <p>Loading SAM models...</p>}
      {!loading && encoderSession && decoderSession && (
        <p>Encoder and Decoder models loaded. Ready for inference.</p>
      )}
      {!loading && (!encoderSession || !decoderSession) && (
        <p>Failed to load one or both models.</p>
      )}
      <div style={{ marginTop: 20 }}>
        <img
          src={imageUrl}
          alt="Test"
          style={{ maxWidth: 400, border: "1px solid #ccc" }}
          onLoad={handleImageLoad}
        />
        <br />
        <button
          onClick={handleSegment}
          disabled={!encoderSession || !decoderSession || !imageLoaded}
          style={{ marginTop: 10 }}
        >
          Run Segmentation
        </button>
      </div>
    </div>
  );
};
