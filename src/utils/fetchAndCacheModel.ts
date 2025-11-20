import { get, set } from "idb-keyval";
import type { ModelInfo } from "../types/common";

export async function fetchAndCacheModel(
  model: ModelInfo,
  onProgress?: (percent: number) => void
): Promise<ArrayBuffer> {
  const cacheKey = `sam2-model-${model.key}`;
  const cached = await get<ArrayBuffer>(cacheKey);
  if (cached) {
    if (onProgress) onProgress(100);
    return cached;
  } else {
    const response = await fetch(model.url);
    const contentLength = response.headers.get("content-length");
    if (!response.body || !contentLength) {
      const buffer = await response.arrayBuffer();
      if (onProgress) onProgress(100);
      await set(cacheKey, buffer);
      return buffer;
    }
    const total = parseInt(contentLength, 10);
    let loaded = 0;
    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        chunks.push(value);
        loaded += value.length;
        if (onProgress) onProgress(Math.round((loaded / total) * 100));
      }
    }
    const buffer = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
      buffer.set(chunk, offset);
      offset += chunk.length;
    }
    await set(cacheKey, buffer.buffer);
    if (onProgress) onProgress(100);
    return buffer.buffer;
  }
}
