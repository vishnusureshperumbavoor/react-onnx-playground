import * as ort from "onnxruntime-web";

let session: ort.InferenceSession | null = null;

self.onmessage = async (event: MessageEvent) => {
  const { type, modelBlob, imageBitmap } = event.data;

  try {
    switch (type) {
      case "init":
        if (!session) {
          const modelBuffer = await modelBlob.arrayBuffer();
          session = await ort.InferenceSession.create(modelBuffer, {
            executionProviders: ["wasm"],
          });
        }
        console.log("[Worker] Segmentation model loaded successfully");
        self.postMessage({ status: "model-loaded" });
        break;

      case "predict":
        if (!session) {
          self.postMessage({
            status: "error",
            error: "Session not initialized.",
          });
          return;
        }
        
        const modelWidth = 640;
        const modelHeight = 640;

        // The preprocess function now returns the tensor AND the letterboxing info
        const [tensor, xRatio, yRatio, xPad, yPad] = preprocess(
          imageBitmap,
          modelWidth,
          modelHeight
        );

        const feeds = { images: tensor };
        const results = await session.run(feeds);

        // Get the output tensors - YOLOv11-seg output is typically:
        // output0: [1, 116, 8400] - boxes + mask coefficients (116 = 4 box + 80 classes + 32 mask coeffs)
        // output1: [1, 32, 160, 160] - prototype masks
        // Try different possible output names
        const outputTensor = results.output0 || results.output || results[Object.keys(results)[0]];
        const maskTensor = results.output1 || results[Object.keys(results)[1]] || null;
        
        // The postprocess function now needs the letterboxing info to scale boxes correctly
        const { boxes, masks } = postprocess(
          outputTensor.data as Float32Array,
          outputTensor.dims,
          maskTensor ? (maskTensor.data as Float32Array) : null,
          maskTensor ? maskTensor.dims : null,
          xRatio,
          yRatio,
          xPad,
          yPad,
          imageBitmap.width,
          imageBitmap.height
        );
        self.postMessage({ status: "complete", boxes, masks });
        break;
      
      default:
        console.warn("[Worker] Unknown message type:", type);
    }
  } catch (e: any) {
    console.error("[Worker] Error:", e);
    self.postMessage({ status: "error", error: e.message });
  }
};

// --- CORRECTED PREPROCESS FUNCTION WITH LETTERBOXING ---
function preprocess(
  imageBitmap: ImageBitmap,
  modelWidth: number,
  modelHeight: number
): [ort.Tensor, number, number, number, number] {
  const canvas = new OffscreenCanvas(modelWidth, modelHeight);
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not get 2D context");

  // 1. Calculate aspect ratios and new dimensions
  const aspectRatio = imageBitmap.width / imageBitmap.height;
  let newWidth, newHeight;
  if (aspectRatio > 1) {
    // Landscape
    newWidth = modelWidth;
    newHeight = modelWidth / aspectRatio;
  } else {
    // Portrait or square
    newHeight = modelHeight;
    newWidth = modelHeight * aspectRatio;
  }

  // 2. Calculate padding
  const xPad = (modelWidth - newWidth) / 2;
  const yPad = (modelHeight - newHeight) / 2;

  // 3. Fill canvas with a neutral color (gray)
  ctx.fillStyle = "#808080"; // Gray padding
  ctx.fillRect(0, 0, modelWidth, modelHeight);

  // 4. Draw the resized image onto the padded canvas
  ctx.drawImage(imageBitmap, xPad, yPad, newWidth, newHeight);

  // 5. Get pixel data and transpose
  const imageData = ctx.getImageData(0, 0, modelWidth, modelHeight);
  const data = imageData.data;
  const red: number[] = [],
    green: number[] = [],
    blue: number[] = [];
  for (let i = 0; i < data.length; i += 4) {
    red.push(data[i] / 255.0);
    green.push(data[i + 1] / 255.0);
    blue.push(data[i + 2] / 255.0);
  }
  const transposedData = [...red, ...green, ...blue];

  const tensor = new ort.Tensor("float32", new Float32Array(transposedData), [
    1,
    3,
    modelHeight,
    modelWidth,
  ]);

  // 6. Return the tensor and letterboxing information for postprocessing
  const xRatio = imageBitmap.width / newWidth;
  const yRatio = imageBitmap.height / newHeight;
  return [tensor, xRatio, yRatio, xPad, yPad];
}

interface Box {
  classId: number;
  probability: number;
  box: [number, number, number, number];
  maskCoeffs?: Float32Array; // Mask coefficients for this instance
}

// --- CORRECTED POSTPROCESS FUNCTION WITH LETTERBOXING AND SEGMENTATION ---
function postprocess(
  output: Float32Array,
  dims: readonly number[],
  maskProtos: Float32Array | null,
  maskDims: readonly number[] | null,
  xRatio: number,
  yRatio: number,
  xPad: number,
  yPad: number,
  originalWidth: number,
  originalHeight: number
): { boxes: Box[]; masks: Float32Array[] | null } {
  const boxes: Box[] = [];
  const numClasses = 80;
  const numMaskCoeffs = 32; // Number of mask coefficients in YOLO segmentation models
  
  // YOLOv11-seg output is typically [1, 116, 8400] which means [batch, features, boxes]
  // Features are: [x, y, w, h, class1, class2, ..., class80, mask_coeff1, ..., mask_coeff32]
  const isTransposed = dims[1] >= 84 || dims[1] < dims[2];
  const numBoxes = isTransposed ? dims[2] : dims[1];
  const numFeatures = isTransposed ? dims[1] : dims[2];
  // Check if this is a segmentation model:
  // 1. Has 116+ features (84 for detection + 32 mask coefficients)
  // 2. Has mask prototype tensors available
  const isSegmentation = numFeatures >= 116 && maskProtos !== null && maskDims !== null;

  for (let i = 0; i < numBoxes; i++) {
    let x_center, y_center, w, h;
    let classProbs: number[] = [];
    let maskCoeffs: Float32Array | undefined;

    if (isTransposed) {
      // Output format: [1, 116, 8400] - features are in rows, boxes in columns
      x_center = output[i];
      y_center = output[numBoxes + i];
      w = output[2 * numBoxes + i];
      h = output[3 * numBoxes + i];
      
      // Get class probabilities
      for (let j = 0; j < numClasses; j++) {
        classProbs.push(output[(4 + j) * numBoxes + i]);
      }
      
      // Get mask coefficients if this is a segmentation model
      if (isSegmentation) {
        const coeffs = new Float32Array(numMaskCoeffs);
        for (let j = 0; j < numMaskCoeffs; j++) {
          coeffs[j] = output[(4 + numClasses + j) * numBoxes + i];
        }
        maskCoeffs = coeffs;
      }
    } else {
      // Output format: [1, 8400, 116] - boxes are in rows, features in columns
      const offset = i * numFeatures;
      x_center = output[offset + 0];
      y_center = output[offset + 1];
      w = output[offset + 2];
      h = output[offset + 3];
      
      for (let j = 0; j < numClasses; j++) {
        classProbs.push(output[offset + 4 + j]);
      }
      
      // Get mask coefficients if this is a segmentation model
      if (isSegmentation) {
        const coeffs = new Float32Array(numMaskCoeffs);
        for (let j = 0; j < numMaskCoeffs; j++) {
          coeffs[j] = output[offset + 4 + numClasses + j];
        }
        maskCoeffs = coeffs;
      }
    }

    const maxProb = Math.max(...classProbs);
    
    // Lower threshold to detect more objects
    if (maxProb < 0.25) continue;

    const classIndex = classProbs.indexOf(maxProb);

    // Remove padding and scale back to original image dimensions
    const x1 = (x_center - w / 2 - xPad) * xRatio;
    const y1 = (y_center - h / 2 - yPad) * yRatio;
    const x2 = (x_center + w / 2 - xPad) * xRatio;
    const y2 = (y_center + h / 2 - yPad) * yRatio;

    boxes.push({
      classId: classIndex,
      probability: maxProb,
      box: [x1, y1, x2, y2],
      maskCoeffs,
    });
  }

  const filtered = nonMaxSuppression(boxes, 0.45);
  
  // Process masks if available
  let masks: Float32Array[] | null = null;
  if (maskProtos && maskDims && filtered.length > 0) {
    masks = processMasks(filtered, maskProtos, maskDims, originalWidth, originalHeight);
  }
  
  return { boxes: filtered, masks };
}

function processMasks(
  boxes: Box[],
  maskProtos: Float32Array,
  maskDims: readonly number[],
  _originalWidth: number,
  _originalHeight: number
): Float32Array[] {
  // maskProtos shape is typically [1, 32, 160, 160]
  const [, channels, protoHeight, protoWidth] = maskDims;
  const masks: Float32Array[] = [];

  for (const box of boxes) {
    if (!box.maskCoeffs) {
      masks.push(new Float32Array(0));
      continue;
    }

    const [x1, y1, x2, y2] = box.box;
    const boxWidth = Math.max(1, x2 - x1);
    const boxHeight = Math.max(1, y2 - y1);

    // Compute mask by multiplying prototype masks with coefficients
    const mask = new Float32Array(protoWidth * protoHeight);
    
    for (let y = 0; y < protoHeight; y++) {
      for (let x = 0; x < protoWidth; x++) {
        let sum = 0;
        for (let c = 0; c < channels; c++) {
          const protoIdx = c * protoHeight * protoWidth + y * protoWidth + x;
          sum += maskProtos[protoIdx] * box.maskCoeffs![c];
        }
        mask[y * protoWidth + x] = sum;
      }
    }

    // Apply sigmoid activation
    for (let i = 0; i < mask.length; i++) {
      mask[i] = 1 / (1 + Math.exp(-mask[i]));
    }

    // Resize mask to box dimensions
    const resizedMask = resizeMask(mask, protoWidth, protoHeight, boxWidth, boxHeight);
    masks.push(resizedMask);
  }

  return masks;
}

function resizeMask(
  mask: Float32Array,
  srcWidth: number,
  srcHeight: number,
  dstWidth: number,
  dstHeight: number
): Float32Array {
  const resized = new Float32Array(Math.floor(dstWidth) * Math.floor(dstHeight));
  const scaleX = srcWidth / dstWidth;
  const scaleY = srcHeight / dstHeight;

  for (let y = 0; y < dstHeight; y++) {
    for (let x = 0; x < dstWidth; x++) {
      const srcX = Math.floor(x * scaleX);
      const srcY = Math.floor(y * scaleY);
      const srcIdx = Math.min(srcY * srcWidth + srcX, mask.length - 1);
      const dstIdx = Math.floor(y) * Math.floor(dstWidth) + Math.floor(x);
      if (dstIdx < resized.length) {
        resized[dstIdx] = mask[srcIdx];
      }
    }
  }

  return resized;
}

function nonMaxSuppression(boxes: Box[], iouThreshold: number): Box[] {
  const sortedBoxes = boxes.sort((a, b) => b.probability - a.probability);
  const selectedBoxes: Box[] = [];

  while (sortedBoxes.length > 0) {
    const primaryBox = sortedBoxes.shift();
    if (primaryBox) {
      selectedBoxes.push(primaryBox);
      for (let i = sortedBoxes.length - 1; i >= 0; i--) {
        const iou = calculateIoU(primaryBox.box, sortedBoxes[i].box);
        if (iou > iouThreshold) {
          sortedBoxes.splice(i, 1);
        }
      }
    }
  }
  return selectedBoxes;
}

function calculateIoU(box1: number[], box2: number[]): number {
  const [x1, y1, x2, y2] = box1;
  const [x3, y3, x4, y4] = box2;
  const interX1 = Math.max(x1, x3);
  const interY1 = Math.max(y1, y3);
  const interX2 = Math.min(x2, x4);
  const interY2 = Math.min(y2, y4);
  const interArea =
    Math.max(0, interX2 - interX1) * Math.max(0, interY2 - interY1);
  const box1Area = (x2 - x1) * (y2 - y1);
  const box2Area = (x4 - x3) * (y4 - y3);
  const unionArea = box1Area + box2Area - interArea;
  return interArea / unionArea;
}

