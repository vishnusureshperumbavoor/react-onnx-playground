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

        // The postprocess function now needs the letterboxing info to scale boxes correctly
        const boxes = postprocess(
          results.output0.data as Float32Array,
          xRatio,
          yRatio,
          xPad,
          yPad
        );
        self.postMessage({ status: "complete", boxes });
        break;
    }
  } catch (e: any) {
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
}

// --- CORRECTED POSTPROCESS FUNCTION WITH LETTERBOXING ---
function postprocess(
  output: Float32Array,
  xRatio: number,
  yRatio: number,
  xPad: number,
  yPad: number
): Box[] {
  const boxes: Box[] = [];
  const numClasses = 80;
  const numBoxes = 8400;

  for (let i = 0; i < numBoxes; i++) {
    const classProbs = output.slice(
      i * (4 + numClasses) + 4,
      i * (4 + numClasses) + 4 + numClasses
    );
    const maxProb = Math.max(...classProbs);

    if (maxProb < 0.25) continue;

    const classIndex = classProbs.indexOf(maxProb);
    const x_center = output[i * (4 + numClasses) + 0];
    const y_center = output[i * (4 + numClasses) + 1];
    const w = output[i * (4 + numClasses) + 2];
    const h = output[i * (4 + numClasses) + 3];

    // Remove padding and scale back to original image dimensions
    const x1 = (x_center - w / 2 - xPad) * xRatio;
    const y1 = (y_center - h / 2 - yPad) * yRatio;
    const x2 = (x_center + w / 2 - xPad) * xRatio;
    const y2 = (y_center + h / 2 - yPad) * yRatio;

    boxes.push({
      classId: classIndex,
      probability: maxProb,
      box: [x1, y1, x2, y2],
    });
  }

  return nonMaxSuppression(boxes, 0.5);
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
