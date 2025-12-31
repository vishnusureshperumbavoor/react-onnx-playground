// COCO class labels and a color palette
const Cocolabels = [
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "airplane",
  "bus",
  "train",
  "truck",
  "boat",
  "traffic light",
  "fire hydrant",
  "stop sign",
  "parking meter",
  "bench",
  "bird",
  "cat",
  "dog",
  "horse",
  "sheep",
  "cow",
  "elephant",
  "bear",
  "zebra",
  "giraffe",
  "backpack",
  "umbrella",
  "handbag",
  "tie",
  "suitcase",
  "frisbee",
  "skis",
  "snowboard",
  "sports ball",
  "kite",
  "baseball bat",
  "baseball glove",
  "skateboard",
  "surfboard",
  "tennis racket",
  "bottle",
  "wine glass",
  "cup",
  "fork",
  "knife",
  "spoon",
  "bowl",
  "banana",
  "apple",
  "sandwich",
  "orange",
  "broccoli",
  "carrot",
  "hot dog",
  "pizza",
  "donut",
  "cake",
  "chair",
  "couch",
  "potted plant",
  "bed",
  "dining table",
  "toilet",
  "tv",
  "laptop",
  "mouse",
  "remote",
  "keyboard",
  "cell phone",
  "microwave",
  "oven",
  "toaster",
  "sink",
  "refrigerator",
  "book",
  "clock",
  "vase",
  "scissors",
  "teddy bear",
  "hair drier",
  "toothbrush",
];

const colors = [
  "#FF3838",
  "#FF9D97",
  "#FF701F",
  "#FFB21D",
  "#CFD231",
  "#48F90A",
  "#92CC17",
  "#3DDB86",
  "#1A9334",
  "#00D4BB",
  "#2C99A8",
  "#00C2FF",
  "#344593",
  "#6473FF",
  "#0018EC",
  "#8438FF",
  "#520085",
  "#CB38FF",
  "#FF95C8",
  "#FF37C7",
];

interface Box {
  classId: number;
  probability: number;
  box: [number, number, number, number];
  mask?: Float32Array;
}

export function renderBoxes(
  canvas: HTMLCanvasElement,
  boxes: Box[],
  sourceElement?: HTMLImageElement | HTMLVideoElement
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  const font = "16px sans-serif";
  ctx.font = font;
  ctx.textBaseline = "top";

  boxes.forEach((box) => {
    const [x1, y1, x2, y2] = box.box;
    const label = Cocolabels[box.classId];
    const color = colors[box.classId % colors.length];

    // Render segmentation mask if available
    if (box.mask && sourceElement) {
      renderMask(ctx, box, color, sourceElement);
    } else {
      // Fallback to bounding box if no mask
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }

    // Draw label
    ctx.fillStyle = color;
    const textWidth = ctx.measureText(
      `${label} ${(box.probability * 100).toFixed(1)}%`
    ).width;
    const textHeight = parseInt(font, 10);
    ctx.fillRect(x1 - 1, y1 - (textHeight + 2), textWidth + 4, textHeight + 4);

    ctx.fillStyle = "#ffffff";
    ctx.fillText(
      `${label} ${(box.probability * 100).toFixed(1)}%`,
      x1 + 1,
      y1 - textHeight
    );
  });
}

function renderMask(
  ctx: CanvasRenderingContext2D,
  box: Box,
  color: string,
  _sourceElement: HTMLImageElement | HTMLVideoElement
) {
  if (!box.mask || box.mask.length === 0) return;

  const [x1, y1, x2, y2] = box.box;
  const boxWidth = Math.max(1, Math.floor(x2 - x1));
  const boxHeight = Math.max(1, Math.floor(y2 - y1));

  // Calculate mask dimensions (mask is already resized to box size in worker)
  const maskSize = Math.sqrt(box.mask.length);
  const maskWidth = Math.floor(maskSize);
  const maskHeight = Math.floor(maskSize);

  // Create a temporary canvas for the mask
  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = boxWidth;
  maskCanvas.height = boxHeight;
  const maskCtx = maskCanvas.getContext("2d");
  if (!maskCtx) return;

  // Convert mask data to image data
  const maskData = maskCtx.createImageData(boxWidth, boxHeight);
  const maskArray = box.mask;

  // Parse color
  const r = parseInt(color.slice(1, 3), 16);
  const g = parseInt(color.slice(3, 5), 16);
  const b = parseInt(color.slice(5, 7), 16);

  // Apply mask with color overlay
  for (let y = 0; y < boxHeight; y++) {
    for (let x = 0; x < boxWidth; x++) {
      // Sample from mask (which is already resized to box dimensions)
      const maskX = Math.floor((x / boxWidth) * maskWidth);
      const maskY = Math.floor((y / boxHeight) * maskHeight);
      const maskIndex = Math.min(maskY * maskWidth + maskX, maskArray.length - 1);
      const maskValue = Math.max(0, Math.min(1, maskArray[maskIndex]));

      const pixelIndex = (y * boxWidth + x) * 4;
      maskData.data[pixelIndex] = r; // R
      maskData.data[pixelIndex + 1] = g; // G
      maskData.data[pixelIndex + 2] = b; // B
      maskData.data[pixelIndex + 3] = Math.floor(maskValue * 180); // Alpha (semi-transparent)
    }
  }

  maskCtx.putImageData(maskData, 0, 0);

  // Draw the mask on the main canvas
  ctx.save();
  ctx.globalAlpha = 0.5;
  ctx.drawImage(maskCanvas, x1, y1, boxWidth, boxHeight);
  ctx.restore();

  // Draw border
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.strokeRect(x1, y1, boxWidth, boxHeight);
}
