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
}

export function renderBoxes(canvas: HTMLCanvasElement, boxes: Box[]) {
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

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

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
