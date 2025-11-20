import { useNavigate, Routes, Route } from "react-router-dom";
import { Mnist, Yolo, Sam2Component } from "./components";
import { Sam3Component } from "./components/Sam3Component";

function Home() {
  const navigate = useNavigate();

  const cards = [
    {
      title: "MNIST",
      description: "Handwritten digit recognition",
      route: "/mnist",
    },
    {
      title: "YOLOv11",
      description: "Object detection",
      route: "/yolo",
    },
    {
      title: "SAM2",
      description: "Segment Anything Model v2",
      route: "/sam2",
    },
    {
      title: "SAM3",
      description: "Segment Anything Model v3",
      route: "/sam3",
    },
  ];

  return (
    <div style={{ padding: 20 }}>
      <h1>ONNX Runtime Web Playground</h1>
      <h2>Select a model to test:</h2>
      <div
        style={{
          display: "flex",
          gap: 32,
          justifyContent: "center",
          marginTop: 32,
        }}
      >
        {cards.map((card) => (
          <div
            key={card.title}
            style={{
              border: "1px solid #ccc",
              borderRadius: 12,
              padding: 24,
              width: 220,
              boxShadow: "0 2px 8px rgba(0,0,0,0.05)",
              cursor: "pointer",
              textAlign: "center",
              transition: "box-shadow 0.2s",
            }}
            onClick={() => navigate(card.route)}
            tabIndex={0}
            role="button"
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") navigate(card.route);
            }}
          >
            <h3>{card.title}</h3>
            <p>{card.description}</p>
            <button
              style={{
                marginTop: 12,
                padding: "8px 16px",
                borderRadius: 6,
                border: "none",
                background: "#0078d4",
                color: "#fff",
                fontWeight: "bold",
                cursor: "pointer",
              }}
              onClick={(e) => {
                e.stopPropagation();
                navigate(card.route);
              }}
            >
              Test {card.title}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/mnist" element={<Mnist />} />
      <Route path="/yolo" element={<Yolo />} />
      <Route path="/sam2" element={<Sam2Component />} />
      <Route path="/sam3" element={<Sam3Component />} />
    </Routes>
  );
}

export default App;
