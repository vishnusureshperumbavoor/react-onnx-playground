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
      icon: "✍️",
      gradient: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    },
    {
      title: "YOLOv11",
      description: "Object detection",
      route: "/yolo",
      icon: "🎯",
      gradient: "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
    },
    {
      title: "SAM2",
      description: "Segment Anything Model v2",
      route: "/sam2",
      icon: "✂️",
      gradient: "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
    },
    {
      title: "SAM3",
      description: "Segment Anything Model v3",
      route: "/sam3",
      icon: "🔬",
      gradient: "linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)",
    },
  ];

  return (
    <div style={{ padding: "40px 20px", maxWidth: "1400px", margin: "0 auto" }}>
      <div style={{ textAlign: "center", marginBottom: "50px" }}>
        <h1 style={{ fontSize: "3.5em", marginBottom: "10px" }}>
          ONNX Runtime Web Playground
        </h1>
        <p style={{ fontSize: "1.2em", color: "#a1a1aa", marginTop: "10px" }}>
          Explore cutting-edge AI models running directly in your browser
        </p>
      </div>
      
      <div
        className="cards-grid"
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(4, 1fr)",
          gap: "24px",
          maxWidth: "1400px",
          margin: "0 auto",
        }}
      >
        {cards.map((card) => (
          <div
            key={card.title}
            style={{
              background: "rgba(30, 30, 46, 0.6)",
              backdropFilter: "blur(10px)",
              border: "1px solid rgba(129, 140, 248, 0.2)",
              borderRadius: "20px",
              padding: "32px",
              cursor: "pointer",
              textAlign: "center",
              transition: "all 0.3s ease",
              position: "relative",
              overflow: "hidden",
            }}
            onClick={() => navigate(card.route)}
            tabIndex={0}
            role="button"
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") navigate(card.route);
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = "translateY(-8px)";
              e.currentTarget.style.boxShadow = "0 20px 40px rgba(99, 102, 241, 0.3)";
              e.currentTarget.style.borderColor = "rgba(129, 140, 248, 0.5)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = "translateY(0)";
              e.currentTarget.style.boxShadow = "0 8px 20px rgba(0, 0, 0, 0.2)";
              e.currentTarget.style.borderColor = "rgba(129, 140, 248, 0.2)";
            }}
          >
            <div
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                right: 0,
                height: "4px",
                background: card.gradient,
              }}
            />
            <div
              style={{
                fontSize: "3em",
                marginBottom: "16px",
                filter: "drop-shadow(0 4px 8px rgba(99, 102, 241, 0.3))",
              }}
            >
              {card.icon}
            </div>
            <h3 style={{ fontSize: "1.5em", marginBottom: "12px" }}>
              {card.title}
            </h3>
            <p style={{ fontSize: "0.95em", lineHeight: "1.6", marginBottom: "24px" }}>
              {card.description}
            </p>
            <button
              style={{
                width: "100%",
                background: card.gradient,
                border: "none",
              }}
              onClick={(e) => {
                e.stopPropagation();
                navigate(card.route);
              }}
            >
              <span>Launch {card.title}</span>
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
