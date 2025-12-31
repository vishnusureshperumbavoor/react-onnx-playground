import React from "react";

export const YoloSam2 = () => {
  return (
    <div style={{ padding: "20px", maxWidth: "1000px", margin: "0 auto", height: "100%" }}>
      <div style={{ textAlign: "center", marginBottom: "30px" }}>
        <h2 style={{ fontSize: "2em", margin: "0", background: "linear-gradient(135deg, #f5576c 0%, #f093fb 50%, #4facfe 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>
          YOLOv11 + SAM2 🚀
        </h2>
        <p style={{ fontSize: "1.1em", color: "#a1a1aa", margin: "12px 0 0 0" }}>
          The Ultimate Object Detection & Segmentation Combo
        </p>
      </div>
      
      <div style={{ 
        background: "rgba(30, 30, 46, 0.6)",
        backdropFilter: "blur(10px)",
        border: "1px solid rgba(129, 140, 248, 0.2)",
        borderRadius: "16px",
        padding: "40px",
        boxShadow: "0 8px 32px rgba(0, 0, 0, 0.3)",
        textAlign: "center",
      }}>
        <div style={{
          fontSize: "6em",
          marginBottom: "20px",
          animation: "pulse 2s ease-in-out infinite",
        }}>
          🎯✨
        </div>
        
        <h3 style={{ 
          fontSize: "1.5em", 
          color: "#e4e4e7",
          marginBottom: "20px",
        }}>
          Coming Soon!
        </h3>
        
        <p style={{ 
          color: "#a1a1aa", 
          fontSize: "1.1em",
          lineHeight: "1.8",
          marginBottom: "30px",
        }}>
          Imagine the power of <span style={{ color: "#f5576c", fontWeight: "600" }}>YOLOv11</span> for lightning-fast object detection<br/>
          combined with <span style={{ color: "#4facfe", fontWeight: "600" }}>SAM2</span> for pixel-perfect segmentation! 🔥
        </p>

        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
          gap: "20px",
          marginTop: "40px",
        }}>
          <div style={{
            padding: "24px",
            background: "linear-gradient(135deg, rgba(245, 87, 108, 0.15) 0%, rgba(240, 147, 251, 0.15) 100%)",
            borderRadius: "12px",
            border: "1px solid rgba(245, 87, 108, 0.3)",
          }}>
            <div style={{ fontSize: "2.5em", marginBottom: "12px" }}>🎯</div>
            <h4 style={{ color: "#e4e4e7", margin: "0 0 8px 0", fontSize: "1.1em" }}>YOLOv11</h4>
            <p style={{ color: "#a1a1aa", margin: 0, fontSize: "0.9em" }}>
              Real-time object detection with state-of-the-art accuracy
            </p>
          </div>

          <div style={{
            padding: "24px",
            background: "linear-gradient(135deg, rgba(79, 172, 254, 0.15) 0%, rgba(0, 242, 254, 0.15) 100%)",
            borderRadius: "12px",
            border: "1px solid rgba(79, 172, 254, 0.3)",
          }}>
            <div style={{ fontSize: "2.5em", marginBottom: "12px" }}>✨</div>
            <h4 style={{ color: "#e4e4e7", margin: "0 0 8px 0", fontSize: "1.1em" }}>SAM2</h4>
            <p style={{ color: "#a1a1aa", margin: 0, fontSize: "0.9em" }}>
              Segment Anything Model 2 for precise instance segmentation
            </p>
          </div>
        </div>

        <div style={{
          marginTop: "40px",
          padding: "20px",
          background: "rgba(99, 102, 241, 0.1)",
          borderRadius: "12px",
          border: "1px solid rgba(129, 140, 248, 0.2)",
        }}>
          <p style={{ 
            margin: 0, 
            color: "#818cf8", 
            fontSize: "1em",
            fontStyle: "italic",
          }}>
            💡 This is just for fun! The real implementation would combine both models for the ultimate computer vision experience.
          </p>
        </div>
      </div>
    </div>
  );
};

