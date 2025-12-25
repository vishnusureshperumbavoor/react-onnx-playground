import React, { useState, useRef } from "react";
import { Mnist } from "./Mnist";
import { Yolo } from "./Yolo";
import { Sam2Component } from "./Sam2Component";
import { Sam3Component } from "./Sam3Component";

type TabType = "MNIST" | "YOLOv11" | "SAM2" | "SAM3";

interface Tab {
  id: string;
  type: TabType;
  title: string;
}

const tabConfig: Record<TabType, { icon: string; color: string; component: React.ComponentType }> = {
  MNIST: { icon: "✍️", color: "#667eea", component: Mnist },
  YOLOv11: { icon: "🎯", color: "#f5576c", component: Yolo },
  SAM2: { icon: "✂️", color: "#4facfe", component: Sam2Component },
  SAM3: { icon: "🔬", color: "#43e97b", component: Sam3Component },
};

export const TabBrowser: React.FC = () => {
  const [tabs, setTabs] = useState<Tab[]>([
    { id: "tab-1", type: "MNIST", title: "MNIST" },
    { id: "tab-2", type: "YOLOv11", title: "YOLOv11" },
    { id: "tab-3", type: "SAM2", title: "SAM2" },
    { id: "tab-4", type: "SAM3", title: "SAM3" },
  ]);
  const [activeTabId, setActiveTabId] = useState<string>("tab-2");
  const [draggedTab, setDraggedTab] = useState<string | null>(null);
  const tabIdCounter = useRef(5);

  const closeTab = (tabId: string) => {
    const newTabs = tabs.filter((tab) => tab.id !== tabId);
    if (newTabs.length === 0) {
      // If closing the last tab, add a default one
      const defaultTab = { id: `tab-${tabIdCounter.current++}`, type: "YOLOv11" as TabType, title: "YOLOv11" };
      setTabs([defaultTab]);
      setActiveTabId(defaultTab.id);
    } else {
      setTabs(newTabs);
      if (activeTabId === tabId) {
        // If closing active tab, switch to the last tab
        setActiveTabId(newTabs[newTabs.length - 1].id);
      }
    }
  };

  const handleDragStart = (e: React.DragEvent, tabId: string) => {
    setDraggedTab(tabId);
    e.dataTransfer.effectAllowed = "move";
  };

  const handleDragOver = (e: React.DragEvent, targetTabId: string) => {
    e.preventDefault();
    if (!draggedTab || draggedTab === targetTabId) return;

    const draggedIndex = tabs.findIndex((tab) => tab.id === draggedTab);
    const targetIndex = tabs.findIndex((tab) => tab.id === targetTabId);

    if (draggedIndex === -1 || targetIndex === -1) return;

    const newTabs = [...tabs];
    const [removed] = newTabs.splice(draggedIndex, 1);
    newTabs.splice(targetIndex, 0, removed);
    setTabs(newTabs);
  };

  const handleDragEnd = () => {
    setDraggedTab(null);
  };

  const activeTab = tabs.find((tab) => tab.id === activeTabId);
  const ActiveComponent = activeTab ? tabConfig[activeTab.type].component : null;

  return (
    <div style={{ 
      height: "100vh", 
      display: "flex", 
      flexDirection: "column",
      background: "linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%)",
    }}>
      {/* Browser chrome - tabs bar */}
      <div style={{
        background: "rgba(20, 20, 30, 0.95)",
        borderBottom: "1px solid rgba(129, 140, 248, 0.2)",
        display: "flex",
        alignItems: "flex-end",
        padding: "8px 8px 0 8px",
        gap: "4px",
        minHeight: "48px",
      }}>
        {tabs.map((tab) => {
          const config = tabConfig[tab.type];
          const isActive = tab.id === activeTabId;
          const isDragging = draggedTab === tab.id;

          return (
            <div
              key={tab.id}
              draggable
              onDragStart={(e) => handleDragStart(e, tab.id)}
              onDragOver={(e) => handleDragOver(e, tab.id)}
              onDragEnd={handleDragEnd}
              onClick={() => setActiveTabId(tab.id)}
              style={{
                background: isActive 
                  ? "rgba(30, 30, 46, 0.95)" 
                  : "rgba(20, 20, 30, 0.6)",
                border: "1px solid rgba(129, 140, 248, 0.2)",
                borderBottom: isActive ? "none" : "1px solid rgba(129, 140, 248, 0.2)",
                borderTopLeftRadius: "8px",
                borderTopRightRadius: "8px",
                padding: "8px 12px",
                display: "flex",
                alignItems: "center",
                gap: "8px",
                cursor: "pointer",
                minWidth: "160px",
                maxWidth: "240px",
                position: "relative",
                transition: "all 0.2s ease",
                opacity: isDragging ? 0.5 : 1,
                marginBottom: isActive ? "-1px" : "0",
              }}
              onMouseEnter={(e) => {
                if (!isActive) {
                  e.currentTarget.style.background = "rgba(30, 30, 46, 0.8)";
                }
              }}
              onMouseLeave={(e) => {
                if (!isActive) {
                  e.currentTarget.style.background = "rgba(20, 20, 30, 0.6)";
                }
              }}
            >
              {/* Colored indicator */}
              <div style={{
                position: "absolute",
                top: 0,
                left: 0,
                right: 0,
                height: "3px",
                background: isActive ? config.color : "transparent",
                borderTopLeftRadius: "8px",
                borderTopRightRadius: "8px",
              }} />
              
              <span style={{ fontSize: "1.2em" }}>{config.icon}</span>
              <span style={{ 
                flex: 1, 
                overflow: "hidden", 
                textOverflow: "ellipsis", 
                whiteSpace: "nowrap",
                fontSize: "0.9em",
                color: isActive ? "#e4e4e7" : "#a1a1aa",
              }}>
                {tab.title}
              </span>
              
              {/* Close button */}
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  closeTab(tab.id);
                }}
                style={{
                  background: "transparent",
                  border: "none",
                  color: "#a1a1aa",
                  cursor: "pointer",
                  padding: "4px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  borderRadius: "4px",
                  fontSize: "0.9em",
                  transition: "all 0.2s ease",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = "rgba(239, 68, 68, 0.2)";
                  e.currentTarget.style.color = "#ef4444";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = "transparent";
                  e.currentTarget.style.color = "#a1a1aa";
                }}
              >
                ✕
              </button>
            </div>
          );
        })}

      </div>

      {/* Content area */}
      <div style={{ 
        flex: 1, 
        overflow: "auto",
        background: "linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%)",
      }}>
        {ActiveComponent && <ActiveComponent />}
      </div>
    </div>
  );
};

