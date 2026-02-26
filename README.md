# SyncroFlow - README Documentation

A visual logic engine for real-time video intelligence that transforms complex AI video pipelines into simple, drag-and-drop nodes using ultra-low latency WebRTC streaming.

## Overview

SyncroFlow enables users to build custom computer vision and AI monitoring tools in seconds without requiring weeks of development. It combines drag-and-drop visual programming with AI-powered video analysis capabilities.

## Core Features

- **Drag-and-Drop Editor**: React Flow-based visual interface for intuitive AI pipeline creation
- **Ultra-Low Latency Video**: Sub-500ms WebRTC streaming via Ant Media Server integration
- **Temporal AI Monitoring**: Rolling frame buffers for action understanding over time
- **Multi-Source Inputs**: Support for webcams, MP4 files, screen shares, and RTMP streams
- **No-Code Automation**: Trigger real-world events based on AI analysis conditions

## Technology Stack

**Frontend**: React 18, Vite, React Flow, Tailwind CSS
**Backend API**: Node.js / Express
**AI/Vision Backend**: Python, FastAPI, YOLOv8
**Video Infrastructure**: Ant Media Server, WebRTC, RTMP
**AI Models**: Gemini Pro Vision / OpenRouter API

## Requirements

- Node.js v18+
- Python 3.10+
- Ant Media Server Community Edition
- OBS Studio (optional)

## Setup Instructions

1. Configure environment variables in `.env` file with API keys
2. Start Python backend with uvicorn on port 8001
3. Install dependencies and run Node backend with npm
4. Configure Ant Media Server with stream key
5. Access application at http://localhost:5173

## License

MIT License
