👁️ SyncroFlow

Don't just record the world—understand it.

A visual logic engine for real-time video intelligence.

💡 The Hook

Have you ever thought about how much "dead" video data is recorded every second? Millions of cameras are watching, but nobody is actually understanding.

If you want to detect a burglary or an elderly fall today, you have two bad options:

Pay humans to stare at screens until they get tired.

Hire developers for weeks to build a custom AI pipeline from scratch.

It’s slow, it’s expensive, and it's over-complicated.

🚀 The Solution

SyncroFlow turns complex AI video pipelines into simple, drag-and-drop nodes. Using ultra-low latency WebRTC, we allow you to build custom computer vision and AI monitoring tools in seconds, not months.

Drag a camera node (Webcam, Screen, or Remote Ant Media Stream).

Connect an AI logic block (Object Detection, Pose Estimation, Gemini Vision Analysis).

Automate the response (Discord, SMS, Email, API Webhooks).

✨ Key Features

Drag-and-Drop Editor: Built with React Flow, making AI pipeline generation visual and intuitive.

Ultra-Low Latency Video: Native integration with Ant Media Server allows sub-500ms WebRTC streaming directly inside the canvas nodes.

Temporal AI Monitoring: Rolling frame buffers allow the AI to understand actions over time (e.g., "Is the person picking up a cup and drinking?"), not just static images.

Multi-Source Inputs: Connect local webcams, pre-recorded MP4s, screen shares, or live RTMP streams from OBS.

No-Code Action Triggers: Trigger real-world events based on AI conditions (e.g., "If person falls -> Send SMS").

🛠️ Tech Stack

Frontend: React 18, Vite, React Flow, Tailwind CSS

Backend (API/Flow Logic): Node.js / Express

Backend (AI/Vision): Python, FastAPI, YOLOv8 (Pose/Object Detection)

Video Infrastructure: Ant Media Server (Community Edition), WebRTC, RTMP

AI Models: Gemini Pro Vision / OpenRouter API

⚙️ Installation & Setup

1. Prerequisites

Node.js (v18+)

Python 3.10+

Ant Media Server Community Edition (Running locally or via Docker/WSL)

OBS Studio (Optional, for simulating remote camera feeds)

2. Environment Variables

Create a .env file in the root directory and add your API keys:

OPENROUTER_API_KEY=your_openrouter_or_gemini_key


3. Start the Application

Terminal 1: Python AI Engine

cd python_backend
pip install -r requirements.txt
uvicorn main:app --port 8001 --reload


Terminal 2: Node Backend & React Frontend

npm install
npm run dev


4. Configure Ant Media Server

Start Ant Media Server (sudo service antmedia start or via Docker).

Go to http://localhost:5080 and create your admin account.

Use OBS Studio to publish a stream to rtmp://localhost/LiveApp with the Stream Key stream1.

🎮 How to Use

Open SyncroFlow at http://localhost:5173 (or your Vite port).

Open the Flow Editor.

Drag a Camera Node to the canvas.

Set the Video Source to Ant Media Server and use stream1. Click Connect Stream.

Drag an Analysis Node and connect it to the Camera Node.

Write your prompt: "Look at the video. Is someone stealing a package?"

Click Run Flow and watch the magic happen in real-time.

📄 License

MIT License.