import type { Express, Request, Response } from "express";
import express from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import multer from "multer";
import { spawn } from "child_process";
import path from "path";
import fs from "fs";
import { getTwilioClient, getTwilioFromPhoneNumber } from "./integrations/twilio";
import { getUncachableResendClient } from "./integrations/resend";
import { getGeminiClient } from "./integrations/gemini";
import { WebSocketServer, WebSocket } from "ws";
import { GoogleGenAI, Modality } from "@google/genai";

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB
});

// Video upload configuration - disk storage with higher limits
const videoUpload = multer({
  storage: multer.diskStorage({
    destination: (req, file, cb) => {
      const uploadDir = path.join(process.cwd(), 'uploads', 'videos');
      if (!fs.existsSync(uploadDir)) {
        fs.mkdirSync(uploadDir, { recursive: true });
      }
      cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
      const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
      const ext = path.extname(file.originalname);
      cb(null, `video-${uniqueSuffix}${ext}`);
    }
  }),
  limits: { fileSize: 200 * 1024 * 1024 }, // 200MB limit for videos
  fileFilter: (req, file, cb) => {
    // Only accept video files
    if (file.mimetype.startsWith('video/')) {
      cb(null, true);
    } else {
      cb(new Error('Only video files are allowed'));
    }
  }
});

// Python backend URL
const PYTHON_API_URL = process.env.PYTHON_API_URL || "http://localhost:8001";

export async function registerRoutes(app: Express): Promise<Server> {
  // Flow endpoints
  app.get("/api/flows", async (req: Request, res: Response) => {
    try {
      const flows = await storage.getAllFlows();
      res.json(flows);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch flows" });
    }
  });

  app.post("/api/flows", async (req: Request, res: Response) => {
    try {
      const flow = await storage.createFlow(req.body);
      res.json(flow);
    } catch (error) {
      res.status(500).json({ error: "Failed to create flow" });
    }
  });

  app.get("/api/flows/:id", async (req: Request, res: Response) => {
    try {
      const flow = await storage.getFlow(req.params.id);
      if (!flow) {
        res.status(404).json({ error: "Flow not found" });
        return;
      }
      res.json(flow);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch flow" });
    }
  });

  app.put("/api/flows/:id", async (req: Request, res: Response) => {
    try {
      const flow = await storage.updateFlow(req.params.id, req.body);
      if (!flow) {
        res.status(404).json({ error: "Flow not found" });
        return;
      }
      res.json(flow);
    } catch (error) {
      res.status(500).json({ error: "Failed to update flow" });
    }
  });

  app.delete("/api/flows/:id", async (req: Request, res: Response) => {
    try {
      const success = await storage.deleteFlow(req.params.id);
      if (!success) {
        res.status(404).json({ error: "Flow not found" });
        return;
      }
      res.json({ success: true });
    } catch (error) {
      res.status(500).json({ error: "Failed to delete flow" });
    }
  });

  // Demo flow template - Golf Swing Analysis
  app.get("/api/demo-flows/golf-swing", (_req: Request, res: Response) => {
    const golfSwingFlow = {
      id: `golf-swing-demo-${Date.now()}`,
      name: "Golf Swing Posture Analysis",
      description: "AI-powered golf swing analysis using pose detection to analyze form, posture, and technique",
      nodes: [
        {
          id: 'camera-1',
          type: 'camera',
          position: { x: 100, y: 200 },
          data: {
            label: 'Golf Swing Video',
            config: {
              inputMode: 'video',
              videoPath: '/api/media/videos/golf-swing-demo.avi'
            },
            videoUrl: '/api/media/videos/golf-swing-demo.avi',
            configured: true
          }
        },
        {
          id: 'pose-1',
          type: 'pose',
          position: { x: 400, y: 200 },
          data: {
            label: 'Pose Detection',
            config: {},
            configured: true
          }
        },
        {
          id: 'analysis-1',
          type: 'analysis',
          position: { x: 700, y: 200 },
          data: {
            label: 'Swing Analysis',
            config: {
              userPrompt: `Analyze the golf swing posture and form. Evaluate:

1. **Stance & Setup**: Feet position, weight distribution, spine angle
2. **Backswing**: Shoulder rotation, hip turn, club path, wrist hinge
3. **Impact Position**: Hip rotation, shoulder alignment, head position, weight transfer
4. **Follow-through**: Balance, extension, finish position

Provide specific feedback on:
- Posture quality (excellent/good/needs improvement)
- Key technical strengths
- Areas for improvement with actionable tips
- Overall swing rating (1-10)

Return detailed JSON with analysis results.`
            },
            configured: true
          }
        },
        {
          id: 'save-1',
          type: 'save',
          position: { x: 1000, y: 150 },
          data: {
            label: 'Save Analysis',
            config: {
              format: 'json'
            },
            configured: true
          }
        },
        {
          id: 'email-1',
          type: 'email',
          position: { x: 1000, y: 280 },
          data: {
            label: 'Email Results',
            config: {
              to: 'coach@golfacademy.com',
              subject: 'Golf Swing Analysis - {{timestamp}}',
              body: `Golf Swing Analysis Report:

{{analysis}}

Analysis completed at {{timestamp}}.

This AI-powered analysis uses pose detection to evaluate swing mechanics and provide personalized feedback.`
            },
            configured: true
          }
        }
      ],
      edges: [
        { id: 'e1', source: 'camera-1', target: 'pose-1' },
        { id: 'e2', source: 'pose-1', target: 'analysis-1' },
        { id: 'e3', source: 'analysis-1', target: 'save-1' },
        { id: 'e4', source: 'analysis-1', target: 'email-1' }
      ],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };

    res.json(golfSwingFlow);
  });

  // Execution endpoints
  app.get("/api/executions", async (req: Request, res: Response) => {
    try {
      const executions = await storage.getAllExecutions();
      res.json(executions);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch executions" });
    }
  });

  app.post("/api/executions", async (req: Request, res: Response) => {
    try {
      const execution = await storage.createExecution(req.body);
      res.json(execution);
    } catch (error) {
      res.status(500).json({ error: "Failed to create execution" });
    }
  });

  app.get("/api/executions/:id", async (req: Request, res: Response) => {
    try {
      const execution = await storage.getExecution(req.params.id);
      if (!execution) {
        res.status(404).json({ error: "Execution not found" });
        return;
      }
      res.json(execution);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch execution" });
    }
  });

  app.put("/api/executions/:id", async (req: Request, res: Response) => {
    try {
      const execution = await storage.updateExecution(req.params.id, req.body);
      if (!execution) {
        res.status(404).json({ error: "Execution not found" });
        return;
      }
      res.json(execution);
    } catch (error) {
      res.status(500).json({ error: "Failed to update execution" });
    }
  });

  // Video media endpoints
  app.post("/api/media/videos", videoUpload.single("video"), async (req: Request, res: Response) => {
    try {
      console.log('[VIDEO_UPLOAD] Received upload request');
      console.log('[VIDEO_UPLOAD] req.file:', req.file ? 'present' : 'missing');
      console.log('[VIDEO_UPLOAD] req.body:', req.body);
      
      if (!req.file) {
        console.log('[VIDEO_UPLOAD] Error: No file in request');
        res.status(400).json({ error: "No video file provided" });
        return;
      }

      console.log('[VIDEO_UPLOAD] File details:', {
        originalname: req.file.originalname,
        mimetype: req.file.mimetype,
        size: req.file.size,
        path: req.file.path,
      });

      // Create video asset metadata in storage
      const asset = await storage.createVideoAsset({
        flowId: req.body.flowId || null,
        fileName: req.file.originalname,
        mimeType: req.file.mimetype,
        size: req.file.size,
        storagePath: req.file.path,
      });

      console.log('[VIDEO_UPLOAD] Created asset:', asset.id);

      // Extract server-generated filename from path
      const serverFileName = path.basename(req.file.path);
      console.log('[VIDEO_UPLOAD] Server filename:', serverFileName);

      // Return asset metadata with download URL
      res.json({
        id: asset.id,
        fileName: asset.fileName,
        serverFileName: serverFileName, // Include server-generated filename for persistence
        mimeType: asset.mimeType,
        size: asset.size,
        downloadUrl: `/api/media/videos/${asset.id}`,
        createdAt: asset.createdAt,
      });
    } catch (error) {
      console.error("[VIDEO_UPLOAD] Error:", error);
      res.status(500).json({ error: "Failed to upload video" });
    }
  });

  app.get("/api/media/videos/:id", async (req: Request, res: Response) => {
    try {
      // Check if it's a direct file request (e.g., golf-swing-demo.avi)
      if (req.params.id.includes('.')) {
        // Direct file access
        const filename = req.params.id;
        const filePath = path.join(process.cwd(), 'uploads', 'videos', filename);
        
        if (!fs.existsSync(filePath)) {
          res.status(404).json({ error: "Video file not found" });
          return;
        }
        
        // Determine MIME type from file extension
        const ext = path.extname(filename).toLowerCase();
        const mimeTypes: Record<string, string> = {
          '.mp4': 'video/mp4',
          '.webm': 'video/webm',
          '.avi': 'video/x-msvideo',
          '.mov': 'video/quicktime',
          '.mkv': 'video/x-matroska',
          '.m4v': 'video/x-m4v',
        };
        const mimeType = mimeTypes[ext] || 'video/mp4';
        
        res.setHeader('Content-Type', mimeType);
        res.setHeader('Content-Disposition', `inline; filename="${filename}"`);
        res.setHeader('Accept-Ranges', 'bytes');
        
        const fileStream = fs.createReadStream(filePath);
        fileStream.pipe(res);
        return;
      }
      
      // Database asset lookup
      const asset = await storage.getVideoAsset(req.params.id);
      if (!asset) {
        res.status(404).json({ error: "Video not found" });
        return;
      }

      // Check if file exists
      if (!fs.existsSync(asset.storagePath)) {
        res.status(404).json({ error: "Video file not found on disk" });
        return;
      }

      // Stream the video file
      res.setHeader('Content-Type', asset.mimeType);
      res.setHeader('Content-Disposition', `inline; filename="${asset.fileName}"`);
      res.setHeader('Accept-Ranges', 'bytes');
      
      const fileStream = fs.createReadStream(asset.storagePath);
      fileStream.pipe(res);
    } catch (error) {
      console.error("Video download error:", error);
      res.status(500).json({ error: "Failed to download video" });
    }
  });

  app.delete("/api/media/videos/:id", async (req: Request, res: Response) => {
    try {
      const asset = await storage.getVideoAsset(req.params.id);
      if (!asset) {
        res.status(404).json({ error: "Video not found" });
        return;
      }

      // Delete file from disk if it exists
      if (fs.existsSync(asset.storagePath)) {
        fs.unlinkSync(asset.storagePath);
      }

      // Delete from storage
      await storage.deleteVideoAsset(req.params.id);

      res.json({ success: true });
    } catch (error) {
      console.error("Video deletion error:", error);
      res.status(500).json({ error: "Failed to delete video" });
    }
  });

  // Detection endpoint - proxy to Python backend
  app.post(
    "/api/detect",
    upload.single("image"),
    async (req: Request, res: Response) => {
      try {
        if (!req.file) {
          res.status(400).json({ error: "No image file provided" });
          return;
        }

        // Forward to Python backend
        const formData = new FormData();
        const blob = new Blob([new Uint8Array(req.file.buffer)], { type: req.file.mimetype });
        formData.append("file", blob, req.file.originalname);

        const response = await fetch(`${PYTHON_API_URL}/api/detect`, {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Python API error: ${response.statusText}`);
        }

        const result = await response.json();
        res.json(result);
      } catch (error) {
        console.error("Detection error:", error);
        res.status(500).json({ error: "Detection failed" });
      }
    }
  );

  // Pose detection endpoint - proxy to Python backend
  app.post(
    "/api/detect-pose",
    upload.single("file"),
    async (req: Request, res: Response) => {
      try {
        if (!req.file) {
          res.status(400).json({ error: "No image file provided" });
          return;
        }

        // Forward to Python backend
        const formData = new FormData();
        const blob = new Blob([new Uint8Array(req.file.buffer)], { type: req.file.mimetype });
        formData.append("file", blob, req.file.originalname);

        const response = await fetch(`${PYTHON_API_URL}/api/detect-pose`, {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Python API error: ${response.statusText}`);
        }

        const result = await response.json();
        res.json(result);
      } catch (error) {
        console.error("Pose detection error:", error);
        res.status(500).json({ error: "Pose detection failed" });
      }
    }
  );

  // Analysis endpoint - proxy to Python backend
  app.post("/api/analyze", async (req: Request, res: Response) => {
    try {
      const response = await fetch(`${PYTHON_API_URL}/api/analyze`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(req.body),
      });

      if (!response.ok) {
        throw new Error(`Python API error: ${response.statusText}`);
      }

      const result = await response.json();
      res.json(result);
    } catch (error) {
      console.error("Analysis error:", error);
      res.status(500).json({ error: "Analysis failed" });
    }
  });

  // Save results endpoint - proxy to Python backend
  app.post("/api/save-results", async (req: Request, res: Response) => {
    try {
      const response = await fetch(`${PYTHON_API_URL}/api/save-results`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(req.body),
      });

      if (!response.ok) {
        throw new Error(`Python API error: ${response.statusText}`);
      }

      const result = await response.json();
      res.json(result);
    } catch (error) {
      console.error("Save error:", error);
      res.status(500).json({ error: "Save failed" });
    }
  });

  // List results endpoint - proxy to Python backend
  app.get("/api/results", async (req: Request, res: Response) => {
    try {
      const response = await fetch(`${PYTHON_API_URL}/api/results`);

      if (!response.ok) {
        throw new Error(`Python API error: ${response.statusText}`);
      }

      const result = await response.json();
      res.json(result);
    } catch (error) {
      console.error("List results error:", error);
      res.status(500).json({ error: "Failed to list results" });
    }
  });

  // Generate flow endpoint - proxy to Python backend (Gemini AI)
  app.post("/api/generate-flow", async (req: Request, res: Response) => {
    try {
      const response = await fetch(`${PYTHON_API_URL}/api/generate-flow`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(req.body),
      });

      if (!response.ok) {
        throw new Error(`Python API error: ${response.statusText}`);
      }

      const result = await response.json();
      res.json(result);
    } catch (error) {
      console.error("Flow generation error:", error);
      res.status(500).json({ error: "Failed to generate flow" });
    }
  });

  // Edit flow endpoint - proxy to Python backend (Gemini AI)
  app.post("/api/edit-flow", async (req: Request, res: Response) => {
    try {
      const response = await fetch(`${PYTHON_API_URL}/api/edit-flow`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(req.body),
      });

      if (!response.ok) {
        throw new Error(`Python API error: ${response.statusText}`);
      }

      const result = await response.json();
      res.json(result);
    } catch (error) {
      console.error("Flow edit error:", error);
      res.status(500).json({ error: "Failed to edit flow" });
    }
  });

  // Transcribe audio endpoint - proxy to Python backend with multipart/form-data
  app.post("/api/transcribe", upload.single("audio"), async (req: Request, res: Response) => {
    try {
      console.log('[EXPRESS-TRANSCRIBE] Received transcription request');
      
      if (!req.file) {
        console.log('[EXPRESS-TRANSCRIBE] No file in request');
        res.status(400).json({ error: "No audio file provided" });
        return;
      }

      console.log(`[EXPRESS-TRANSCRIBE] File received: ${req.file.size} bytes, type: ${req.file.mimetype}`);

      // Forward to Python backend using native FormData with Buffer
      const formData = new FormData();
      const blob = new Blob([new Uint8Array(req.file.buffer)], { type: req.file.mimetype || 'audio/webm' });
      formData.append("audio", blob, req.file.originalname || 'audio.webm');

      console.log('[EXPRESS-TRANSCRIBE] Forwarding to Python backend...');
      const response = await fetch(`${PYTHON_API_URL}/api/transcribe`, {
        method: "POST",
        body: formData,
      });

      console.log(`[EXPRESS-TRANSCRIBE] Python response: ${response.status} ${response.statusText}`);

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`[EXPRESS-TRANSCRIBE] Python error: ${errorText}`);
        res.status(500).json({ error: `Python API error: ${response.statusText}` });
        return;
      }

      const result = await response.json();
      console.log(`[EXPRESS-TRANSCRIBE] Success: ${result.transcript?.length || 0} chars`);
      res.json(result);
    } catch (error) {
      console.error("[EXPRESS-TRANSCRIBE] Error:", error);
      res.status(500).json({ error: "Transcription failed", details: error instanceof Error ? error.message : String(error) });
    }
  });

  // Serve individual result files
  app.use('/results', (req, res, next) => {
    const resultsPath = path.join(process.cwd(), 'results');
    if (!fs.existsSync(resultsPath)) {
      fs.mkdirSync(resultsPath, { recursive: true });
    }
    next();
  });
  app.use('/results', (req, res, next) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    next();
  });
  app.use('/results', express.static(path.join(process.cwd(), 'results')));

  // Execute flow endpoint - process nodes in sequence or individual steps
  app.post("/api/execute-flow", async (req: Request, res: Response) => {
    try {
      const { imageData, imagesData, userPrompt, outputFormat, stepOnly, detections: inputDetections, analysis: inputAnalysis, motion_state, velocity, execution_id, isDemoFlow } = req.body;

      if (!imageData && stepOnly !== 'save') {
        res.status(400).json({ error: "No image data provided" });
        return;
      }

      // Handle individual step execution
      if (stepOnly === 'detection') {
        // Just run detection step
        const base64Data = imageData.split(',')[1];
        const buffer = Buffer.from(base64Data, 'base64');
        const imageBlob = new Blob([new Uint8Array(buffer)], { type: 'image/jpeg' });

        const detectFormData = new FormData();
        detectFormData.append('file', imageBlob, 'capture.jpg');

        console.log('[EXPRESS-DETECT] Calling Python API...');
        const detectResponse = await fetch(`${PYTHON_API_URL}/api/detect`, {
          method: 'POST',
          body: detectFormData,
        });

        if (!detectResponse.ok) {
          throw new Error('Detection failed');
        }

        const detectResult = await detectResponse.json();
        console.log('[EXPRESS-DETECT] Python response:', detectResult);
        console.log('[EXPRESS-DETECT] Detections count:', detectResult.detections?.length || 0);
        console.log('[EXPRESS-DETECT] Returning to frontend:', { detections: detectResult.detections || [] });
        res.json({ detections: detectResult.detections || [] });
        return;
      }

      if (stepOnly === 'analysis') {
        // Just run analysis step
        const analyzeResponse = await fetch(`${PYTHON_API_URL}/api/analyze`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image: imageData,
            images: imagesData || [imageData], 
            detections: inputDetections || [],
            context: "general",
            userPrompt: userPrompt || "Analyze this image",
            motion_state,
            velocity,
            execution_id,
            isDemoFlow: isDemoFlow || false
          }),
        });

        if (!analyzeResponse.ok) {
          throw new Error('Analysis failed');
        }

        const analyzeResult = await analyzeResponse.json();
        res.json({ analysis: analyzeResult.analysis });
        return;
      }

      if (stepOnly === 'save') {
        // Just run save step
        console.log('[SAVE] Received data:', {
          detections: inputDetections?.length || 0,
          hasAnalysis: !!inputAnalysis,
          userPrompt
        });
        
        const saveResponse = await fetch(`${PYTHON_API_URL}/api/save-results`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            data: {
              detections: inputDetections || [],
              analysis: inputAnalysis || null,
              timestamp: new Date().toISOString(),
              userPrompt: userPrompt
            },
            format: outputFormat || "json"
          }),
        });

        if (!saveResponse.ok) {
          throw new Error('Save failed');
        }

        const saveResult = await saveResponse.json();
        console.log('[SAVE] Result:', saveResult);
        res.json({ saved: saveResult });
        return;
      }

      // Full pipeline execution
      // Create execution record
      const execution = await storage.createExecution({
        flowName: "Camera Pipeline",
        status: "running",
        currentNode: "camera",
        detections: [],
        analysis: null,
      });

      try {
        // Step 1: Detection (Gemini Vision)
        await storage.updateExecution(execution.id, {
          currentNode: "detection",
          status: "running",
        });

        // Convert base64 to buffer (Node.js compatible)
        const base64Data = imageData.split(',')[1];
        const buffer = Buffer.from(base64Data, 'base64');
        const imageBlob = new Blob([new Uint8Array(buffer)], { type: 'image/jpeg' });

        const detectFormData = new FormData();
        detectFormData.append('file', imageBlob, 'capture.jpg');

        const detectResponse = await fetch(`${PYTHON_API_URL}/api/detect`, {
          method: 'POST',
          body: detectFormData,
        });

        if (!detectResponse.ok) {
          throw new Error('Detection failed');
        }

        const detectResult = await detectResponse.json();
        const detections = detectResult.detections || [];

        // Step 2: Analysis (Gemini LLM with user prompt)
        await storage.updateExecution(execution.id, {
          currentNode: "analysis",
          detections: detections,
        });

        const analyzeResponse = await fetch(`${PYTHON_API_URL}/api/analyze`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image: imageData,
            detections: detections,
            context: "general",
            userPrompt: userPrompt || "Extract any relevant information from this image"
          }),
        });

        if (!analyzeResponse.ok) {
          throw new Error('Analysis failed');
        }

        const analyzeResult = await analyzeResponse.json();
        const analysis = analyzeResult.analysis;

        // Step 3: Save results
        await storage.updateExecution(execution.id, {
          currentNode: "save",
          analysis: analysis,
        });

        const saveResponse = await fetch(`${PYTHON_API_URL}/api/save-results`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            data: {
              detections: detections,
              analysis: analysis,
              timestamp: new Date().toISOString(),
              userPrompt: userPrompt
            },
            format: outputFormat || "json"
          }),
        });

        if (!saveResponse.ok) {
          throw new Error('Save failed');
        }

        const saveResult = await saveResponse.json();

        // Mark execution as completed
        await storage.updateExecution(execution.id, {
          status: "completed",
        });

        res.json({
          success: true,
          execution: execution,
          detections: detections,
          analysis: analysis,
          saved: saveResult
        });

      } catch (error: any) {
        // Mark execution as failed
        await storage.updateExecution(execution.id, {
          status: "failed",
          error: error.message
        });
        throw error;
      }

    } catch (error: any) {
      console.error("Flow execution error:", error);
      res.status(500).json({ error: error.message || "Flow execution failed" });
    }
  });

  // Send email notification
  // Analyze transcript using Gemini API
  app.post("/api/analyze-transcript", async (req: Request, res: Response) => {
    try {
      const { transcript, userPrompt } = req.body;
      
      if (!transcript) {
        return res.status(400).json({ error: "No transcript provided" });
      }

      const gemini = await getGeminiClient();
      
      const prompt = userPrompt || "Summarize this transcript and extract key points";
      const fullPrompt = `${prompt}\n\nTranscript:\n${transcript}`;
      
      const result = await gemini.models.generateContent({
        model: 'gemini-2.0-flash-exp',
        contents: fullPrompt,
      });
      const analysisText = result.text;
      
      console.log("[TRANSCRIPT-ANALYSIS] Analyzed transcript");
      
      res.json({ 
        analysis: {
          summary: analysisText,
          condition_met: true // For compatibility with existing flow logic
        }
      });
    } catch (error: any) {
      console.error("Transcript analysis error:", error);
      res.status(500).json({ error: error.message || "Failed to analyze transcript" });
    }
  });

  app.post("/api/send-email", async (req: Request, res: Response) => {
    try {
      const { to, subject, body } = req.body;
      
      if (!to || !subject || !body) {
        return res.status(400).json({ error: "Missing required fields: to, subject, body" });
      }

      // Validate email address
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (!emailRegex.test(to)) {
        return res.status(400).json({ error: "Invalid email address" });
      }

      // Send email via Resend integration
      const { client, fromEmail } = await getUncachableResendClient();
      
      const result = await client.emails.send({
        from: 'Mauto <onboarding@resend.dev>',
        to: [to],
        subject: subject,
        html: `<p>${body}</p>`,
      });

      console.log("[EMAIL] Email sent successfully:", result.data?.id);
      console.log("[EMAIL] Full result:", JSON.stringify(result));
      
      res.json({ 
        success: true, 
        message: "Email sent successfully",
        details: { to, subject, id: result.data?.id }
      });
    } catch (error: any) {
      console.error("Email send error:", error);
      res.status(500).json({ error: error.message || "Failed to send email" });
    }
  });

  // Send SMS notification
  app.post("/api/send-sms", async (req: Request, res: Response) => {
    try {
      const { to, message } = req.body;
      
      if (!to || !message) {
        return res.status(400).json({ error: "Missing required fields: to, message" });
      }

      // Validate phone number format (basic check for E.164 format)
      const phoneRegex = /^\+[1-9]\d{1,14}$/;
      if (!phoneRegex.test(to)) {
        return res.status(400).json({ 
          error: "Invalid phone number. Must be in E.164 format (e.g., +1234567890)" 
        });
      }

      // Send SMS via Twilio integration
      const twilioClient = await getTwilioClient();
      const fromNumber = await getTwilioFromPhoneNumber();
      
      const smsResult = await twilioClient.messages.create({
        body: message,
        from: fromNumber,
        to: to
      });

      console.log("[SMS] SMS sent successfully:", smsResult.sid);
      
      res.json({ 
        success: true, 
        message: "SMS sent successfully",
        details: { to, sid: smsResult.sid }
      });
    } catch (error: any) {
      console.error("SMS send error:", error);
      res.status(500).json({ error: error.message || "Failed to send SMS" });
    }
  });

  // Make voice call
  app.post("/api/make-call", async (req: Request, res: Response) => {
    try {
      const { to, message } = req.body;
      
      if (!to || !message) {
        return res.status(400).json({ error: "Missing required fields: to, message" });
      }

      // Validate phone number format (basic check for E.164 format)
      const phoneRegex = /^\+[1-9]\d{1,14}$/;
      if (!phoneRegex.test(to)) {
        return res.status(400).json({ 
          error: "Invalid phone number. Must be in E.164 format (e.g., +1234567890)" 
        });
      }

      // Make voice call via Twilio Studio Flow integration
      const twilioClient = await getTwilioClient();
      const fromNumber = await getTwilioFromPhoneNumber();
      const flowSid = process.env.TWILIO_STUDIO_FLOW_SID || 'FW24c25889f3dd13f9044e1cee3d0ae481';
      
      const callResult = await twilioClient.studio.v2.flows(flowSid)
        .executions
        .create({
          to: to,
          from: fromNumber,
          parameters: {
            message: message // Passes the AI's analysis message to your Twilio Flow
          }
        });

      console.log("[CALL] Studio Flow execution initiated successfully:", callResult.sid);
      
      res.json({ 
        success: true, 
        message: "Call initiated successfully",
        details: { to, sid: callResult.sid }
      });
    } catch (error: any) {
      console.error("Call error:", error);
      res.status(500).json({ error: error.message || "Failed to make call" });
    }
  });

  // Send Discord message
  app.post("/api/send-discord", async (req: Request, res: Response) => {
    try {
      const { webhookUrl, message } = req.body;
      
      if (!webhookUrl || !message) {
        return res.status(400).json({ error: "Missing required fields: webhookUrl, message" });
      }

      // Security: Validate Discord webhook URL to prevent SSRF attacks
      try {
        const url = new URL(webhookUrl);
        const allowedDomains = ['discord.com', 'discordapp.com'];
        if (!allowedDomains.includes(url.hostname)) {
          return res.status(400).json({ 
            error: "Invalid webhook URL. Only Discord webhooks are allowed (discord.com or discordapp.com)" 
          });
        }
        if (!url.pathname.startsWith('/api/webhooks/')) {
          return res.status(400).json({ 
            error: "Invalid webhook URL. Must be a valid Discord webhook endpoint" 
          });
        }
      } catch (urlError) {
        return res.status(400).json({ error: "Invalid webhook URL format" });
      }

      // Send message to Discord webhook
      const response = await fetch(webhookUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content: message
        })
      });

      if (!response.ok) {
        throw new Error(`Discord API error: ${response.status} ${response.statusText}`);
      }

      console.log("[DISCORD] Message sent to Discord");
      
      res.json({ 
        success: true, 
        message: "Discord message sent successfully"
      });
    } catch (error: any) {
      console.error("Discord send error:", error);
      res.status(500).json({ error: error.message || "Failed to send Discord message" });
    }
  });

  // Health check endpoint
  app.get("/api/health", async (req: Request, res: Response) => {
    try {
      // Check if Python backend is running
      const pythonHealth = await fetch(`${PYTHON_API_URL}/health`).then((r) =>
        r.json()
      );
      res.json({
        status: "healthy",
        services: {
          express: "healthy",
          python: pythonHealth.status,
        },
      });
    } catch (error) {
      res.json({
        status: "degraded",
        services: {
          express: "healthy",
          python: "unavailable",
        },
      });
    }
  });

  const httpServer = createServer(app);

  // WebSocket server for Gemini Live API transcription
  const wss = new WebSocketServer({ 
    server: httpServer,
    path: '/api/transcribe/live'
  });

  wss.on('connection', async (ws: WebSocket) => {
    console.log('[GEMINI_LIVE_WS] Client connected');
    
    let geminiSession: any = null;
    let isConnected = false;

    try {
      // Initialize Gemini AI client
      const apiKey = process.env.GEMINI_API_KEY;
      if (!apiKey) {
        ws.send(JSON.stringify({ type: 'error', error: 'GEMINI_API_KEY not configured' }));
        ws.close();
        return;
      }

      const ai = new GoogleGenAI({ apiKey });

      // Connect to Gemini Live API
      geminiSession = await ai.live.connect({
        model: 'gemini-2.0-flash-live-001',
        callbacks: {
          onopen: () => {
            isConnected = true;
            console.log('[GEMINI_LIVE_WS] Connected to Gemini Live API');
          },
          onmessage: (message: any) => {
            try {
              if (message.serverContent?.modelTurn?.parts) {
                const parts = message.serverContent.modelTurn.parts;
                for (const part of parts) {
                  if (part.text) {
                    console.log('[GEMINI_LIVE_WS] Transcript from Gemini:', part.text);
                    ws.send(JSON.stringify({ type: 'transcript', text: part.text }));
                  }
                }
              }
            } catch (error) {
              console.error('[GEMINI_LIVE_WS] Error processing Gemini message:', error);
            }
          },
          onerror: (error: any) => {
            console.error('[GEMINI_LIVE_WS] Gemini session error:', error);
            ws.send(JSON.stringify({ type: 'error', error: error.message || 'Gemini error' }));
          },
          onclose: () => {
            console.log('[GEMINI_LIVE_WS] Gemini session closed');
            isConnected = false;
          },
        },
        config: {
          responseModalities: [Modality.TEXT],
          systemInstruction: 'You are a real-time transcription assistant. Transcribe the audio input accurately and provide only the transcribed text without any additional commentary. Combine partial transcripts into complete sentences.',
        },
      });

      // Listen for audio from client
      ws.on('message', async (data: Buffer) => {
        try {
          const message = JSON.parse(data.toString());
          
          if (message.type === 'audio' && message.data && isConnected && geminiSession) {
            // Forward audio to Gemini Live API
            await geminiSession.sendRealtimeInput({
              audio: {
                data: message.data,
                mimeType: 'audio/pcm;rate=16000',
              },
            });
          }
        } catch (error) {
          console.error('[GEMINI_LIVE_WS] Error processing client message:', error);
        }
      });

      ws.on('close', () => {
        console.log('[GEMINI_LIVE_WS] Client disconnected');
        if (geminiSession) {
          geminiSession.close?.();
        }
      });

    } catch (error: any) {
      console.error('[GEMINI_LIVE_WS] Error setting up session:', error);
      ws.send(JSON.stringify({ type: 'error', error: error.message || 'Failed to connect to Gemini Live API' }));
      ws.close();
    }
  });

  console.log('[GEMINI_LIVE_WS] WebSocket server initialized on /api/transcribe/live');

  return httpServer;
}
