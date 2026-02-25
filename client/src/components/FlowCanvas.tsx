// client\src\components\FlowCanvas.tsx

import { useCallback, useState, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { WebRTCAdaptor } from '@antmedia/webrtc_adaptor';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  addEdge,
  useNodesState,
  useEdgesState,
  Connection,
  Edge,
  Node,
  BackgroundVariant,
  Handle,
  Position,
  useReactFlow,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { Camera, Search, Brain, Play, CheckCircle, Maximize2, Minimize2, Mail, MessageSquare, Phone, Send, Mic, User, RadioTower } from 'lucide-react';
import { Card } from '@/components/ui/card';

interface FlowCanvasProps {
  initialNodes?: Node[];
  initialEdges?: Edge[];
  onNodesChange?: (nodes: Node[]) => void;
  onEdgesChange?: (edges: Edge[]) => void;
  onNodeClick?: (event: any, node: Node) => void;
  onPaneClick?: () => void;
  onNodeAdd?: (nodeType: string) => void;
}

const nodeTypes = {
  camera: CameraNode,
  transcription: TranscriptionNode,
  detection: DetectionNode,
  pose: PoseNode,
  analysis: AnalysisNode,
  action: ActionNode,
  approval: ApprovalNode,
  email: EmailNode,
  sms: SMSNode,
  call: CallNode,
  discord: DiscordNode,
};

function CameraNode({ data, id }: { data: any; id: string }) {
  const inputMode = data.config?.inputMode || 'webcam';
  const [stream, setStream] = useState<MediaStream | null>(data.stream || null);
  const [isPreviewActive, setIsPreviewActive] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [amsError, setAmsError] = useState<string | null>(null);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const expandedVideoRef = useRef<HTMLVideoElement>(null);
  const expandedCanvasRef = useRef<HTMLCanvasElement>(null);
  const webRTCAdaptorRef = useRef<any>(null); // NEW: Ant Media Reference
  
  // Clean up ALL streams
  const stopPreview = useCallback(() => {
    // 1. Stop Ant Media
    if (webRTCAdaptorRef.current) {
      console.log(`[CAMERA] Stopping Ant Media Preview`);
      try { webRTCAdaptorRef.current.stop(); } catch(e) {}
      webRTCAdaptorRef.current = null;
    }

    // 2. Stop Webcams/Screen
    if (stream && stream instanceof MediaStream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }

    // 3. Stop Videos
    if (videoUrl && videoRef.current) {
      videoRef.current.pause();
    }

    setIsPreviewActive(false);
    if (data.onStreamChange) {
      data.onStreamChange(id, null);
    }
  }, [stream, videoUrl, id, data]);


  const startPreview = useCallback(async () => {
    console.log('[CAMERA] startPreview() called, inputMode:', inputMode);
    stopPreview(); // Clean up first
    setAmsError(null);

    // 1. Handle Pre-Recorded Video Mode
    if (inputMode === 'video') {
      if (!data.videoUrl) return;
      setIsPreviewActive(true);
      return;
    }
    
    // 2. NEW: Handle Ant Media Server Mode
    if (inputMode === 'antmedia') {
      const amsUrl = data.config?.amsUrl || 'ws://localhost:5080/LiveApp/websocket';
      const streamId = data.config?.streamId || 'stream1';
      console.log(`[CAMERA] Connecting Ant Media Preview to ${streamId}`);
      
      try {
        webRTCAdaptorRef.current = new WebRTCAdaptor({
          websocket_url: amsUrl,
          mediaConstraints: { video: false, audio: false },
          peerconnection_config: { iceServers: [{ urls: 'stun:stun1.l.google.com:19302' }] },
          sdp_constraints: { OfferToReceiveAudio: false, OfferToReceiveVideo: true },
          remoteVideoId: `preview-video-${id}`, // Critical for AMS binding
          callback: (info: string, obj: any) => {
            if (info === 'initialized') {
              webRTCAdaptorRef.current.play(streamId);
            } else if (info === 'play_started') {
              setIsPreviewActive(true);
              setStream(new MediaStream()); // Mock stream to pass internal checks
            } else if (info === 'closed') {
              setIsPreviewActive(false);
            }
          },
          callbackError: (err: string) => {
            setAmsError(`AMS Error: ${err}`);
            setIsPreviewActive(false);
          }
        });
      } catch (e: any) {
        setAmsError(e.message || "Failed to start AMS");
      }
      return;
    }

    // 3. Handle Webcam / Screen Share
    try {
      let newStream;
      if (inputMode === 'screen') {
        const includeAudio = data.config?.includeAudio !== false;
        newStream = await navigator.mediaDevices.getDisplayMedia({
          video: { width: 640, height: 360 },
          audio: includeAudio ? { echoCancellation: true, noiseSuppression: true } : false,
        });
      } else {
        newStream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 360 },
        });
      }
      
      setStream(newStream);
      setIsPreviewActive(true);
      
      if (data.onStreamChange) {
        data.onStreamChange(id, newStream);
      }
    } catch (error) {
      console.error('[CAMERA] Failed to start camera:', error);
    }
  }, [inputMode, data, id, stopPreview]);

  // Handle video loading for both stream and file
  useEffect(() => {
    const setupVideo = async () => {
      if (!videoRef.current) return;
      
      // We skip setting srcObject for Ant Media because WebRTCAdaptor handles the HTML element directly via its ID
      if (inputMode === 'antmedia') return; 

      if (inputMode === 'video' && videoUrl) {
        videoRef.current.src = videoUrl;
        videoRef.current.loop = true;
        try { await videoRef.current.play(); } catch (e) {}
      } else if (stream && stream instanceof MediaStream && stream.getTracks().length > 0) {
        videoRef.current.srcObject = stream;
        try { await videoRef.current.play(); } catch (e) {}
      }
    };
    setupVideo();
  }, [stream, videoUrl, inputMode]);

  // Separate effect for expanded video
  useEffect(() => {
    const setupExpandedVideo = async () => {
      if (!isExpanded || !expandedVideoRef.current) return;
      if (inputMode === 'antmedia') return; // Ant media doesn't easily duplicate streams to a second tag

      if (inputMode === 'video' && videoUrl) {
        expandedVideoRef.current.src = videoUrl;
        expandedVideoRef.current.loop = true;
        try { await expandedVideoRef.current.play(); } catch (e) {}
      } else if (stream && stream instanceof MediaStream && stream.getTracks().length > 0) {
        expandedVideoRef.current.srcObject = stream;
        try { await expandedVideoRef.current.play(); } catch (e) {}
      }
    };
    setupExpandedVideo();
  }, [isExpanded, stream, videoUrl, inputMode]);

  // ESC key to close fullscreen
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isExpanded) setIsExpanded(false);
    };
    window.addEventListener('keydown', handleEscape);
    return () => window.removeEventListener('keydown', handleEscape);
  }, [isExpanded]);
  
  // Handle video URL changes (including injected demo videos)
  useEffect(() => {
    const injectedVideoPath = data.config?.videoPath;
    if (injectedVideoPath) {
      setVideoUrl(injectedVideoPath);
      setIsPreviewActive(true);
    } else if (data.videoUrl && inputMode === 'video') {
      setVideoUrl(data.videoUrl);
      setIsPreviewActive(true);
    } else if (inputMode !== 'video' && videoUrl && !injectedVideoPath) {
      setVideoUrl(null);
      if (!stream) setIsPreviewActive(false);
    }
  }, [data.videoUrl, data.config?.videoPath, inputMode, stream, videoUrl]);
  
  // Use refs to remember the last trigger time and prevent infinite loops!
  const lastPreviewRef = useRef(data.triggerPreview);
  const lastStopRef = useRef(data.triggerStop);

  // Handle preview triggers from config panel safely
  useEffect(() => {
    if (data.triggerPreview && data.triggerPreview !== lastPreviewRef.current) {
      lastPreviewRef.current = data.triggerPreview; // Save the memory
      startPreview(); // Only run once per click
    }
  }, [data.triggerPreview, startPreview]);
  
  useEffect(() => {
    if (data.triggerStop && data.triggerStop !== lastStopRef.current) {
      lastStopRef.current = data.triggerStop; // Save the memory
      stopPreview(); // Only run once per click
    }
  }, [data.triggerStop, stopPreview]);
  
  // Draw bounding boxes on both canvases (Unchanged!)
  useEffect(() => {
    if (inputMode === 'video') return;
    
    const detections = data.detections || [];
    const drawBoxesOnCanvas = (canvas: HTMLCanvasElement | null, video: HTMLVideoElement | null, scale: number = 1) => {
      if (!canvas || !video) return null;
      const ctx = canvas.getContext('2d');
      if (!ctx) return null;
      
      const drawBoxes = () => {
        if (!video.videoWidth || !video.videoHeight) return;
        
        canvas.width = video.clientWidth;
        canvas.height = video.clientHeight;
        
        const videoAspect = video.videoWidth / video.videoHeight;
        const canvasAspect = canvas.width / canvas.height;
        let displayWidth, displayHeight, offsetX, offsetY;
        
        if (canvasAspect > videoAspect) {
          displayHeight = canvas.height;
          displayWidth = displayHeight * videoAspect;
          offsetX = (canvas.width - displayWidth) / 2;
          offsetY = 0;
        } else {
          displayWidth = canvas.width;
          displayHeight = displayWidth / videoAspect;
          offsetX = 0;
          offsetY = (canvas.height - displayHeight) / 2;
        }
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        detections.forEach((detection: any) => {
          const [xPercent, yPercent, widthPercent, heightPercent] = detection.bbox;
          const x = offsetX + (xPercent / 100) * displayWidth;
          const y = offsetY + (yPercent / 100) * displayHeight;
          const width = (widthPercent / 100) * displayWidth;
          const height = (heightPercent / 100) * displayHeight;
          
          if (detection.keypoints && detection.keypoints.length > 0) {
            const keypoints = detection.keypoints;
            const skeleton = [
              [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6], [5, 7], [7, 9],
              [6, 8], [8, 10], [5, 6], [5, 11], [6, 12], [11, 12], [11, 13],
              [13, 15], [12, 14], [14, 16],
            ];
            
            ctx.strokeStyle = '#00ff00';
            ctx.lineWidth = 2 * scale;
            skeleton.forEach(([idx1, idx2]) => {
              const kp1 = keypoints[idx1];
              const kp2 = keypoints[idx2];
              if (kp1 && kp2 && kp1.visibility > 0.5 && kp2.visibility > 0.5) {
                ctx.beginPath();
                ctx.moveTo(offsetX + (kp1.x / 100) * displayWidth, offsetY + (kp1.y / 100) * displayHeight);
                ctx.lineTo(offsetX + (kp2.x / 100) * displayWidth, offsetY + (kp2.y / 100) * displayHeight);
                ctx.stroke();
              }
            });
            
            keypoints.forEach((kp: any) => {
              if (kp.visibility > 0.5) {
                ctx.beginPath();
                ctx.arc(offsetX + (kp.x / 100) * displayWidth, offsetY + (kp.y / 100) * displayHeight, 4 * scale, 0, 2 * Math.PI);
                ctx.fillStyle = '#00ff00';
                ctx.fill();
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 1 * scale;
                ctx.stroke();
              }
            });
          } else {
            ctx.strokeStyle = '#ff00ff';
            ctx.lineWidth = 2 * scale;
            ctx.strokeRect(x, y, width, height);
            
            const label = `${detection.class} ${(detection.confidence * 100).toFixed(0)}%`;
            ctx.font = `${12 * scale}px monospace`;
            const textMetrics = ctx.measureText(label);
            const textHeight = 16 * scale;
            
            ctx.fillStyle = '#ff00ff';
            ctx.fillRect(x, y - textHeight - 2, textMetrics.width + 8, textHeight + 2);
            ctx.fillStyle = '#000000';
            ctx.fillText(label, x + 4, y - 6);
          }
        });
      };
      
      drawBoxes();
      return setInterval(drawBoxes, 50);
    };
    
    const interval1 = drawBoxesOnCanvas(canvasRef.current, videoRef.current, 1);
    const interval2 = isExpanded ? drawBoxesOnCanvas(expandedCanvasRef.current, expandedVideoRef.current, 2) : null;
    
    return () => {
      if (interval1) clearInterval(interval1);
      if (interval2) clearInterval(interval2);
    };
  }, [data.detections, isExpanded, inputMode]);
  
  return (
    <>
      <div className="relative">
        <Card className="min-w-56 border-2 border-emerald-500 shadow-lg hover-elevate overflow-hidden">
          {amsError && (
             <div className="p-2 bg-destructive/10 text-destructive text-xs border-b border-destructive/20 font-medium text-center">
               {amsError}
             </div>
          )}
          {isPreviewActive && (stream || videoUrl) && (
            <div className="relative w-full h-32 bg-black flex items-center justify-center">
              <video
                id={`preview-video-${id}`}
                ref={videoRef}
                autoPlay
                muted
                playsInline
                className="w-full h-full object-contain"
              />
              <canvas
                ref={canvasRef}
                className="absolute top-0 left-0 w-full h-full pointer-events-none"
              />
              <div className="absolute top-1 right-1 px-2 py-0.5 bg-red-500 text-white text-xs rounded-md font-medium">
                {inputMode === 'video' ? '● VIDEO' : inputMode === 'antmedia' ? '● AMS' : '● LIVE'}
              </div>
              {data.detections && data.detections.length > 0 && (
                <div className="absolute bottom-1 left-1 px-2 py-0.5 bg-primary text-primary-foreground text-xs rounded-md font-medium">
                  {data.detections.length} detected
                </div>
              )}
            </div>
          )}
          <div className="px-3 py-2 space-y-2">
            <div className="flex items-center gap-2">
              {inputMode === 'antmedia' ? <RadioTower className="w-4 h-4 text-emerald-500" /> : <Camera className="w-4 h-4 text-emerald-500" />}
              <div className="flex-1 min-w-0">
                <div className="font-medium text-sm truncate">{data.label}</div>
                <div className="text-xs text-muted-foreground">
                  {inputMode === 'screen' ? 'Screen Share' : inputMode === 'video' ? 'Upload Video' : inputMode === 'antmedia' ? 'Ant Media' : 'Webcam'}
                </div>
              </div>
            </div>
            {!isPreviewActive ? (
              <button
                onClick={startPreview}
                className="w-full px-2 py-1 text-xs bg-primary text-primary-foreground rounded hover-elevate active-elevate-2"
                data-testid={`button-start-preview-${id}`}
              >
                {inputMode === 'screen' ? 'Start Sharing' : inputMode === 'antmedia' ? 'Connect Stream' : 'Start Preview'}
              </button>
            ) : (
              <button
                onClick={stopPreview}
                className="w-full px-2 py-1 text-xs bg-destructive text-destructive-foreground rounded hover-elevate active-elevate-2"
                data-testid={`button-stop-preview-${id}`}
              >
                {inputMode === 'screen' ? 'Stop Sharing' : inputMode === 'antmedia' ? 'Disconnect AMS' : 'Stop Preview'}
              </button>
            )}
          </div>
        </Card>
        
        {/* Fullscreen button floating above the node */}
        {isPreviewActive && (stream || videoUrl) && inputMode !== 'antmedia' && (
          <button
            onClick={() => setIsExpanded(true)}
            className="absolute -top-3 -right-3 px-3 py-1.5 bg-primary text-primary-foreground rounded-md hover-elevate active-elevate-2 flex items-center gap-1.5 shadow-xl border-2 border-background z-10"
            title="Open fullscreen view"
          >
            <Maximize2 className="w-4 h-4" />
            <span className="font-semibold text-sm">Fullscreen</span>
          </button>
        )}
        
        <Handle type="source" position={Position.Right} className="w-3 h-3" />
      </div>

      {/* Fullscreen Preview Modal (Hidden for AMS to prevent dual-video ID conflicts) */}
      {isExpanded && isPreviewActive && (stream || videoUrl) && inputMode !== 'antmedia' && createPortal(
        <div 
          className="fixed inset-0 z-[9999] bg-black/95 backdrop-blur-sm flex flex-col"
          onClick={() => setIsExpanded(false)}
        >
          <div className="flex items-center justify-between px-6 py-4 bg-background/10 border-b border-white/10">
            <div className="flex items-center gap-3">
              <Camera className="w-5 h-5 text-emerald-500" />
              <div>
                <h2 className="text-lg font-semibold text-white">{data.label}</h2>
                <p className="text-xs text-white/60">Fullscreen View</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="px-3 py-1 bg-red-500 text-white text-sm rounded-md font-medium">
                {inputMode === 'video' ? '● VIDEO' : '● LIVE'}
              </div>
              <button
                onClick={(e) => { e.stopPropagation(); setIsExpanded(false); }}
                className="px-4 py-2 bg-background/90 text-foreground rounded-md hover-elevate active-elevate-2 flex items-center gap-2 border border-border"
              >
                <Minimize2 className="w-4 h-4" />
                <span className="font-medium">Close</span>
              </button>
            </div>
          </div>

          <div className="flex-1 flex items-center justify-center p-8 overflow-hidden" onClick={(e) => e.stopPropagation()}>
            <div className="relative flex items-center justify-center" style={{ maxWidth: '100%', maxHeight: '100%' }}>
              <video
                ref={expandedVideoRef}
                autoPlay muted playsInline
                className="max-w-full max-h-full object-contain rounded-lg shadow-2xl"
                style={{ maxHeight: 'calc(100vh - 160px)' }}
              />
              <canvas ref={expandedCanvasRef} className="absolute top-0 left-0 w-full h-full pointer-events-none" />
            </div>
          </div>
        </div>,
        document.body
      )}
    </>
  );
}

function TranscriptionNode({ data, id }: { data: any; id: string }) {
  const { setNodes } = useReactFlow();
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [transcript, setTranscript] = useState<string>(data.transcript || '');
  const sessionRef = useRef<any>(null);
  const lastSaveTimeRef = useRef<number>(0);
  const transcriptRef = useRef<string>(transcript);
  
  // Keep ref in sync with state
  useEffect(() => {
    transcriptRef.current = transcript;
  }, [transcript]);
  
  // Update node data whenever transcript changes
  useEffect(() => {
    setNodes((nds) =>
      nds.map((node) =>
        node.id === id ? { ...node, data: { ...node.data, transcript } } : node
      )
    );
  }, [transcript, id, setNodes]);

  // Auto-save transcript to results panel immediately when transcript changes
  useEffect(() => {
    if (!transcript) return;

    // Save directly to localStorage
    const flowId = localStorage.getItem('currentFlowId');
    if (!flowId) return;

    try {
      // Get flow info
      const flowsJson = localStorage.getItem('flows');
      const flows = flowsJson ? JSON.parse(flowsJson) : [];
      const currentFlow = flows.find((f: any) => f.id === flowId);
      
      if (!currentFlow) return;

      // Create result object
      const result = {
        id: `result-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        flowId: currentFlow.id,
        flowName: currentFlow.name,
        timestamp: new Date().toISOString(),
        transcript: transcript,
      };

      // Save to localStorage
      const storageKey = `flow_results_${flowId}`;
      const existingJson = localStorage.getItem(storageKey);
      const existing = existingJson ? JSON.parse(existingJson) : [];
      const updated = [result, ...existing].slice(0, 100);
      
      localStorage.setItem(storageKey, JSON.stringify(updated));
      console.log('[TRANSCRIPTION] ✓ Saved transcript to results panel');
    } catch (error) {
      console.error('[TRANSCRIPTION] Save failed:', error);
    }
  }, [transcript]); // Save immediately when transcript changes
  
  const startTranscription = useCallback(async () => {
    try {
      console.log('[TRANSCRIPTION] Starting Gemini Live API transcription...');
      
      // Dynamically import the transcription client
      const { GeminiLiveTranscription } = await import('@/lib/geminiLiveTranscription');
      
      const client = new GeminiLiveTranscription();
      const session = await client.createSession();
      sessionRef.current = session;
      
      // Set up transcript callback
      session.onTranscript((text: string) => {
        console.log('[TRANSCRIPTION] Received transcript:', text);
        setTranscript(prev => {
          const newTranscript = prev ? `${prev}\n${text}` : text;
          return newTranscript;
        });
      });
      
      // Start the session
      await session.start();
      setIsTranscribing(true);
      
      console.log('[TRANSCRIPTION] Started Gemini Live transcription');
    } catch (error) {
      console.error('[TRANSCRIPTION] Failed to start transcription:', error);
      alert('Failed to start transcription. Please grant microphone permission and check server logs.');
      setIsTranscribing(false);
    }
  }, []);
  
  const stopTranscription = useCallback(() => {
    console.log('[TRANSCRIPTION] Stopping transcription...');
    
    if (sessionRef.current) {
      sessionRef.current.stop();
      sessionRef.current = null;
    }
    
    setIsTranscribing(false);
    console.log('[TRANSCRIPTION] Stopped transcription');
    console.log('[TRANSCRIPTION] Final transcript:', transcript);
  }, [transcript]);
  
  // Listen for auto-start/stop triggers from flow execution
  useEffect(() => {
    if (data.triggerStart && !isTranscribing) {
      console.log('[TRANSCRIPTION] Auto-start triggered by flow execution');
      startTranscription();
    }
  }, [data.triggerStart, isTranscribing, startTranscription]);

  useEffect(() => {
    if (data.triggerStop && isTranscribing) {
      console.log('[TRANSCRIPTION] Auto-stop triggered by flow execution');
      stopTranscription();
    }
  }, [data.triggerStop, isTranscribing, stopTranscription]);
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (sessionRef.current) {
        sessionRef.current.stop();
      }
    };
  }, []);
  
  return (
    <div>
      <Handle type="target" position={Position.Left} className="w-3 h-3" />
      <Card className="min-w-56 border-2 border-purple-500 shadow-lg hover-elevate overflow-hidden">
        <div className="px-3 py-2 space-y-2">
          <div className="flex items-center gap-2">
            <Mic className="w-4 h-4 text-purple-500" />
            <div className="flex-1 min-w-0">
              <div className="font-medium text-sm truncate">{data.label}</div>
              <div className="text-xs text-muted-foreground">Microphone</div>
            </div>
          </div>
          
          {isTranscribing && (
            <div className="px-2 py-1 bg-red-500/10 border border-red-500/30 rounded text-xs text-red-500 flex items-center gap-1">
              <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
              Recording
            </div>
          )}
          
          {!isTranscribing ? (
            <button
              onClick={startTranscription}
              className="w-full px-2 py-1 text-xs bg-purple-500 text-white rounded hover-elevate active-elevate-2"
              data-testid={`button-start-transcription-${id}`}
            >
              Start Transcription
            </button>
          ) : (
            <button
              onClick={stopTranscription}
              className="w-full px-2 py-1 text-xs bg-destructive text-destructive-foreground rounded hover-elevate active-elevate-2"
              data-testid={`button-stop-transcription-${id}`}
            >
              Stop Transcription
            </button>
          )}
          
          {transcript && (
            <div className="text-xs text-muted-foreground max-h-20 overflow-y-auto">
              {transcript}
            </div>
          )}
        </div>
      </Card>
      <Handle type="source" position={Position.Right} className="w-3 h-3" />
    </div>
  );
}

function DetectionNode({ data }: { data: any }) {
  return (
    <div>
      <Handle type="target" position={Position.Left} className="w-3 h-3" />
      <Card className="min-w-48 px-4 py-3 border-2 border-cyan-500 shadow-lg hover-elevate">
        <div className="flex items-center gap-3">
          <Search className="w-5 h-5 text-cyan-500" />
          <div>
            <div className="font-medium text-sm">{data.label}</div>
            <div className="text-xs text-muted-foreground">Gemini Detection</div>
          </div>
        </div>
      </Card>
      <Handle type="source" position={Position.Right} className="w-3 h-3" />
    </div>
  );
}

function PoseNode({ data }: { data: any }) {
  return (
    <div>
      <Handle type="target" position={Position.Left} className="w-3 h-3" />
      <Card className="min-w-48 px-4 py-3 border-2 border-pink-500 shadow-lg hover-elevate">
        <div className="flex items-center gap-3">
          <User className="w-5 h-5 text-pink-500" />
          <div>
            <div className="font-medium text-sm">{data.label}</div>
            <div className="text-xs text-muted-foreground">Pose Detection</div>
          </div>
        </div>
        {data.detections && data.detections.length > 0 && (
          <div className="mt-2 text-xs text-muted-foreground">
            {data.detections.length} pose{data.detections.length > 1 ? 's' : ''} detected
          </div>
        )}
      </Card>
      <Handle type="source" position={Position.Right} className="w-3 h-3" />
    </div>
  );
}

function AnalysisNode({ data }: { data: any }) {
  return (
    <div>
      <Handle type="target" position={Position.Left} className="w-3 h-3" />
      <Card className="min-w-48 px-4 py-3 border-2 border-blue-500 shadow-lg hover-elevate">
        <div className="flex items-center gap-3">
          <Brain className="w-5 h-5 text-blue-500" />
          <div>
            <div className="font-medium text-sm">{data.label}</div>
            <div className="text-xs text-muted-foreground">Gemini Analysis</div>
          </div>
        </div>
      </Card>
      <Handle type="source" position={Position.Right} className="w-3 h-3" />
    </div>
  );
}

function ActionNode({ data }: { data: any }) {
  return (
    <div>
      <Handle type="target" position={Position.Left} className="w-3 h-3" />
      <Card className="min-w-48 px-4 py-3 border-2 border-orange-500 shadow-lg hover-elevate">
        <div className="flex items-center gap-3">
          <Play className="w-5 h-5 text-orange-500" />
          <div>
            <div className="font-medium text-sm">{data.label}</div>
            <div className="text-xs text-muted-foreground">Execute Action</div>
          </div>
        </div>
      </Card>
    </div>
  );
}

function ApprovalNode({ data }: { data: any }) {
  return (
    <div>
      <Handle type="target" position={Position.Left} className="w-3 h-3" />
      <Card className="min-w-48 px-4 py-3 border-2 border-green-500 shadow-lg hover-elevate">
        <div className="flex items-center gap-3">
          <CheckCircle className="w-5 h-5 text-green-500" />
          <div>
            <div className="font-medium text-sm">{data.label}</div>
            <div className="text-xs text-muted-foreground">Human Approval</div>
          </div>
        </div>
      </Card>
      <Handle type="source" position={Position.Right} className="w-3 h-3" />
    </div>
  );
}

function EmailNode({ data }: { data: any }) {
  return (
    <div>
      <Handle type="target" position={Position.Left} className="w-3 h-3" />
      <Card className="min-w-48 px-4 py-3 border-2 border-red-500 shadow-lg hover-elevate">
        <div className="flex items-center gap-3">
          <Mail className="w-5 h-5 text-red-500" />
          <div>
            <div className="font-medium text-sm">{data.label}</div>
            <div className="text-xs text-muted-foreground">Send Email</div>
          </div>
        </div>
      </Card>
    </div>
  );
}

function SMSNode({ data }: { data: any }) {
  return (
    <div>
      <Handle type="target" position={Position.Left} className="w-3 h-3" />
      <Card className="min-w-48 px-4 py-3 border-2 border-yellow-500 shadow-lg hover-elevate">
        <div className="flex items-center gap-3">
          <MessageSquare className="w-5 h-5 text-yellow-500" />
          <div>
            <div className="font-medium text-sm">{data.label}</div>
            <div className="text-xs text-muted-foreground">Send SMS</div>
          </div>
        </div>
      </Card>
    </div>
  );
}

function CallNode({ data }: { data: any }) {
  return (
    <div>
      <Handle type="target" position={Position.Left} className="w-3 h-3" />
      <Card className="min-w-48 px-4 py-3 border-2 border-indigo-500 shadow-lg hover-elevate">
        <div className="flex items-center gap-3">
          <Phone className="w-5 h-5 text-indigo-500" />
          <div>
            <div className="font-medium text-sm">{data.label}</div>
            <div className="text-xs text-muted-foreground">Make Call</div>
          </div>
        </div>
      </Card>
    </div>
  );
}

function DiscordNode({ data }: { data: any }) {
  return (
    <div>
      <Handle type="target" position={Position.Left} className="w-3 h-3" />
      <Card className="min-w-48 px-4 py-3 border-2 border-violet-500 shadow-lg hover-elevate">
        <div className="flex items-center gap-3">
          <Send className="w-5 h-5 text-violet-500" />
          <div>
            <div className="font-medium text-sm">{data.label}</div>
            <div className="text-xs text-muted-foreground">Discord Message</div>
          </div>
        </div>
      </Card>
    </div>
  );
}

export function FlowCanvas({
  initialNodes = [],
  initialEdges = [],
  onNodesChange,
  onEdgesChange,
  onNodeClick,
  onPaneClick,
  onNodeAdd,
}: FlowCanvasProps) {
  // Use React Flow's hooks for proper state management
  const [nodes, setNodes, handleNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, handleEdgesChange] = useEdgesState(initialEdges);
  const [reactFlowInstance, setReactFlowInstance] = useState<any>(null);

  // Sync with parent when nodes change
  useEffect(() => {
    setNodes(initialNodes);
  }, [initialNodes, setNodes]);

  // Sync with parent when edges change
  useEffect(() => {
    setEdges(initialEdges);
  }, [initialEdges, setEdges]);

  // Wrap React Flow's handlers to also notify parent
  const onNodesChangeWrapper = useCallback(
    (changes: any) => {
      handleNodesChange(changes);
      // Notify parent after state updates
      setTimeout(() => {
        setNodes((currentNodes) => {
          onNodesChange?.(currentNodes);
          return currentNodes;
        });
      }, 0);
    },
    [handleNodesChange, onNodesChange, setNodes]
  );

  const onEdgesChangeWrapper = useCallback(
    (changes: any) => {
      handleEdgesChange(changes);
      // Notify parent after state updates
      setTimeout(() => {
        setEdges((currentEdges) => {
          onEdgesChange?.(currentEdges);
          return currentEdges;
        });
      }, 0);
    },
    [handleEdgesChange, onEdgesChange, setEdges]
  );

  const onConnect = useCallback(
    (params: Connection | Edge) => {
      setEdges((eds) => {
        const newEdges = addEdge(params, eds);
        onEdgesChange?.(newEdges);
        return newEdges;
      });
    },
    [setEdges, onEdgesChange]
  );

  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const handleDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      const type = event.dataTransfer.getData('application/reactflow');
      if (!type || !reactFlowInstance) return;

      // Convert screen coordinates to flow coordinates (accounts for zoom/pan)
      const position = reactFlowInstance.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      const newNode: Node = {
        id: `${type}-${Date.now()}`,
        type,
        position,
        data: { 
          label: `${type.charAt(0).toUpperCase() + type.slice(1)} Node`,
          config: type === 'analysis' ? { userPrompt: 'Analyze this image' } : {}
        },
      };

      setNodes((nds) => {
        const updatedNodes = nds.concat(newNode);
        onNodesChange?.(updatedNodes);
        return updatedNodes;
      });

      // Notify parent about node addition
      onNodeAdd?.(type);
    },
    [reactFlowInstance, setNodes, onNodesChange, onNodeAdd]
  );

  return (
    <div className="w-full h-full bg-[hsl(220,15%,8%)]" data-testid="flow-canvas">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChangeWrapper}
        onEdgesChange={onEdgesChangeWrapper}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onInit={setReactFlowInstance}
        nodeTypes={nodeTypes}
        fitView
        className="bg-[hsl(220,15%,8%)]"
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="hsl(220, 10%, 20%)" />
      </ReactFlow>
    </div>
  );
}
