// client\src\pages\FlowEditor.tsx

import { useState, useCallback, useEffect, useRef } from 'react';
import { Play, Save, StopCircle, ArrowLeft, Send, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { FlowCanvas } from '@/components/FlowCanvas';
import { NodeConfigPanel } from '@/components/NodeConfigPanel';
import { NodePalette } from '@/components/NodePalette';
import { FlowResults } from '@/components/FlowResults';
import type { FlowNode, FlowEdge } from '@shared/schema';
import { Node, Edge } from '@xyflow/react';
import { useToast } from '@/hooks/use-toast';
import { apiRequest } from '@/lib/queryClient';
import { useLocation } from 'wouter';
import { getFlow, saveFlow, setCurrentFlowId, createDefaultFlow } from '@/lib/flowStorage';
import { saveFlowResult } from '@/lib/resultsStorage';
import { WebRTCAdaptor } from '@antmedia/webrtc_adaptor';

export default function FlowEditor() {
  const [location, setLocation] = useLocation();
  const { toast } = useToast();
  
  // Get flow ID from URL query parameters
  const urlParams = new URLSearchParams(window.location.search);
  const flowIdFromUrl = urlParams.get('id');
  
  const [currentFlowId, setCurrentFlowIdState] = useState<string | null>(null);
  const [flowName, setFlowName] = useState('AI Automation Flow');
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [executionLog, setExecutionLog] = useState<string[]>([]);
  const [flowData, setFlowData] = useState<any>({});
  const [isFlowLoaded, setIsFlowLoaded] = useState(false);
  const [aiChatInput, setAiChatInput] = useState('');
  const [isEditingFlow, setIsEditingFlow] = useState(false);
  
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const uploadedVideoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const lastSaveTime = useRef<number>(0);
  const captureIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const isRunningRef = useRef(false);
  const isProcessingFrameRef = useRef(false);
  const currentExecutionIdRef = useRef<string | null>(null);
  const nodesRef = useRef<Node[]>(nodes);
  
  const frameSequenceRef = useRef<string[]>([]);
  const webRTCAdaptorRef = useRef<any>(null);

  // Batching mechanism for AI analysis
  const previousDetectionsRef = useRef<string[]>([]);
  const lastAnalysisTimeRef = useRef<number>(0);
  const ANALYSIS_COOLDOWN = 5000;
  
  // Keep nodesRef in sync with nodes state
  useEffect(() => {
    nodesRef.current = nodes;
  }, [nodes]);

  // Load flow from localStorage on mount
  useEffect(() => {
    if (flowIdFromUrl) {
      const flow = getFlow(flowIdFromUrl);
      if (flow) {
        console.log('[FLOW_LOAD] Loading flow:', flow.id, flow.name);
        setCurrentFlowIdState(flow.id);
        setFlowName(flow.name);
        
        // Preserve runtime data (callbacks, videoUrl) but NOT streams (can't be serialized)
        setNodes(prevNodes => {
          // Create a map of node IDs to their runtime data (callbacks, videoUrl, etc.)
          const runtimeDataMap = new Map(
            prevNodes.map(node => [
              node.id,
              {
                onStreamChange: node.data.onStreamChange,
                videoUrl: node.data.videoUrl,
                // Note: We don't preserve stream or detections - they should be re-created
              }
            ])
          );
          
          // Merge loaded nodes with preserved runtime data
          return flow.nodes.map(node => {
            const runtimeData = runtimeDataMap.get(node.id);
            
            // Resolve video URL from server filename or asset ID (for camera nodes with uploaded videos)
            let videoUrl = node.data.videoUrl;
            if (node.type === 'camera' && !videoUrl) {
              // Prefer server filename for direct file access (persists across server restarts)
              if (node.data.config?.videoServerFileName) {
                videoUrl = `/api/media/videos/${node.data.config.videoServerFileName}`;
                console.log('[FLOW_LOAD] Resolved video URL from server filename:', node.data.config.videoServerFileName);
              }
              // Fallback to asset ID (only works within current session)
              else if (node.data.config?.videoAssetId) {
                videoUrl = `/api/media/videos/${node.data.config.videoAssetId}`;
                console.log('[FLOW_LOAD] Resolved video URL from asset ID:', node.data.config.videoFileName);
              }
            }
            
            if (runtimeData) {
              return {
                ...node,
                data: {
                  ...node.data,
                  ...runtimeData,
                  videoUrl: videoUrl || runtimeData.videoUrl,
                }
              };
            }
            return {
              ...node,
              data: {
                ...node.data,
                videoUrl,
              }
            };
          });
        });
        
        setEdges(flow.edges);
        setCurrentFlowId(flow.id);
        setIsFlowLoaded(true);
      } else {
        console.warn('[FLOW_LOAD] Flow not found, creating new:', flowIdFromUrl);
        const newFlow = createDefaultFlow('New Flow');
        setCurrentFlowIdState(newFlow.id);
        setFlowName(newFlow.name);
        setNodes(newFlow.nodes);
        setEdges(newFlow.edges);
        setCurrentFlowId(newFlow.id);
        setIsFlowLoaded(true);
        // Update URL to the new flow ID (fix for missing flow)
        window.history.replaceState({}, '', `/flow-editor?id=${newFlow.id}`);
        // Notify user that the requested flow wasn't found
        toast({
          title: 'Flow Not Found',
          description: 'The requested flow could not be found. A new flow has been created.',
          variant: 'destructive',
        });
      }
    } else {
      // No flow ID in URL, create a new flow
      console.log('[FLOW_LOAD] No flow ID in URL, creating new flow');
      const newFlow = createDefaultFlow('New Flow');
      setCurrentFlowIdState(newFlow.id);
      setFlowName(newFlow.name);
      setNodes(newFlow.nodes);
      setEdges(newFlow.edges);
      setCurrentFlowId(newFlow.id);
      setIsFlowLoaded(true);
      // Update URL to include the flow ID
      window.history.replaceState({}, '', `/flow-editor?id=${newFlow.id}`);
    }
  }, [flowIdFromUrl]);

  // Save execution state to sessionStorage when it changes
  // But skip the first save on mount to avoid overwriting existing running state
  const hasRestoredRef = useRef(false);
  
  useEffect(() => {
    if (currentFlowId && hasRestoredRef.current) {
      sessionStorage.setItem(`flow_${currentFlowId}_running`, JSON.stringify(isRunning));
      sessionStorage.setItem(`flow_${currentFlowId}_executionId`, currentExecutionIdRef.current || '');
      
      console.log('[SESSION_SAVE] Saved flow state:', {
        flowId: currentFlowId,
        isRunning,
        executionId: currentExecutionIdRef.current
      });
    }
  }, [isRunning, currentFlowId]);

  // Track if we need to auto-resume a video flow
  const [shouldAutoResume, setShouldAutoResume] = useState(false);
  
  // Restore or cleanup execution state from previous session
  useEffect(() => {
    if (currentFlowId && isFlowLoaded && nodes.length > 0) {
      const wasRunning = sessionStorage.getItem(`flow_${currentFlowId}_running`);
      const executionId = sessionStorage.getItem(`flow_${currentFlowId}_executionId`);
      
      // Check the actual camera node configuration from the loaded flow
      const cameraNode = nodes.find(n => n.type === 'camera');
      const isVideoUploadMode = cameraNode?.data?.config?.inputMode === 'video';
      
      console.log('[FLOW_CHECK] Flow restoration check:', {
        flowId: currentFlowId,
        wasRunning,
        executionId,
        isVideoUploadMode,
        cameraInputMode: cameraNode?.data?.config?.inputMode
      });
      
      if (wasRunning === 'true') {
        if (isVideoUploadMode) {
          // Restore running state for video upload flows by actually restarting execution
          console.log('[FLOW_RESTORE] Resuming video upload flow execution');
          
          // Preserve the execution ID if it exists
          if (executionId) {
            currentExecutionIdRef.current = executionId;
          }
          
          // Trigger auto-resume (will call handleRun in separate effect)
          setShouldAutoResume(true);
        } else {
          // Clean up stale state for webcam/screen flows (they were stopped by navigation)
          console.log('[FLOW_CLEANUP] Clearing stale state for webcam/screen flow (cannot persist)');
          sessionStorage.removeItem(`flow_${currentFlowId}_running`);
          sessionStorage.removeItem(`flow_${currentFlowId}_executionId`);
        }
      }
      
      // Mark that we've completed the restoration check
      hasRestoredRef.current = true;
    }
  }, [currentFlowId, isFlowLoaded, nodes, toast]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      // Cleanup capture interval
      if (captureIntervalRef.current) {
        console.log('[CLEANUP] Clearing capture timeout on unmount');
        clearTimeout(captureIntervalRef.current);
        captureIntervalRef.current = null;
      }
    };
  }, []); // Empty deps - only run on mount/unmount

  const log = useCallback((message: string) => {
    setExecutionLog(prev => [...prev, `${new Date().toLocaleTimeString()}: ${message}`]);
  }, []);

  // Throttled save to prevent excessive writes
  const saveCurrentFlow = useCallback((data: { name: string; nodes: Node[]; edges: Edge[] }) => {
    if (!currentFlowId || !isFlowLoaded) {
      console.log('[FLOW_SAVE] Skipped - flow not loaded yet');
      return;
    }
    
    const now = Date.now();
    const timeSinceLastSave = now - lastSaveTime.current;
    
    // Only save if at least 100ms has passed since last save (10 saves/second max)
    if (timeSinceLastSave >= 100) {
      console.log('[FLOW_SAVE] Auto-saving flow:', currentFlowId);
      saveFlow({
        id: currentFlowId,
        name: data.name,
        nodes: data.nodes,
        edges: data.edges,
      });
      lastSaveTime.current = now;
    } else {
      console.log('[FLOW_SAVE] Skipped (throttled), time since last save:', timeSinceLastSave, 'ms');
    }
  }, [currentFlowId, isFlowLoaded]);

  // We don't need a separate useEffect for auto-save!
  // Auto-saves already happen in:
  // - handleNodeUpdate (when node config changes)
  // - handleNodesChange (when nodes are moved/deleted)
  // - handleEdgesChange (when edges are added/deleted)
  // - handleFlowNameChange (when flow name changes)

  // Handle camera stream changes from node preview
  const handleStreamChange = useCallback((nodeId: string, stream: MediaStream | null) => {
    setNodes(prevNodes => prevNodes.map(n => 
      n.id === nodeId 
        ? { ...n, data: { ...n.data, stream } }
        : n
    ));
    // Update streamRef for flow execution - only if it's a valid MediaStream
    if (stream && stream instanceof MediaStream) {
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play().catch(() => {});
      }
    }
  }, []);

  // Add onStreamChange to all camera nodes
  useEffect(() => {
    setNodes(prevNodes => prevNodes.map(n => 
      n.type === 'camera' 
        ? { ...n, data: { ...n.data, onStreamChange: handleStreamChange } }
        : n
    ));
  }, [handleStreamChange]);

  const handleSave = useCallback(() => {
    if (!currentFlowId) return;
    
    saveFlow({
      id: currentFlowId,
      name: flowName,
      nodes,
      edges,
    });
    toast({
      title: 'Flow Saved',
      description: 'Your workflow has been saved successfully',
    });
  }, [currentFlowId, flowName, nodes, edges, toast]);

  const handleAiEditFlow = useCallback(async () => {
    if (!aiChatInput.trim() || isEditingFlow) return;

    setIsEditingFlow(true);
    
    try {
      const currentFlow = {
        name: flowName,
        nodes: nodes.map(n => ({
          id: n.id,
          type: n.type,
          position: n.position,
          data: {
            label: n.data.label,
            config: n.data.config || {}
          }
        })),
        edges: edges.map(e => ({
          id: e.id,
          source: e.source,
          target: e.target
        }))
      };

      const response = await apiRequest('POST', '/api/edit-flow', {
        prompt: aiChatInput,
        currentFlow
      });
      
      const result = await response.json();

      if (result.success && result.flow) {
        // Apply the edited flow
        setFlowName(result.flow.name || flowName);
        setNodes(result.flow.nodes.map((n: any) => ({
          ...n,
          data: {
            ...n.data,
            onStreamChange: n.type === 'camera' ? handleStreamChange : undefined
          }
        })));
        setEdges(result.flow.edges);
        
        // Auto-save the edited flow
        if (currentFlowId) {
          saveFlow({
            id: currentFlowId,
            name: result.flow.name || flowName,
            nodes: result.flow.nodes,
            edges: result.flow.edges
          });
        }

        toast({
          title: 'Flow Updated',
          description: 'AI successfully edited your flow',
        });
        setAiChatInput('');
      } else {
        toast({
          title: 'Edit Failed',
          description: result.error || 'Could not edit flow',
          variant: 'destructive'
        });
      }
    } catch (error) {
      console.error('[AI-EDIT] Error:', error);
      toast({
        title: 'Error',
        description: 'Failed to edit flow with AI',
        variant: 'destructive'
      });
    } finally {
      setIsEditingFlow(false);
    }
  }, [aiChatInput, isEditingFlow, flowName, nodes, edges, currentFlowId, handleStreamChange, toast]);

  const executeCameraNode = useCallback(async (node: Node) => {
    log(`Executing camera node: ${node.id}`);
    
    try {
      const inputMode = node.data.config?.inputMode || 'webcam';
      
      // Handle video file upload mode
      if (inputMode === 'video') {
        const videoUrl = node.data.videoUrl;
        if (!videoUrl) throw new Error('No video file uploaded');
        
        if (uploadedVideoRef.current) {
          uploadedVideoRef.current.src = videoUrl;
          await uploadedVideoRef.current.play();
          await new Promise(resolve => setTimeout(resolve, 500));
          log(`Video file loaded: ${node.data.config?.videoFileName || 'video file'}`);
        }
        
        if (uploadedVideoRef.current && canvasRef.current) {
          const video = uploadedVideoRef.current;
          const canvas = canvasRef.current;
          canvas.width = 640; canvas.height = 360;
          const ctx = canvas.getContext('2d');
          if (ctx) {
            ctx.drawImage(video, 0, 0, 640, 360);
            return { imageData: canvas.toDataURL('image/jpeg', 0.7), isVideoMode: true };
          }
        }
        throw new Error('Failed to capture video frame');
      }
      
      // --- NEW: Handle Ant Media Server Mode ---
      if (inputMode === 'antmedia') {
        const amsUrl = node.data.config?.amsUrl || 'ws://localhost:5080/LiveApp/websocket';
        const streamId = node.data.config?.streamId || 'stream1';

        if (!webRTCAdaptorRef.current) {
          log(`Connecting to Ant Media Server stream: ${streamId}...`);
          
          await new Promise((resolve, reject) => {
            webRTCAdaptorRef.current = new WebRTCAdaptor({
              websocket_url: amsUrl,
              mediaConstraints: { video: false, audio: false }, // We are receiving, not sending
              peerconnection_config: { iceServers: [{ urls: 'stun:stun1.l.google.com:19302' }] },
              sdp_constraints: { OfferToReceiveAudio: false, OfferToReceiveVideo: true },
              remoteVideoId: 'flow-video-player', // Targets the hidden video tag we updated
              callback: (info: string, obj: any) => {
                if (info === 'initialized') {
                  webRTCAdaptorRef.current.play(streamId);
                } else if (info === 'play_started') {
                  log('Ant Media stream connected successfully!');
                  
                  // Create a mock stream to satisfy the rest of the app's state checks
                  const mockStream = new MediaStream();
                  streamRef.current = mockStream;
                  setNodes(prevNodes => prevNodes.map(n => 
                    n.id === node.id ? { ...n, data: { ...n.data, stream: mockStream } } : n
                  ));
                  
                  resolve(true);
                } else if (info === 'closed') {
                  log('Ant Media connection closed');
                }
              },
              callbackError: (error: string) => {
                log(`AMS Error: ${error}`);
                reject(new Error(`Ant Media Error: ${error}`));
              }
            });
          });
          
          // Wait a moment for the first frame to render
          await new Promise(resolve => setTimeout(resolve, 500));
        }
      } 
      // --- End Ant Media Server Mode ---
      
      // Handle standard webcam/screen share modes
      else {
        if (node.data.stream && node.data.stream instanceof MediaStream) {
          log('Using existing camera preview stream');
          streamRef.current = node.data.stream;
          if (videoRef.current) {
            videoRef.current.srcObject = node.data.stream;
            await videoRef.current.play();
          }
        } else if (!streamRef.current) {
          let stream;
          if (inputMode === 'screen') {
            stream = await navigator.mediaDevices.getDisplayMedia({ video: { width: 640, height: 360 } });
            log('Screen share started');
          } else {
            stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 360 } });
            log('Camera started');
          }
          streamRef.current = stream;
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            await videoRef.current.play();
          }
          setNodes(prevNodes => prevNodes.map(n => n.id === node.id ? { ...n, data: { ...n.data, stream } } : n));
        }
        await new Promise(resolve => setTimeout(resolve, 500));
      }

      // Universal Frame Capture (Works for Webcam, Screen, AND Ant Media!)
      if (videoRef.current && canvasRef.current) {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        canvas.width = 640;
        canvas.height = 360;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(video, 0, 0, 640, 360);
          const imageData = canvas.toDataURL('image/jpeg', 0.7);
          log(`Image captured (${Math.round(imageData.length / 1024)}KB)`);
          return { imageData };
        }
      }
      throw new Error('Failed to capture image');
    } catch (error: any) {
      log(`Camera error: ${error.message}`);
      throw error;
    }
  }, [log]);

  const executeDetectionNode = useCallback(async (node: Node, inputData: any) => {
    log(`Executing detection node: ${node.id}`);
    
    try {
      const objectFilter = node.data.config?.objectFilter || [];
      // Motion tracking only works when exactly one object class is selected
      const useMotionTracking = Array.isArray(objectFilter) && objectFilter.length === 1 && currentExecutionIdRef.current;
      
      let detections = [];
      let motion_state = undefined;
      let velocity = undefined;
      let should_analyze = false;
      
      if (useMotionTracking) {
        // Use motion-tracking endpoint (only works for single object)
        const targetClass = objectFilter[0];
        log(`  Using motion tracking for: ${targetClass}`);
        
        // Convert base64 to blob for FormData
        const base64Data = inputData.imageData.split(',')[1];
        const byteCharacters = atob(base64Data);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: 'image/jpeg' });
        
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');
        formData.append('execution_id', currentExecutionIdRef.current!);
        formData.append('target_class', targetClass);
        
        const response = await fetch('http://localhost:8001/api/detect-stream', {
          method: 'POST',
          body: formData,
        });
        
        if (!response.ok) {
          throw new Error(`Detection failed: ${response.statusText}`);
        }
        
        const data = await response.json();
        detections = data.detections || [];
        motion_state = data.motion_state;
        velocity = data.velocity;
        should_analyze = data.should_analyze || false;
        
        log(`  Motion state: ${motion_state}, Velocity: ${velocity}%/s, Should analyze: ${should_analyze}`);
      } else {
        // Use standard detection endpoint
        const response = await apiRequest('POST', '/api/execute-flow', {
          imageData: inputData.imageData,
          userPrompt: 'detect objects',
          outputFormat: 'json',
          stepOnly: 'detection',
        });
        
        const data = await response.json();
        detections = data.detections || [];
      }
      
      // Apply object filter if needed (for standard detection)
      if (!useMotionTracking && objectFilter && Array.isArray(objectFilter) && objectFilter.length > 0) {
        const beforeFilterCount = detections.length;
        const allClasses = detections.map((d: any) => d.class).join(', ');
        detections = detections.filter((d: any) => objectFilter.includes(d.class));
        log(`  Before filter: ${beforeFilterCount} objects [${allClasses}]`);
        log(`  After filter: ${detections.length} objects (showing only: ${objectFilter.join(', ')})`);
      }
      
      const result = { 
        ...inputData, 
        detections,
        motion_state,
        velocity,
        should_analyze
      };
      log(`Final detections: ${detections.length} objects${detections.length > 0 ? ' [' + detections.map((d: any) => d.class).join(', ') + ']' : ''}`);
      
      // Update camera node with detections for visualization
      setNodes(prev => prev.map(n => 
        n.type === 'camera' ? { ...n, data: { ...n.data, detections } } : n
      ));
      
      return result;
    } catch (error: any) {
      log(`Detection error: ${error.message}`);
      throw error;
    }
  }, [log]);

  const executePoseNode = useCallback(async (node: Node, inputData: any) => {
    log(`Executing pose detection node: ${node.id}`);
    
    try {
      // Convert base64 to blob for FormData
      // Extract MIME type from data URI (e.g., "data:image/jpeg;base64,...")
      const mimeMatch = inputData.imageData.match(/^data:(image\/[a-z]+);base64,/);
      const mimeType = mimeMatch ? mimeMatch[1] : 'image/jpeg';
      
      const base64Data = inputData.imageData.split(',')[1];
      const byteCharacters = atob(base64Data);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: mimeType });
      
      const formData = new FormData();
      formData.append('file', blob, 'frame.jpg');
      
      // Use proxy route for Replit webview compatibility
      const response = await fetch('/api/detect-pose', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Pose detection failed: ${response.statusText}`);
      }
      
      const data = await response.json();
      const detections = data.detections || [];
      
      log(`Detected ${detections.length} pose${detections.length > 1 ? 's' : ''}`);
      if (detections.length > 0) {
        detections.forEach((det: any, idx: number) => {
          const visibleKeypoints = det.keypoints?.filter((kp: any) => kp.visibility > 0.5).length || 0;
          log(`  Pose ${idx + 1}: ${visibleKeypoints} visible keypoints (confidence: ${det.confidence})`);
        });
      }
      
      const result = { 
        ...inputData, 
        detections
      };
      
      // Update camera node with pose detections for visualization
      setNodes(prev => prev.map(n => 
        n.type === 'camera' ? { ...n, data: { ...n.data, detections } } : n
      ));
      
      return result;
    } catch (error: any) {
      log(`Pose detection error: ${error.message}`);
      throw error;
    }
  }, [log]);

  const executeTranscriptionNode = useCallback(async (node: Node) => {
    log(`Executing transcription node: ${node.id}`);
    
    // Extract transcript from node data
    const transcript = node.data.transcript || '';
    
    if (!transcript) {
      log('No transcript available yet - waiting for audio capture');
      return { transcript: '' };
    }
    
    log(`Transcript length: ${transcript.length} characters`);
    return { transcript };
  }, [log]);

  const executeAnalysisNode = useCallback(async (node: Node, inputData: any) => {
    log(`Executing analysis node: ${node.id}`);
    
    const userPrompt = node.data.config?.userPrompt || 'Analyze this image';
    
    // Check if we have transcript data (transcript-based flow) or image data (visual flow)
    if (inputData.transcript) {
      // Transcript-based analysis
      log('Analyzing transcript with Gemini...');
      try {
        const response = await apiRequest('POST', '/api/analyze-transcript', {
          transcript: inputData.transcript,
          userPrompt,
        });
        
        const data = await response.json();
        const analysis = data.analysis || null;
        
        log(`Analysis: ${analysis?.summary?.substring(0, 100) || 'No summary'}...`);
        return { ...inputData, analysis };
      } catch (error: any) {
        log(`Transcript analysis error: ${error.message}`);
        throw error;
      }
    } else {
      // Image-based analysis (existing logic)
      const detections = inputData.detections || [];
      const motion_state = inputData.motion_state;
      const velocity = inputData.velocity;
      const should_analyze = inputData.should_analyze;
      
      // Check if this is a video upload mode flow (demonstration flow)
      const cameraNode = nodes.find(n => n.type === 'camera');
      const isVideoDemo = cameraNode?.data?.config?.inputMode === 'video';
      
      // If using motion tracking, only analyze when explicitly requested
      if (motion_state !== undefined && !should_analyze) {
        log(`  Skipping analysis (motion state: ${motion_state}, waiting for stationary state)`);
        return { ...inputData, analysis: null };
      }
      
      try {
        const response = await apiRequest('POST', '/api/execute-flow', {
          imageData: inputData.imageData,
          imagesData: inputData.imagesData,
          userPrompt,
          outputFormat: 'json',
          stepOnly: 'analysis',
          detections,
          motion_state,
          velocity,
          execution_id: currentExecutionIdRef.current,
          isDemoFlow: isVideoDemo, // Pass demo flow flag to backend
        });
        
        const data = await response.json();
        const analysis = data.analysis || null;
        const result = { ...inputData, analysis };
        
        if (motion_state) {
          log(`Analysis (motion: ${motion_state}): ${analysis?.summary || 'No summary'} (condition: ${analysis?.condition_met ? 'MET' : 'NOT MET'})`);
        } else {
          log(`Analysis: ${analysis?.summary || 'No summary'} (condition: ${analysis?.condition_met ? 'MET' : 'NOT MET'})`);
        }
        
        return result;
      } catch (error: any) {
        log(`Analysis error: ${error.message}`);
        throw error;
      }
    }
  }, [log]);

  const executeSaveNode = useCallback(async (node: Node, inputData: any) => {
    log(`Executing save node: ${node.id}`);
    
    const outputFormat = node.data.config?.outputFormat || 'json';
    const detections = inputData.detections || [];
    const analysis = inputData.analysis || null;
    
    log(`Save data: ${detections.length} detections, analysis: ${analysis ? 'yes' : 'no'}`);
    
    try {
      const response = await apiRequest('POST', '/api/execute-flow', {
        imageData: inputData.imageData,
        userPrompt: node.data.config?.userPrompt || 'Pipeline results',
        outputFormat,
        stepOnly: 'save',
        detections,
        analysis,
      });
      
      const data = await response.json();
      log(`Results saved to ${data.saved?.filepath || 'file'}`);
      return { ...inputData, saved: data.saved };
    } catch (error: any) {
      log(`Save error: ${error.message}`);
      throw error;
    }
  }, [log]);

  const executeEmailNode = useCallback(async (node: Node, inputData: any) => {
    log(`Executing email node: ${node.id}`);
    
    const { to, subject, body } = node.data.config || {};
    const analysis = inputData.analysis || null;
    const transcript = inputData.transcript || '';
    
    console.log('[EMAIL_NODE_DEBUG] Config:', { to, subject, body, fullConfig: node.data.config });
    
    if (!to || !subject || !body) {
      log(`Email skipped: Missing configuration (to=${to}, subject=${subject}, body=${body})`);
      return inputData;
    }

    const timestamp = new Date().toISOString();
    const analysisText = analysis?.summary || JSON.stringify(analysis) || 'No analysis available';
    const processedSubject = subject
      .replace(/\{\{timestamp\}\}/g, timestamp)
      .replace(/\{\{analysis\}\}/g, analysisText)
      .replace(/\{\{transcript\}\}/g, transcript);
    const processedBody = body
      .replace(/\{\{timestamp\}\}/g, timestamp)
      .replace(/\{\{analysis\}\}/g, analysisText)
      .replace(/\{\{transcript\}\}/g, transcript);
    
    try {
      const response = await apiRequest('POST', '/api/send-email', {
        to,
        subject: processedSubject,
        body: processedBody,
      });
      
      log(`Email sent to ${to}`);
      return { ...inputData, emailSent: true };
    } catch (error: any) {
      log(`Email error: ${error.message}`);
      throw error;
    }
  }, [log]);

  const executeSMSNode = useCallback(async (node: Node, inputData: any) => {
    log(`Executing SMS node: ${node.id}`);
    
    const { to, message } = node.data.config || {};
    const analysis = inputData.analysis || null;
    const transcript = inputData.transcript || '';
    
    if (!to || !message) {
      log('SMS skipped: Missing configuration (to or message)');
      return inputData;
    }

    const timestamp = new Date().toISOString();
    const analysisText = analysis?.summary || JSON.stringify(analysis) || 'No analysis';
    const processedMessage = message
      .replace(/\{\{timestamp\}\}/g, timestamp)
      .replace(/\{\{analysis\}\}/g, analysisText)
      .replace(/\{\{transcript\}\}/g, transcript);
    
    try {
      const response = await apiRequest('POST', '/api/send-sms', {
        to,
        message: processedMessage,
      });
      
      log(`SMS sent to ${to}`);
      return { ...inputData, smsSent: true };
    } catch (error: any) {
      log(`SMS error: ${error.message}`);
      throw error;
    }
  }, [log]);

  const executeCallNode = useCallback(async (node: Node, inputData: any) => {
    log(`Executing call node: ${node.id}`);
    
    const { to, message } = node.data.config || {};
    const analysis = inputData.analysis || null;
    const transcript = inputData.transcript || '';
    
    if (!to || !message) {
      log('Call skipped: Missing configuration (to or message)');
      return inputData;
    }

    const timestamp = new Date().toISOString();
    const analysisText = analysis?.summary || JSON.stringify(analysis) || 'No analysis';
    const processedMessage = message
      .replace(/\{\{timestamp\}\}/g, timestamp)
      .replace(/\{\{analysis\}\}/g, analysisText)
      .replace(/\{\{transcript\}\}/g, transcript);
    
    try {
      const response = await apiRequest('POST', '/api/make-call', {
        to,
        message: processedMessage,
      });
      
      log(`Call initiated to ${to}`);
      return { ...inputData, callMade: true };
    } catch (error: any) {
      log(`Call error: ${error.message}`);
      throw error;
    }
  }, [log]);

  const executeDiscordNode = useCallback(async (node: Node, inputData: any) => {
    log(`Executing Discord node: ${node.id}`);
    
    const { webhookUrl, message } = node.data.config || {};
    const analysis = inputData.analysis || null;
    const transcript = inputData.transcript || '';
    
    if (!webhookUrl || !message) {
      log('Discord skipped: Missing configuration (webhookUrl or message)');
      return inputData;
    }

    const timestamp = new Date().toISOString();
    const analysisText = analysis?.summary || JSON.stringify(analysis) || 'No analysis';
    const processedMessage = message
      .replace(/\{\{timestamp\}\}/g, timestamp)
      .replace(/\{\{analysis\}\}/g, analysisText)
      .replace(/\{\{transcript\}\}/g, transcript);
    
    try {
      const response = await apiRequest('POST', '/api/send-discord', {
        webhookUrl,
        message: processedMessage,
      });
      
      log(`Discord message sent`);
      return { ...inputData, discordSent: true };
    } catch (error: any) {
      log(`Discord error: ${error.message}`);
      throw error;
    }
  }, [log]);

  const executeNode = useCallback(async (node: Node, inputData: any) => {
    setNodes(prev => prev.map(n => 
      n.id === node.id ? { ...n, data: { ...n.data, status: 'running' } } : n
    ));

    try {
      let result;
      switch (node.type) {
        case 'camera':
          result = await executeCameraNode(node);
          break;
        case 'transcription':
          result = await executeTranscriptionNode(node);
          break;
        case 'detection':
          result = await executeDetectionNode(node, inputData);
          break;
        case 'pose':
          result = await executePoseNode(node, inputData);
          break;
        case 'analysis':
          result = await executeAnalysisNode(node, inputData);
          break;
        case 'action':
          result = await executeSaveNode(node, inputData);
          break;
        case 'email':
          result = await executeEmailNode(node, inputData);
          break;
        case 'sms':
          result = await executeSMSNode(node, inputData);
          break;
        case 'call':
          result = await executeCallNode(node, inputData);
          break;
        case 'discord':
          result = await executeDiscordNode(node, inputData);
          break;
        default:
          result = inputData;
      }

      setNodes(prev => prev.map(n => 
        n.id === node.id ? { ...n, data: { ...n.data, status: 'completed' } } : n
      ));
      
      return result;
    } catch (error) {
      setNodes(prev => prev.map(n => 
        n.id === node.id ? { ...n, data: { ...n.data, status: 'error' } } : n
      ));
      throw error;
    }
  }, [executeCameraNode, executeTranscriptionNode, executeDetectionNode, executePoseNode, executeAnalysisNode, executeSaveNode, executeEmailNode, executeSMSNode, executeCallNode, executeDiscordNode]);

  const handleRun = useCallback(async () => {
    console.log('[HANDLERUN] Setting isRunning to true');
    setIsRunning(true);
    isRunningRef.current = true;
    setExecutionLog([]);
    log('Starting continuous video stream analysis...');
    
    // Reset batching state for new run
    previousDetectionsRef.current = [];
    lastAnalysisTimeRef.current = 0;

    try {
      // Create execution record
      const execution = await apiRequest('POST', '/api/executions', {
        flowId: currentFlowId || 'unknown',
        status: 'running',
      });
      currentExecutionIdRef.current = execution.id;
      log(`Execution started: ${execution.id}`);

      // Reset node statuses
      setNodes(prev => prev.map(n => ({ ...n, data: { ...n.data, status: undefined, detections: [] } })));

      // Find start node (camera node with no incoming edges)
      const startNode = nodes.find(n => 
        n.type === 'camera' && !edges.some(e => e.target === n.id)
      );

      if (!startNode) {
        throw new Error('No camera node found to start flow');
      }

      // Start camera once
      log('Starting camera...');
      const cameraNode = nodes.find(n => n.id === startNode.id);
      if (!cameraNode) throw new Error('Camera node not found');
      
      const cameraResult = await executeCameraNode(cameraNode);
      
      // Check if video input mode
      const inputMode = startNode.data.config?.inputMode || 'webcam';
      const isVideoMode = inputMode === 'video';
      
      if (isVideoMode) {
        log('Camera started - video input mode with AI analysis');
      } else {
        log('Camera started - beginning continuous analysis');
      }

      // Continuous capture and analysis loop
      let frameCount = 0;
      const captureAndAnalyze = async () => {
        if (!isRunningRef.current) {
          console.log('[CAPTURE] Stopped - isRunningRef is false');
          return;
        }
        
        // Check if we're using uploaded video or live stream
        const inputMode = startNode.data.config?.inputMode || 'webcam';
        const isVideoMode = inputMode === 'video';
        
        console.log('[CAPTURE] Processing frame...');
        
        try {
          frameCount++;
          
          // Get the appropriate video source (uploaded video or live stream)
          const activeVideo = isVideoMode ? uploadedVideoRef.current : videoRef.current;
          
          // For uploaded video, loop when it ends
          if (isVideoMode && activeVideo) {
            if (activeVideo.ended) {
              console.log('[VIDEO] Video ended, looping...');
              activeVideo.currentTime = 0;
              activeVideo.play();
              return; // Skip this frame, wait for video to restart
            }
          }
          
          if (!activeVideo || !canvasRef.current) {
            log('Camera/Video not ready, skipping frame');
            isProcessingFrameRef.current = false;
            return;
          }

          const video = activeVideo;
          const canvas = canvasRef.current;
          canvas.width = 640;
          canvas.height = 360;
          const ctx = canvas.getContext('2d');
          if (!ctx) return;
          
          ctx.drawImage(video, 0, 0, 640, 360);
          const imageData = canvas.toDataURL('image/jpeg', 0.7);
          
          frameSequenceRef.current.push(imageData);
          if (frameSequenceRef.current.length > 3) {
            frameSequenceRef.current.shift(); // Keep only the latest 3 frames
          }

          // Find detection, pose, and analysis nodes
          const detectionNode = nodes.find(n => n.type === 'detection' && edges.some(e => e.source === startNode.id && e.target === n.id));
          const poseNode = nodes.find(n => n.type === 'pose' && edges.some(e => e.source === startNode.id && e.target === n.id));
          const analysisNode = nodes.find(n => n.type === 'analysis' && edges.some(e => e.source === (detectionNode?.id || poseNode?.id) && e.target === n.id));
          
          let currentData = { 
            imageData,
            imagesData: [...frameSequenceRef.current] 
          };
          
          // Run detection (always - needed for bounding boxes)
          if (detectionNode) {
            currentData = await executeDetectionNode(detectionNode, currentData);
            const detections = currentData.detections || [];
            const currentClasses = detections.map((d: any) => d.class).sort();
            const previousClasses = previousDetectionsRef.current;
            
            // Check if detections changed
            const detectionsChanged = 
              currentClasses.length !== previousClasses.length ||
              currentClasses.some((cls, idx) => cls !== previousClasses[idx]);
            
            // Check if cooldown expired
            const now = Date.now();
            const cooldownExpired = (now - lastAnalysisTimeRef.current) > ANALYSIS_COOLDOWN;
            
            log(`Frame ${frameCount}: Detected ${detections.length} objects ${detections.length > 0 ? '(' + currentClasses.join(', ') + ')' : ''}`);
            
            // Smart batching: Only analyze periodically (cooldown), not on every change
            // This prevents analysis from blocking continuous detection
            if (analysisNode && cooldownExpired) {
              log(`  → Periodic analysis (${ANALYSIS_COOLDOWN/1000}s interval)...`);
              
              currentData = await executeAnalysisNode(analysisNode, currentData);
              const analysis = currentData.analysis;
              if (analysis) {
                log(`  → Analysis: ${analysis.summary?.substring(0, 80)}...`);
                
                // Save result to localStorage
                if (currentFlowId) {
                  saveFlowResult({
                    flowId: currentFlowId,
                    flowName: flowName,
                    timestamp: new Date().toISOString(),
                    detections: detections,
                    analysis: analysis,
                  });
                  log(`  ✓ Result saved`);
                }
                
                // Execute action nodes if condition is met
                if (analysis.condition_met) {
                  log(`  ✓ Condition met! Executing action nodes...`);
                  
                  // Find all action nodes connected to the analysis node
                  const actionEdges = edges.filter(e => e.source === analysisNode.id);
                  for (const edge of actionEdges) {
                    const actionNode = nodes.find(n => n.id === edge.target);
                    if (actionNode) {
                      try {
                        log(`  → Executing ${actionNode.type} node: ${actionNode.data.label}`);
                        await executeNode(actionNode, currentData);
                        log(`  ✓ ${actionNode.type} completed`);
                      } catch (error: any) {
                        log(`  ✗ ${actionNode.type} failed: ${error.message}`);
                      }
                    }
                  }
                } else {
                  log(`  → Condition not met, skipping action nodes`);
                }
              }
              
              // Update tracking
              previousDetectionsRef.current = currentClasses;
              lastAnalysisTimeRef.current = now;
            } else if (analysisNode) {
              log(`  → Skipping analysis (no changes)`);
            }
            
            // If no analysis node, save detection results periodically
            if (!analysisNode && cooldownExpired && detections.length > 0) {
              if (currentFlowId) {
                saveFlowResult({
                  flowId: currentFlowId,
                  flowName: flowName,
                  timestamp: new Date().toISOString(),
                  detections: detections,
                });
                log(`  ✓ Detection result saved`);
                lastAnalysisTimeRef.current = now;
              }
            }
          }
          
          // Run pose detection (always - needed for skeleton visualization)
          if (poseNode) {
            const poseData = await executePoseNode(poseNode, { imageData });
            const poseDetections = poseData.detections || [];
            
            log(`Frame ${frameCount}: Detected ${poseDetections.length} pose${poseDetections.length !== 1 ? 's' : ''}`);
            if (poseDetections.length > 0) {
              poseDetections.forEach((det: any, idx: number) => {
                const visibleKeypoints = det.keypoints?.filter((kp: any) => kp.visibility > 0.5).length || 0;
                log(`  Pose ${idx + 1}: ${visibleKeypoints}/17 visible keypoints`);
              });
            }
            
            // Merge pose detections with existing detections for the camera node
            currentData.detections = [...(currentData.detections || []), ...poseDetections];
            
            // Check if there's an analysis node connected to pose node
            const poseAnalysisNode = nodes.find(n => n.type === 'analysis' && edges.some(e => e.source === poseNode.id && e.target === n.id));
            
            if (poseAnalysisNode && poseDetections.length > 0) {
              const now = Date.now();
              const cooldownExpired = (now - lastAnalysisTimeRef.current) > ANALYSIS_COOLDOWN;
              
              if (cooldownExpired) {
                log(`  → Pose analysis (${ANALYSIS_COOLDOWN/1000}s interval)...`);
                
                currentData = await executeAnalysisNode(poseAnalysisNode, currentData);
                const analysis = currentData.analysis;
                
                if (analysis) {
                  log(`  → Analysis: ${analysis.summary?.substring(0, 80)}...`);
                  log(`  → Condition: ${analysis.condition_met ? '✓ MET' : '✗ NOT MET'}`);
                  
                  // Save pose analysis result to localStorage
                  if (currentFlowId) {
                    saveFlowResult({
                      flowId: currentFlowId,
                      flowName: flowName,
                      timestamp: new Date().toISOString(),
                      detections: poseDetections,
                      analysis: analysis,
                    });
                    log(`  ✓ Pose analysis result saved`);
                  }
                  
                  // Execute action nodes if condition is met
                  if (analysis.condition_met) {
                    log(`  ✓ Condition met! Executing action nodes...`);
                    
                    const actionEdges = edges.filter(e => e.source === poseAnalysisNode.id);
                    for (const edge of actionEdges) {
                      const actionNode = nodes.find(n => n.id === edge.target);
                      if (actionNode) {
                        try {
                          log(`  → Executing ${actionNode.type} node: ${actionNode.data.label}`);
                          await executeNode(actionNode, currentData);
                          log(`  ✓ ${actionNode.type} completed`);
                        } catch (error: any) {
                          log(`  ✗ ${actionNode.type} failed: ${error.message}`);
                        }
                      }
                    }
                  } else {
                    log(`  → Condition not met, skipping action nodes`);
                  }
                }
                
                lastAnalysisTimeRef.current = now;
              }
            } else if (!poseAnalysisNode && poseDetections.length > 0) {
              // If no analysis node, save pose detection results periodically
              const now = Date.now();
              const cooldownExpired = (now - lastAnalysisTimeRef.current) > ANALYSIS_COOLDOWN;
              
              if (cooldownExpired && currentFlowId) {
                saveFlowResult({
                  flowId: currentFlowId,
                  flowName: flowName,
                  timestamp: new Date().toISOString(),
                  detections: poseDetections,
                });
                log(`  ✓ Pose detection result saved`);
                lastAnalysisTimeRef.current = now;
              }
            }
          }
          
          setFlowData(currentData);
          
        } catch (error: any) {
          log(`⚠ Frame ${frameCount} error: ${error.message}`);
          console.error('[FRAME_ERROR]', error);
          // Continue running even after errors
        }
      };

      // Determine target frame rate based on node type
      const hasPoseNode = nodes.some(n => n.type === 'pose');
      const targetFrameInterval = hasPoseNode ? 50 : 100; // 20 FPS for pose, 10 FPS for objects
      const fps = hasPoseNode ? 20 : 10;
      
      // Adaptive capture loop - waits for processing to complete before next frame
      const adaptiveCaptureLoop = async () => {
        if (!isRunningRef.current) return;
        
        const startTime = Date.now();
        await captureAndAnalyze();
        const processingTime = Date.now() - startTime;
        
        // Schedule next frame: wait for remaining time to hit target FPS, or go immediately if processing took longer
        const nextDelay = Math.max(0, targetFrameInterval - processingTime);
        captureIntervalRef.current = setTimeout(adaptiveCaptureLoop, nextDelay) as any;
      };
      
      // Start the adaptive loop
      adaptiveCaptureLoop();
      console.log('[ADAPTIVE] Started adaptive capture loop at target', fps, 'FPS');
      log(`✓ Continuous capture started (adaptive ~${fps} FPS)`);

      toast({
        title: 'Stream Started',
        description: `Continuous analysis running at ${fps} FPS`,
      });
    } catch (error: any) {
      log(`Flow start failed: ${error.message}`);
      
      // Mark execution as failed
      if (currentExecutionIdRef.current) {
        await apiRequest('PUT', `/api/executions/${currentExecutionIdRef.current}`, {
          status: 'failed',
          error: error.message,
        });
      }
      
      toast({
        title: 'Flow Failed',
        description: error.message,
        variant: 'destructive',
      });
      setIsRunning(false);
      isRunningRef.current = false;
    }
  }, [nodes, edges, executeCameraNode, executeDetectionNode, executePoseNode, executeAnalysisNode, executeNode, log, toast, currentFlowId, flowName, saveFlowResult]);

  // Auto-resume video upload flows when returning to them
  useEffect(() => {
    if (shouldAutoResume && !isRunning) {
      console.log('[AUTO_RESUME] Triggering handleRun to resume video flow');
      setShouldAutoResume(false); // Reset flag
      handleRun(); // Actually start the flow execution
    }
  }, [shouldAutoResume, isRunning, handleRun]);

  // Execute transcription-based flow with voice-activated meeting end detection
  const handleExecuteTranscriptionFlow = useCallback(async () => {
    setIsRunning(true);
    isRunningRef.current = true;
    setExecutionLog([]);
    log('Starting voice-activated transcription flow...');

    try {
      // Find transcription node (should be a source node with no incoming edges)
      let transcriptionNode = nodes.find(n => 
        n.type === 'transcription' && !edges.some(e => e.target === n.id)
      );

      // Fallback: if no source transcription node found, find any transcription node
      if (!transcriptionNode) {
        transcriptionNode = nodes.find(n => n.type === 'transcription');
      }

      if (!transcriptionNode) {
        throw new Error('No transcription node found in flow. Please add a Transcription node to use this feature.');
      }

      // Find analysis node to get trigger phrase
      const analysisNode = nodes.find(n => n.type === 'analysis');
      const triggerPhrase = analysisNode?.data.config?.triggerPhrase?.toLowerCase() || 'meeting ended';
      
      log(`Monitoring for trigger phrase: "${triggerPhrase}"`);
      log('Starting microphone capture...');

      // Auto-start transcription by triggering the node
      setNodes(prev => prev.map(n => 
        n.id === transcriptionNode!.id 
          ? { ...n, data: { ...n.data, triggerStart: Date.now() } }
          : n
      ));

      // Wait a bit for transcription to start
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Track last save time for periodic result saving
      let lastSaveTime = Date.now();
      const SAVE_INTERVAL = 10000; // Save results every 10 seconds

      // Monitor transcript for trigger phrase
      const monitoringInterval = setInterval(() => {
        if (!isRunningRef.current) {
          clearInterval(monitoringInterval);
          return;
        }

        // Read from ref to get latest nodes state
        const currentNode = nodesRef.current.find(n => n.id === transcriptionNode!.id);
        const transcript = currentNode?.data.transcript || '';

        log(`Monitoring... (${transcript.length} chars captured)`);

        // Periodically save transcript results
        const now = Date.now();
        if (transcript && now - lastSaveTime > SAVE_INTERVAL && currentFlowId) {
          saveFlowResult({
            flowId: currentFlowId,
            flowName: flowName,
            timestamp: new Date().toISOString(),
            transcript: transcript,
          });
          log(`✓ Transcript auto-saved (${transcript.length} chars)`);
          lastSaveTime = now;
        }

        // Check if trigger phrase is in transcript
        if (transcript.toLowerCase().includes(triggerPhrase)) {
          log(`🎯 Trigger phrase detected: "${triggerPhrase}"`);
          log('Stopping transcription and processing...');
          
          clearInterval(monitoringInterval);
          
          // Stop transcription
          setNodes(prev => prev.map(n => 
            n.id === transcriptionNode!.id 
              ? { ...n, data: { ...n.data, triggerStop: Date.now() } }
              : n
          ));

          // Wait for transcription to stop
          setTimeout(async () => {
            try {
              // Clear trigger flags and reset node statuses
              setNodes(prev => prev.map(n => ({
                ...n,
                data: {
                  ...n.data,
                  status: undefined,
                  triggerStart: undefined,
                  triggerStop: undefined,
                }
              })));

              // Execute flow by traversing the graph using latest nodes
              let currentData = await executeNode(transcriptionNode!, {});
              let currentNodeId = transcriptionNode!.id;

              // Traverse connected nodes
              while (true) {
                const nextEdge = edges.find(e => e.source === currentNodeId);
                if (!nextEdge) break;

                const nextNode = nodesRef.current.find(n => n.id === nextEdge.target);
                if (!nextNode) break;

                currentData = await executeNode(nextNode, currentData);
                currentNodeId = nextNode.id;
              }

              // Save transcription result to result panel
              if (currentFlowId && currentData.transcript) {
                saveFlowResult({
                  flowId: currentFlowId,
                  flowName: flowName,
                  timestamp: new Date().toISOString(),
                  transcript: currentData.transcript,
                  analysis: currentData.analysis,
                });
                log(`✓ Transcription result saved to result panel`);
              }

              log('✓ Flow execution completed');
              toast({
                title: 'Meeting Ended',
                description: 'Transcript processed and email sent successfully',
              });
            } catch (error: any) {
              log(`Flow execution failed: ${error.message}`);
              toast({
                title: 'Flow Failed',
                description: error.message,
                variant: 'destructive',
              });
            } finally {
              setIsRunning(false);
              isRunningRef.current = false;
            }
          }, 500);
        }
      }, 5000); // Check every 5 seconds

      // Store interval reference for cleanup
      captureIntervalRef.current = monitoringInterval as any;

    } catch (error: any) {
      log(`Flow startup failed: ${error.message}`);
      toast({
        title: 'Flow Failed',
        description: error.message,
        variant: 'destructive',
      });
      setIsRunning(false);
      isRunningRef.current = false;
    }
  }, [nodes, edges, executeNode, log, toast, currentFlowId, flowName, saveFlowResult]);

  const handleStop = useCallback(async () => {
    setIsRunning(false);
    isRunningRef.current = false;
    log('Continuous stream stopped by user');
    
    // Clear execution state from sessionStorage
    if (currentFlowId) {
      sessionStorage.removeItem(`flow_${currentFlowId}_running`);
      sessionStorage.removeItem(`flow_${currentFlowId}_executionId`);
    }
    
    // Mark execution as completed and clear session
    if (currentExecutionIdRef.current) {
      try {
        await apiRequest('PUT', `/api/executions/${currentExecutionIdRef.current}`, {
          status: 'completed',
        });
        log(`Execution completed: ${currentExecutionIdRef.current}`);
        
        // Clear motion tracking session on Python backend
        try {
          await fetch('http://localhost:8001/api/execution-session/clear', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ execution_id: currentExecutionIdRef.current }),
          });
          log(`Motion tracking session cleared`);
        } catch (error) {
          console.error('Failed to clear motion tracking session:', error);
        }
      } catch (error) {
        console.error('Failed to update execution status:', error);
      }
      currentExecutionIdRef.current = null;
    }
    
    // Clear timeout
    if (captureIntervalRef.current) {
      clearTimeout(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
    
    // Reset batching and processing state
    previousDetectionsRef.current = [];
    lastAnalysisTimeRef.current = 0;
    isProcessingFrameRef.current = false;
    
    // Stop camera
    if (streamRef.current && streamRef.current instanceof MediaStream) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
    }
    // Stop Ant Media connection
    if (webRTCAdaptorRef.current) {
      log('Disconnecting from Ant Media Server...');
      webRTCAdaptorRef.current.stop();
      webRTCAdaptorRef.current = null;
    }

    // Reset node statuses and mark camera nodes as inactive, clear detections
    setNodes(prev => prev.map(n => ({ 
      ...n, 
      data: { 
        ...n.data, 
        status: undefined,
        isActive: n.type === 'camera' ? false : n.data.isActive,
        stream: n.type === 'camera' ? null : n.data.stream,
        detections: n.type === 'camera' ? [] : n.data.detections
      } 
    })));
    
    toast({
      title: 'Stream Stopped',
      description: 'Continuous analysis stopped and camera released',
    });
  }, [log, toast]);

  const handleFlowNameChange = useCallback((newName: string) => {
    setFlowName(newName);
    // Auto-save when flow name changes (throttled)
    saveCurrentFlow({ name: newName, nodes, edges });
  }, [nodes, edges, saveCurrentFlow]);

  const handleNodesChange = useCallback((newNodes: Node[]) => {
    console.log('[NODES_CHANGE] handleNodesChange called with', newNodes.length, 'nodes');
    console.log('[NODES_CHANGE] Analysis node:', newNodes.find(n => n.type === 'analysis'));
    
    // Check if the only change is the selected property
    const onlySelectionChanged = nodes.length === newNodes.length && 
      nodes.every((oldNode, i) => {
        const newNode = newNodes.find(n => n.id === oldNode.id);
        if (!newNode) return false;
        
        // Check if everything except selected is the same
        const { selected: oldSelected, ...oldRest } = oldNode;
        const { selected: newSelected, ...newRest } = newNode;
        return JSON.stringify(oldRest) === JSON.stringify(newRest);
      });
    
    setNodes(newNodes);
    
    // Don't save if only selection changed (prevents overwriting edits with stale data)
    if (!onlySelectionChanged) {
      console.log('[NODES_CHANGE] Saving because meaningful change detected');
      saveCurrentFlow({ name: flowName, nodes: newNodes, edges });
    } else {
      console.log('[NODES_CHANGE] Skipping save - only selection changed');
    }
  }, [nodes, flowName, edges, saveCurrentFlow]);

  const handleEdgesChange = useCallback((newEdges: Edge[]) => {
    setEdges(newEdges);
    // Auto-save when edges are added or deleted (throttled)
    saveCurrentFlow({ name: flowName, nodes, edges: newEdges });
  }, [flowName, nodes, saveCurrentFlow]);

  const onNodeClick = useCallback((event: any, clickedNode: Node) => {
    // Always get the latest version of the node from our state, not React Flow's potentially stale version
    const latestNode = nodes.find(n => n.id === clickedNode.id);
    setSelectedNode(latestNode || clickedNode);
  }, [nodes]);

  const onPaneClick = useCallback(() => {
    setSelectedNode(null);
  }, []);

  const handleNodeUpdate = useCallback((nodeId: string, data: any) => {
    console.log('[UPDATE] Node update called:', { nodeId, data });
    setNodes((nds) => {
      const updatedNodes = nds.map((node) => {
        if (node.id === nodeId) {
          const updatedNode = { ...node, data };
          console.log('[UPDATE] Updated node:', updatedNode);
          // Also update selectedNode if it's the same node
          if (selectedNode?.id === nodeId) {
            setSelectedNode(updatedNode);
          }
          return updatedNode;
        }
        return node;
      });
      
      // Auto-save to localStorage whenever node config changes (throttled)
      console.log('[UPDATE] Calling saveCurrentFlow with updated nodes');
      saveCurrentFlow({ name: flowName, nodes: updatedNodes, edges });
      
      return updatedNodes;
    });
  }, [selectedNode, flowName, edges, saveCurrentFlow]);

  const handleNodeAdd = useCallback(
    (type: string) => {
      // Auto-save when new node is added (throttled)
      saveCurrentFlow({ name: flowName, nodes, edges });
      toast({
        title: 'Node Added',
        description: `${type} node added to canvas`,
      });
    },
    [toast, flowName, nodes, edges, saveCurrentFlow]
  );

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)] bg-background">
      {/* Hidden video and canvas for camera capture */}
      <video id="flow-video-player" ref={videoRef} className="hidden" autoPlay playsInline muted />
      <video ref={uploadedVideoRef} className="hidden" autoPlay playsInline muted />
      <canvas ref={canvasRef} className="hidden" />

      {/* Top Toolbar */}
      <div className="h-14 border-b border-border bg-card flex items-center justify-between px-4 gap-4">
        <div className="flex items-center gap-4 flex-1">
          <Button
            variant="ghost"
            size="sm"
            className="gap-2 transition-spring"
            onClick={() => setLocation('/')}
            data-testid="button-back-to-dashboard"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Dashboard
          </Button>
          <Input
            value={flowName}
            onChange={(e) => handleFlowNameChange(e.target.value)}
            className="max-w-sm"
            disabled={isRunning}
            data-testid="input-flow-name"
          />
        </div>

        <div className="flex items-center gap-2">
          <Button 
            variant="secondary" 
            size="sm" 
            onClick={handleSave} 
            disabled={isRunning}
            data-testid="button-save-flow"
          >
            <Save className="w-4 h-4 mr-2" />
            Save
          </Button>
          {!isRunning ? (
            <>
              {/* Detect flow type and show appropriate button */}
              {nodes.some(n => n.type === 'transcription') ? (
                <Button 
                  size="sm" 
                  onClick={handleExecuteTranscriptionFlow}
                  data-testid="button-execute-flow"
                >
                  <Play className="w-4 h-4 mr-2" />
                  Execute Flow
                </Button>
              ) : (
                <Button 
                  size="sm" 
                  onClick={handleRun}
                  data-testid="button-run-flow"
                >
                  <Play className="w-4 h-4 mr-2" />
                  Run Flow
                </Button>
              )}
            </>
          ) : (
            <Button 
              size="sm" 
              variant="destructive"
              onClick={handleStop}
              data-testid="button-stop-flow"
            >
              <StopCircle className="w-4 h-4 mr-2" />
              Stop Flow
            </Button>
          )}
        </div>
      </div>

      {/* Canvas Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar - Node Palette & AI Assistant */}
        <div className="flex flex-col w-64 border-r border-border bg-card/50">
          <NodePalette />
          
          {/* AI Flow Editor Chat */}
          <div className="p-3 border-t border-border">
            <Card className="p-3 bg-card/95 backdrop-blur-sm border-primary/20 shadow-lg">
              <div className="flex items-center gap-2 mb-2">
                <Sparkles className="w-4 h-4 text-primary" />
                <span className="text-sm font-semibold gradient-text">AI Assistant</span>
              </div>
              <div className="flex gap-2">
                <Input
                  value={aiChatInput}
                  onChange={(e) => setAiChatInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleAiEditFlow();
                    }
                  }}
                  placeholder="Edit this flow..."
                  disabled={isEditingFlow || isRunning}
                  className="flex-1 text-sm"
                  data-testid="input-ai-chat"
                />
                <Button
                  size="sm"
                  onClick={handleAiEditFlow}
                  disabled={!aiChatInput.trim() || isEditingFlow || isRunning}
                  data-testid="button-ai-send"
                >
                  {isEditingFlow ? (
                    <div className="w-4 h-4 border-2 border-primary-foreground border-t-transparent rounded-full animate-spin" />
                  ) : (
                    <Send className="w-4 h-4" />
                  )}
                </Button>
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                Try: "add email node"
              </p>
            </Card>
          </div>
        </div>

        {/* Main Canvas */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="flex-1 relative" ref={reactFlowWrapper}>
            <FlowCanvas
              initialNodes={nodes}
              initialEdges={edges}
              onNodesChange={handleNodesChange}
              onEdgesChange={handleEdgesChange}
              onNodeClick={onNodeClick}
              onPaneClick={onPaneClick}
              onNodeAdd={handleNodeAdd}
            />
            
            {selectedNode && (
              <NodeConfigPanel
                node={selectedNode}
                onClose={() => setSelectedNode(null)}
                onUpdate={handleNodeUpdate}
              />
            )}
          </div>

          {/* Execution Log */}
          {executionLog.length > 0 && (
            <Card className="m-4 p-4 max-h-48 overflow-y-auto">
              <h3 className="text-sm font-semibold mb-2">Execution Log</h3>
              <div className="space-y-1">
                {executionLog.map((log, i) => (
                  <div key={i} className="text-xs font-mono text-muted-foreground">{log}</div>
                ))}
              </div>
            </Card>
          )}
        </div>
        
        {/* Right Sidebar - Results */}
        <div className="w-96 border-l border-border bg-card p-4 overflow-hidden">
          <FlowResults flowId={currentFlowId} flowName={flowName} />
        </div>
      </div>
    </div>
  );
}
