import { useState, useRef, useCallback, useEffect } from 'react';
import { Camera, Video, X, Image as ImageIcon, Timer, Square, RadioTower } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { WebRTCAdaptor } from '@antmedia/webrtc_adaptor';

interface CameraCaptureProps {
  onCapture: (image: string) => void;
  onClose?: () => void;
}

export function CameraCapture({ onCapture, onClose }: CameraCaptureProps) {
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [isAutoDetecting, setIsAutoDetecting] = useState(false);
  const [sourceMode, setSourceMode] = useState<'local' | 'antmedia'>('local');
  
  // Ant Media State
  const [amsUrl, setAmsUrl] = useState('ws://localhost:5080/LiveApp/websocket');
  const [streamId, setStreamId] = useState('stream1');
  const [isAmsConnected, setIsAmsConnected] = useState(false);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const webRTCAdaptorRef = useRef<any>(null);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  const startLocalCamera = useCallback(async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
      });
      setStream(mediaStream);
      setSourceMode('local');
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        await videoRef.current.play();
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('Failed to access camera. Please ensure camera permissions are granted.');
    }
  }, []);

  // --- ANT MEDIA INTEGRATION ---
  const startAntMediaStream = useCallback(() => {
    if (!videoRef.current) return;
    
    // Initialize Ant Media WebRTC Adaptor
    webRTCAdaptorRef.current = new WebRTCAdaptor({
      websocket_url: amsUrl,
      mediaConstraints: {
        video: false,
        audio: false,
      },
      peerconnection_config: {
        iceServers: [{ urls: 'stun:stun1.l.google.com:19302' }],
      },
      sdp_constraints: {
        OfferToReceiveAudio: true,
        OfferToReceiveVideo: true,
      },
      remoteVideoId: 'video-player', // ID of the video element
      callback: (info: string, obj: any) => {
        if (info === 'initialized') {
          console.log('Ant Media initialized! Playing stream...');
          webRTCAdaptorRef.current.play(streamId);
        } else if (info === 'play_started') {
          console.log('Ant Media play started!');
          setIsAmsConnected(true);
          setSourceMode('antmedia');
          // Mock a stream object just to pass the UI checks
          setStream(new MediaStream()); 
        } else if (info === 'play_finished') {
          console.log('Ant Media play finished');
          stopCamera();
        }
      },
      callbackError: (error: string) => {
        console.error('Ant Media error:', error);
        alert(`Ant Media Connection Error: ${error}`);
      },
    });
  }, [amsUrl, streamId]);

  const stopCamera = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setIsAutoDetecting(false);

    // Stop local stream
    if (sourceMode === 'local' && stream && stream instanceof MediaStream) {
      stream.getTracks().forEach((track) => track.stop());
    }

    // Stop Ant Media stream
    if (sourceMode === 'antmedia' && webRTCAdaptorRef.current) {
      webRTCAdaptorRef.current.stop(streamId);
      webRTCAdaptorRef.current.closeStream(webRTCAdaptorRef.current);
      setIsAmsConnected(false);
    }

    setStream(null);
  }, [stream, sourceMode, streamId]);

  // Frame extraction remains identical! (It seamlessly reads the video tag regardless of the source)
  const getFrameBase64 = useCallback(() => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth || 1280;
      canvas.height = video.videoHeight || 720;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        return canvas.toDataURL('image/jpeg', 0.8); // 0.8 quality to save bandwidth
      }
    }
    return null;
  }, []);

  const toggleAutoDetect = useCallback(() => {
    if (isAutoDetecting) {
      if (intervalRef.current) clearInterval(intervalRef.current);
      intervalRef.current = null;
      setIsAutoDetecting(false);
    } else {
      setIsAutoDetecting(true);
      const firstFrame = getFrameBase64();
      if (firstFrame) onCapture(firstFrame);

      intervalRef.current = setInterval(() => {
        const frameData = getFrameBase64();
        if (frameData) onCapture(frameData);
      }, 3000);
    }
  }, [isAutoDetecting, getFrameBase64, onCapture]);

  return (
    <Card className="p-6 max-w-3xl mx-auto border-[var(--border)] bg-card text-card-foreground">
      <div className="flex items-center justify-between mb-6 border-b border-border pb-4">
        <h3 className="text-lg font-semibold flex items-center gap-2 p5-text text-primary">
          <Camera className="w-5 h-5" />
          VISUAL CAPTURE NODE
        </h3>
        {onClose && (
          <Button variant="ghost" size="icon" onClick={onClose} className="hover:text-destructive">
            <X className="w-5 h-5" />
          </Button>
        )}
      </div>

      {!stream && !capturedImage && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          {/* LOCAL CAMERA SECTION */}
          <div className="border border-border p-4 flex flex-col gap-4">
            <div className="flex items-center gap-2">
              <Video className="w-5 h-5 text-primary" />
              <h4 className="font-bold p5-text">LOCAL DEVICE</h4>
            </div>
            <p className="text-sm text-muted-foreground">Capture video from your current device's webcam or screen.</p>
            <Button onClick={startLocalCamera} className="w-full mt-auto p5-text">
              INITIALIZE CAMERA
            </Button>
          </div>

          {/* ANT MEDIA SERVER SECTION */}
          <div className="border border-border p-4 flex flex-col gap-4">
            <div className="flex items-center gap-2">
              <RadioTower className="w-5 h-5 text-accent" />
              <h4 className="font-bold p5-text text-accent">ANT MEDIA SERVER</h4>
            </div>
            <p className="text-sm text-muted-foreground mb-2">Connect to a remote WebRTC stream via AMS for ultra-low latency AI analysis.</p>
            
            <div className="space-y-2">
              <Label className="text-xs">WebSocket URL</Label>
              <Input 
                value={amsUrl} 
                onChange={(e) => setAmsUrl(e.target.value)}
                className="bg-input h-8 text-xs"
                placeholder="wss://your-server/LiveApp/websocket"
              />
            </div>
            <div className="space-y-2">
              <Label className="text-xs">Stream ID</Label>
              <Input 
                value={streamId} 
                onChange={(e) => setStreamId(e.target.value)}
                className="bg-input h-8 text-xs"
                placeholder="stream1"
              />
            </div>

            <Button onClick={startAntMediaStream} variant="secondary" className="w-full mt-auto p5-text border border-accent text-accent hover:bg-accent hover:text-white">
              CONNECT AMS STREAM
            </Button>
          </div>
        </div>
      )}

      <div className="relative aspect-video bg-black rounded-none overflow-hidden mb-4 border-2 border-primary shadow-[var(--shadow)]">
        {!stream && !capturedImage && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-4">
            <Camera className="w-16 h-16 text-muted-foreground/30" />
            <p className="text-sm text-muted-foreground p5-text">AWAITING VIDEO SOURCE</p>
          </div>
        )}
        
        {/* We keep the video element rendered but hide it if we have a captured image.
            Notice ID is set to 'video-player' for Ant Media to hook into. */}
        <video
          id="video-player"
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`w-full h-full object-cover ${(!stream || capturedImage) ? 'hidden' : 'block'}`}
        />
        
        {isAutoDetecting && (
          <div className="absolute top-4 right-4 flex items-center gap-2 bg-black/80 border border-primary px-3 py-1 text-primary text-xs font-bold tracking-widest uppercase shadow-[var(--shadow)]">
            <div className="w-3 h-3 rounded-full bg-destructive animate-pulse" />
            Monitoring [ {sourceMode.toUpperCase()} ]
          </div>
        )}

        {capturedImage && (
          <img src={capturedImage} alt="Captured" className="w-full h-full object-cover" />
        )}

        <canvas ref={canvasRef} className="hidden" />
      </div>

      <div className="flex gap-4 justify-center flex-wrap">
        {stream && !capturedImage && (
          <>
            <Button 
              onClick={toggleAutoDetect} 
              variant={isAutoDetecting ? "destructive" : "default"}
              className="p5-text font-bold tracking-wider transition-all duration-200 hover:scale-105"
            >
              {isAutoDetecting ? <Square className="w-4 h-4 mr-2" fill="currentColor" /> : <Timer className="w-4 h-4 mr-2" />}
              {isAutoDetecting ? "STOP MONITORING" : "START AI MONITOR (3s)"}
            </Button>
            
            <Button onClick={stopCamera} variant="outline" className="p5-text hover:bg-destructive hover:text-destructive-foreground hover:border-destructive" disabled={isAutoDetecting}>
              DISCONNECT
            </Button>
          </>
        )}

        {capturedImage && (
          <Button onClick={() => setCapturedImage(null)} variant="secondary" className="p5-text">
            RETAKE / RESUME
          </Button>
        )}
      </div>
    </Card>
  );
}