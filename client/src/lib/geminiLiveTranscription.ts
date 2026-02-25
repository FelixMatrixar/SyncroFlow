// Real-time transcription using server-side Gemini Live API proxy
// Frontend streams audio via WebSocket to Express server
// Express server forwards to Gemini Live API and returns transcripts
// This keeps the GEMINI_API_KEY secure on the server side

export interface TranscriptionSession {
  start: () => Promise<void>;
  stop: () => void;
  onTranscript: (callback: (text: string) => void) => void;
  isActive: () => boolean;
}

export class GeminiLiveTranscription {
  private ws: WebSocket | null = null;
  private audioContext: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private scriptProcessor: ScriptProcessorNode | null = null;
  private transcriptCallback: ((text: string) => void) | null = null;
  private isRunning = false;
  private audioBuffer: Int16Array[] = [];
  private readonly BUFFER_SIZE = 16000; // 1 second of audio at 16kHz
  private readonly SEND_INTERVAL_MS = 100; // Send every 100ms
  private sendIntervalId: number | null = null;

  async createSession(): Promise<TranscriptionSession> {
    return {
      start: async () => {
        if (this.isRunning) {
          console.warn('[GEMINI_LIVE] Session already running');
          return;
        }

        try {
          console.log('[GEMINI_LIVE] Starting transcription session...');

          // Request microphone access
          this.mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
              echoCancellation: true,
              noiseSuppression: true,
              sampleRate: 16000,
            },
          });
          console.log('[GEMINI_LIVE] Microphone access granted');

          // Create audio context for processing
          this.audioContext = new AudioContext({ sampleRate: 16000 });
          const source = this.audioContext.createMediaStreamSource(this.mediaStream);

          // Connect to server-side WebSocket proxy
          const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
          const wsUrl = `${protocol}//${window.location.host}/api/transcribe/live`;
          this.ws = new WebSocket(wsUrl);

          this.ws.onopen = () => {
            console.log('[GEMINI_LIVE] Connected to transcription server');
          };

          this.ws.onmessage = (event) => {
            try {
              const message = JSON.parse(event.data);
              if (message.type === 'transcript' && message.text && this.transcriptCallback) {
                console.log('[GEMINI_LIVE] Received transcript:', message.text);
                this.transcriptCallback(message.text);
              } else if (message.type === 'error') {
                console.error('[GEMINI_LIVE] Server error:', message.error);
              }
            } catch (error) {
              console.error('[GEMINI_LIVE] Error parsing message:', error);
            }
          };

          this.ws.onerror = (error) => {
            console.error('[GEMINI_LIVE] WebSocket error:', error);
          };

          this.ws.onclose = () => {
            console.log('[GEMINI_LIVE] WebSocket connection closed');
            this.cleanup();
          };

          // Wait for connection to be established
          await new Promise((resolve, reject) => {
            const timeout = setTimeout(() => reject(new Error('Connection timeout')), 5000);
            if (this.ws) {
              this.ws.addEventListener('open', () => {
                clearTimeout(timeout);
                resolve(null);
              }, { once: true });
              this.ws.addEventListener('error', () => {
                clearTimeout(timeout);
                reject(new Error('Connection failed'));
              }, { once: true });
            }
          });

          // Process audio chunks and buffer them
          this.scriptProcessor = this.audioContext.createScriptProcessor(4096, 1, 1);
          
          this.scriptProcessor.onaudioprocess = (event) => {
            if (!this.isRunning) return;

            const inputData = event.inputBuffer.getChannelData(0);
            
            // Convert Float32Array to Int16Array (16-bit PCM)
            const pcmData = new Int16Array(inputData.length);
            for (let i = 0; i < inputData.length; i++) {
              const s = Math.max(-1, Math.min(1, inputData[i]));
              pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }

            // Add to buffer
            this.audioBuffer.push(pcmData);
          };

          source.connect(this.scriptProcessor);
          this.scriptProcessor.connect(this.audioContext.destination);

          // Set up interval to send buffered audio
          this.sendIntervalId = window.setInterval(() => {
            this.sendBufferedAudio();
          }, this.SEND_INTERVAL_MS);

          this.isRunning = true;
          console.log('[GEMINI_LIVE] Transcription started');
        } catch (error) {
          console.error('[GEMINI_LIVE] Failed to start session:', error);
          this.cleanup();
          throw error;
        }
      },

      stop: () => {
        console.log('[GEMINI_LIVE] Stopping transcription session...');
        this.cleanup();
      },

      onTranscript: (callback: (text: string) => void) => {
        this.transcriptCallback = callback;
      },

      isActive: () => this.isRunning,
    };
  }

  private sendBufferedAudio() {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN || this.audioBuffer.length === 0) {
      return;
    }

    // Combine all buffered chunks
    let totalLength = 0;
    for (const chunk of this.audioBuffer) {
      totalLength += chunk.length;
    }

    const combinedBuffer = new Int16Array(totalLength);
    let offset = 0;
    for (const chunk of this.audioBuffer) {
      combinedBuffer.set(chunk, offset);
      offset += chunk.length;
    }

    // Clear buffer
    this.audioBuffer = [];

    // Convert to base64
    const base64Audio = this.arrayBufferToBase64(combinedBuffer.buffer);

    // Log the size to monitor
    console.log(`[GEMINI_LIVE] Sending ${combinedBuffer.length} samples (${base64Audio.length} chars)`);

    // Send to server
    try {
      this.ws.send(JSON.stringify({
        type: 'audio',
        data: base64Audio,
      }));
    } catch (error) {
      console.error('[GEMINI_LIVE] Error sending audio:', error);
    }
  }

  private arrayBufferToBase64(buffer: ArrayBuffer): string {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  }

  private cleanup() {
    this.isRunning = false;

    if (this.sendIntervalId !== null) {
      clearInterval(this.sendIntervalId);
      this.sendIntervalId = null;
    }

    // Send any remaining buffered audio
    if (this.audioBuffer.length > 0) {
      this.sendBufferedAudio();
    }
    this.audioBuffer = [];

    if (this.scriptProcessor) {
      this.scriptProcessor.disconnect();
      this.scriptProcessor = null;
    }

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.close();
    }
    this.ws = null;

    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }

    console.log('[GEMINI_LIVE] Session cleaned up');
  }
}
