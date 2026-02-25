// client\src\components\NodeConfigPanel.tsx

import { X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import type { Node } from '@xyflow/react';

// 80 COCO object classes that YOLOv8 can detect
const COCO_CLASSES = [
  'all', // Special option to detect all objects
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
  'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
  'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
  'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
  'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
  'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
  'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
  'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
];

interface NodeConfigPanelProps {
  node: Node;
  onClose: () => void;
  onUpdate: (nodeId: string, data: any) => void;
}

export function NodeConfigPanel({ node, onClose, onUpdate }: NodeConfigPanelProps) {
  const getNodeTypeColor = (type: string) => {
    switch (type) {
      case 'camera': return 'text-primary';
      case 'detection': return 'text-chart-3';
      case 'analysis': return 'text-chart-2';
      case 'action':
      case 'approval': return 'text-chart-4';
      case 'email':
      case 'sms':
      case 'call':
      case 'discord': return 'text-primary';
      default: return 'text-foreground';
    }
  };

  const getInputsOutputs = (type: string) => {
    switch (type) {
      case 'camera':
        return { inputs: [], outputs: ['image'] };
      case 'detection':
        return { inputs: ['image'], outputs: ['detections', 'image'] };
      case 'analysis':
        return { inputs: ['detections', 'image'], outputs: ['analysis'] };
      case 'approval':
        return { inputs: ['analysis'], outputs: ['approved'] };
      case 'action':
        return { inputs: ['approved'], outputs: [] };
      case 'email':
      case 'sms':
      case 'call':
      case 'discord':
        return { inputs: ['analysis'], outputs: [] };
      default:
        return { inputs: [], outputs: [] };
    }
  };

  const { inputs, outputs } = getInputsOutputs(node.type || 'default');

  return (
    <div className="absolute top-0 right-0 h-full w-80 bg-card border-l border-border shadow-lg z-10 overflow-y-auto" data-testid="node-config-panel">
      <div className="p-4 space-y-6">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold">Node Configuration</h3>
          <Button variant="ghost" size="icon" onClick={onClose} data-testid="button-close-config">
            <X className="w-4 h-4" />
          </Button>
        </div>

        <Separator />

        <div className="space-y-4">
          <div>
            <Label className="text-xs text-muted-foreground">Node Type</Label>
            <p className={`text-sm font-medium ${getNodeTypeColor(node.type || 'default')}`}>
              {node.type?.charAt(0).toUpperCase()}{node.type?.slice(1)}
            </p>
          </div>

          <div>
            <Label className="text-xs text-muted-foreground">Node ID</Label>
            <p className="text-sm font-mono">{node.id}</p>
          </div>

          <div>
            <Label htmlFor="node-label">Label</Label>
            <Input
              id="node-label"
              value={node.data.label}
              onChange={(e) => onUpdate(node.id, { ...node.data, label: e.target.value })}
              className="mt-1"
              data-testid="input-node-label"
            />
          </div>

          {/* Inputs */}
          <div>
            <Label className="text-xs text-muted-foreground mb-2 block">Inputs</Label>
            {inputs.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {inputs.map((input) => (
                  <Badge key={input} variant="outline" className="text-xs">
                    ← {input}
                  </Badge>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">No inputs</p>
            )}
          </div>

          {/* Outputs */}
          <div>
            <Label className="text-xs text-muted-foreground mb-2 block">Outputs</Label>
            {outputs.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {outputs.map((output) => (
                  <Badge key={output} variant="outline" className="text-xs">
                    {output} →
                  </Badge>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">No outputs</p>
            )}
          </div>

          {/* Node-specific configuration */}
          {node.type === 'camera' && (
            <>
              <div>
                <Label htmlFor="camera-input-mode">Video Source</Label>
                <Select
                  value={node.data.config?.inputMode || 'webcam'}
                  onValueChange={(value) => onUpdate(node.id, { 
                    ...node.data, 
                    config: { ...node.data.config, inputMode: value } 
                  })}
                >
                  <SelectTrigger id="camera-input-mode" className="mt-1" data-testid="select-input-mode">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="webcam">Local Webcam</SelectItem>
                    <SelectItem value="screen">Screen Capture</SelectItem>
                    <SelectItem value="video">Pre-recorded Video</SelectItem>
                    <SelectItem value="antmedia">Ant Media Server</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* NEW: Ant Media Configuration Fields */}
              {node.data.config?.inputMode === 'antmedia' && (
                <div className="space-y-3 p-3 bg-muted/30 border border-border rounded-md">
                  <div className="space-y-1.5">
                    <Label className="text-xs font-medium text-foreground">WebSocket URL</Label>
                    <Input
                      value={node.data.config?.amsUrl || 'ws://localhost:5080/LiveApp/websocket'}
                      onChange={(e) => onUpdate(node.id, { 
                        ...node.data, 
                        config: { ...node.data.config, amsUrl: e.target.value } 
                      })}
                      className="h-8 text-xs font-mono"
                      placeholder="ws://localhost:5080/LiveApp/websocket"
                    />
                  </div>
                  <div className="space-y-1.5">
                    <Label className="text-xs font-medium text-foreground">Stream ID</Label>
                    <Input
                      value={node.data.config?.streamId || 'stream1'}
                      onChange={(e) => onUpdate(node.id, { 
                        ...node.data, 
                        config: { ...node.data.config, streamId: e.target.value } 
                      })}
                      className="h-8 text-xs font-mono"
                      placeholder="stream1"
                    />
                  </div>
                </div>
              )}
              
              {/* Include Audio checkbox for Screen Share mode */}
              {node.data.config?.inputMode === 'screen' && (
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="include-screen-audio"
                    checked={node.data.config?.includeAudio !== false}
                    onChange={(e) => onUpdate(node.id, { 
                      ...node.data, 
                      config: { ...node.data.config, includeAudio: e.target.checked } 
                    })}
                    className="w-4 h-4 rounded border-border"
                    data-testid="checkbox-include-audio"
                  />
                  <label htmlFor="include-screen-audio" className="text-sm">
                    Include Audio (capture system/tab audio)
                  </label>
                </div>
              )}
              
              {/* Preview Controls for Webcam/Screen/AntMedia */}
              {(node.data.config?.inputMode === 'webcam' || node.data.config?.inputMode === 'screen' || node.data.config?.inputMode === 'antmedia' || !node.data.config?.inputMode) && (
                <div className="space-y-2">
                  <Label>Preview Control</Label>
                  <div className="flex gap-2">
                    <Button
                      variant="default"
                      size="sm"
                      className="flex-1"
                      data-testid="button-start-preview"
                      onClick={() => {
                        console.log('[CONFIG PANEL] Start Preview button clicked for node:', node.id);
                        const newData = { 
                          ...node.data, 
                          triggerPreview: Date.now() 
                        };
                        console.log('[CONFIG PANEL] Updating node data:', newData);
                        onUpdate(node.id, newData);
                      }}
                    >
                      {node.data.config?.inputMode === 'screen' ? 'Start Sharing' : node.data.config?.inputMode === 'antmedia' ? 'Connect Stream' : 'Start Preview'}
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="flex-1"
                      data-testid="button-stop-preview"
                      onClick={() => {
                        console.log('[CONFIG PANEL] Stop Preview button clicked for node:', node.id);
                        const newData = { 
                          ...node.data, 
                          triggerStop: Date.now() 
                        };
                        console.log('[CONFIG PANEL] Updating node data:', newData);
                        onUpdate(node.id, newData);
                      }}
                    >
                      {node.data.config?.inputMode === 'screen' ? 'Stop Sharing' : node.data.config?.inputMode === 'antmedia' ? 'Disconnect' : 'Stop Preview'}
                    </Button>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Click "{node.data.config?.inputMode === 'screen' ? 'Start Sharing' : node.data.config?.inputMode === 'antmedia' ? 'Connect Stream' : 'Start Preview'}" to activate the {node.data.config?.inputMode === 'screen' ? 'screen share' : node.data.config?.inputMode === 'antmedia' ? 'remote stream' : 'webcam'}
                    {node.data.config?.inputMode === 'screen' && node.data.config?.includeAudio && ' (remember to check "Share audio" in the dialog)'}
                  </p>
                </div>
              )}
              
              {node.data.config?.inputMode === 'video' && (
                <div>
                  <Label htmlFor="video-upload">Video File</Label>
                  <Input
                    id="video-upload"
                    type="file"
                    accept="video/*,.avi,.mp4,.webm,.mov,.mkv"
                    className="mt-1"
                    data-testid="input-video-upload"
                    onChange={async (e) => {
                      const file = e.target.files?.[0];
                      if (file) {
                        try {
                          console.log('[VIDEO_UPLOAD] Starting upload for file:', file.name, 'size:', file.size);
                          
                          // Clean up old URL if it exists
                          if (node.data.videoUrl) {
                            URL.revokeObjectURL(node.data.videoUrl);
                          }
                          
                          // Upload to backend
                          const formData = new FormData();
                          formData.append('video', file);
                          
                          console.log('[VIDEO_UPLOAD] Sending request to /api/media/videos');
                          const response = await fetch('/api/media/videos', {
                            method: 'POST',
                            body: formData,
                          });
                          
                          console.log('[VIDEO_UPLOAD] Response status:', response.status, response.statusText);
                          console.log('[VIDEO_UPLOAD] Response OK:', response.ok);
                          
                          if (!response.ok) {
                            const errorText = await response.text();
                            console.error('[VIDEO_UPLOAD] Server error response:', errorText);
                            throw new Error(`Failed to upload video: ${response.status} ${response.statusText}`);
                          }
                          
                          const result = await response.json();
                          console.log('[VIDEO_UPLOAD] Server response:', result);
                          
                          // Update node with video asset ID, server filename, and permanent video URL
                          // Use server filename (not asset ID) for permanent access that survives server restarts
                          const permanentVideoUrl = `/api/media/videos/${result.serverFileName}`;
                          
                          onUpdate(node.id, { 
                            ...node.data, 
                            config: { 
                              ...node.data.config, 
                              videoFileName: result.fileName, // Original filename
                              videoServerFileName: result.serverFileName, // Server-generated filename for persistence
                              videoAssetId: result.id, // Store asset ID for current session
                            },
                            videoUrl: permanentVideoUrl  // Use filename-based URL for permanent access
                          });
                          
                          console.log('[VIDEO_UPLOAD] Successfully uploaded video, asset ID:', result.id);
                        } catch (error) {
                          console.error('[VIDEO_UPLOAD] Error during upload:', error);
                          alert(`Failed to upload video. Error: ${error instanceof Error ? error.message : String(error)}`);
                        }
                      }
                    }}
                  />
                  {node.data.config?.videoFileName && (
                    <p className="text-xs text-muted-foreground mt-1">
                      Selected: {node.data.config.videoFileName}
                    </p>
                  )}
                  <p className="text-xs text-muted-foreground mt-1">
                    Video will be processed at 5 frames per second
                  </p>
                </div>
              )}
            </>
          )}

          {node.type === 'detection' && (
            <div className="space-y-2">
              <Label>Object Filter</Label>
              <div className="flex items-center gap-2 mb-2">
                <input
                  type="checkbox"
                  id="select-all-objects"
                  checked={(() => {
                    const filter = node.data.config?.objectFilter;
                    return !filter || filter === 'all' || (Array.isArray(filter) && filter.length === 0);
                  })()}
                  onChange={(e) => {
                    onUpdate(node.id, {
                      ...node.data,
                      config: { ...node.data.config, objectFilter: e.target.checked ? [] : ['person'] }
                    });
                  }}
                  className="h-4 w-4 rounded border-input"
                  data-testid="checkbox-all-objects"
                />
                <Label htmlFor="select-all-objects" className="text-sm font-medium cursor-pointer">
                  All Objects (80 classes)
                </Label>
              </div>
              {(() => {
                const filter = node.data.config?.objectFilter;
                const showList = filter && filter !== 'all' && (!Array.isArray(filter) || filter.length > 0);
                return showList;
              })() && (
                <div className="border rounded-md p-2 max-h-[200px] overflow-y-auto space-y-1">
                  {COCO_CLASSES.filter(c => c !== 'all').map((className) => {
                    const filter = node.data.config?.objectFilter;
                    // Handle both string and array formats
                    const selectedClasses = typeof filter === 'string' ? [filter] : (Array.isArray(filter) ? filter : []);
                    const isSelected = selectedClasses.includes(className);
                    
                    return (
                      <div key={className} className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          id={`object-${className}`}
                          checked={isSelected}
                          onChange={(e) => {
                            const currentFilter = Array.isArray(node.data.config?.objectFilter) 
                              ? node.data.config.objectFilter 
                              : [];
                            
                            const newFilter = e.target.checked
                              ? [...currentFilter, className]
                              : currentFilter.filter((c: string) => c !== className);
                            
                            onUpdate(node.id, {
                              ...node.data,
                              config: { ...node.data.config, objectFilter: newFilter }
                            });
                          }}
                          className="h-4 w-4 rounded border-input"
                          data-testid={`checkbox-${className}`}
                        />
                        <Label htmlFor={`object-${className}`} className="text-sm cursor-pointer">
                          {className}
                        </Label>
                      </div>
                    );
                  })}
                </div>
              )}
              <p className="text-xs text-muted-foreground">
                {(() => {
                  const filter = node.data.config?.objectFilter;
                  if (!filter || (Array.isArray(filter) && filter.length === 0)) {
                    return 'Detecting all 80 COCO object classes';
                  }
                  // Handle old string format (backwards compatibility)
                  if (typeof filter === 'string') {
                    return filter === 'all' ? 'Detecting all 80 COCO object classes' : `Only detecting: ${filter}`;
                  }
                  // Handle new array format
                  return `Detecting ${filter.length} object${filter.length !== 1 ? 's' : ''}: ${filter.join(', ')}`;
                })()}
              </p>
            </div>
          )}

          {node.type === 'transcription' && (
            <div className="space-y-3">
              <Label>Audio Source</Label>
              <p className="text-sm text-muted-foreground">
                Captures microphone audio for real-time transcription
              </p>
              <p className="text-xs text-muted-foreground mt-2">
                💡 To capture system/screen audio, use Camera node with Screen Share mode and enable "Include Audio"
              </p>
            </div>
          )}

          {node.type === 'analysis' && (
            <div className="space-y-4">
              <div>
                <Label htmlFor="analysis-prompt">User Prompt</Label>
                <Textarea
                  id="analysis-prompt"
                  value={node.data.config?.userPrompt || ''}
                  onChange={(e) => onUpdate(node.id, { 
                    ...node.data, 
                    config: { ...node.data.config, userPrompt: e.target.value } 
                  })}
                  className="mt-1 font-mono text-sm"
                  placeholder="Example prompts:&#10;• Check if there is a person in the image&#10;• Count how many laptops are visible&#10;• Is this a valid invoice with amount > $100?&#10;• Extract product name and price from receipt"
                  rows={6}
                  data-testid="input-user-prompt"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  💡 Gemini AI will check your condition and provide structured analysis
                </p>
                <div className="mt-2 p-2 bg-muted/50 rounded-md">
                  <p className="text-xs font-medium mb-1">Example conditions:</p>
                  <p className="text-xs text-muted-foreground">• "Is there a person present?"</p>
                  <p className="text-xs text-muted-foreground">• "Verify if invoice total exceeds $500"</p>
                  <p className="text-xs text-muted-foreground">• "Check for safety equipment (helmet, vest)"</p>
                </div>
              </div>
              
              <div>
                <Label htmlFor="trigger-phrase">Voice Trigger Phrase (for transcription flows)</Label>
                <Input
                  id="trigger-phrase"
                  value={node.data.config?.triggerPhrase || ''}
                  onChange={(e) => onUpdate(node.id, { 
                    ...node.data, 
                    config: { ...node.data.config, triggerPhrase: e.target.value } 
                  })}
                  className="mt-1"
                  placeholder="meeting ended"
                  data-testid="input-trigger-phrase"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  🎤 When this phrase is detected in the transcript, the flow will automatically process and send results
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  Example: "okay SyncroFlow, meeting ended" or "that's all for today"
                </p>
              </div>
            </div>
          )}

          {node.type === 'action' && (
            <div>
              <Label htmlFor="output-format">Output Format</Label>
              <Select
                value={node.data.config?.outputFormat || 'json'}
                onValueChange={(value) => onUpdate(node.id, { 
                  ...node.data, 
                  config: { ...node.data.config, outputFormat: value } 
                })}
              >
                <SelectTrigger id="output-format" className="mt-1" data-testid="select-output-format">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="json">JSON</SelectItem>
                  <SelectItem value="csv">CSV</SelectItem>
                </SelectContent>
              </Select>
            </div>
          )}

          {node.type === 'email' && (
            <div className="space-y-4">
              <div>
                <Label htmlFor="email-to">Recipient Email</Label>
                <Input
                  id="email-to"
                  type="email"
                  value={node.data.config?.to || ''}
                  onChange={(e) => onUpdate(node.id, { 
                    ...node.data, 
                    config: { ...node.data.config, to: e.target.value } 
                  })}
                  className="mt-1"
                  placeholder="user@example.com"
                  data-testid="input-email-to"
                />
              </div>
              <div>
                <Label htmlFor="email-subject">Subject</Label>
                <Input
                  id="email-subject"
                  value={node.data.config?.subject || ''}
                  onChange={(e) => onUpdate(node.id, { 
                    ...node.data, 
                    config: { ...node.data.config, subject: e.target.value } 
                  })}
                  className="mt-1"
                  placeholder="Alert: Detection triggered"
                  data-testid="input-email-subject"
                />
              </div>
              <div>
                <Label htmlFor="email-body">Message Template</Label>
                <Textarea
                  id="email-body"
                  value={node.data.config?.body || ''}
                  onChange={(e) => onUpdate(node.id, { 
                    ...node.data, 
                    config: { ...node.data.config, body: e.target.value } 
                  })}
                  className="mt-1"
                  placeholder="Detection result: {{analysis}}&#10;Timestamp: {{timestamp}}"
                  rows={4}
                  data-testid="input-email-body"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Use {'{{analysis}}'} and {'{{timestamp}}'} as placeholders
                </p>
              </div>
            </div>
          )}

          {node.type === 'sms' && (
            <div className="space-y-4">
              <div>
                <Label htmlFor="sms-to">Phone Number</Label>
                <Input
                  id="sms-to"
                  type="tel"
                  value={node.data.config?.to || ''}
                  onChange={(e) => onUpdate(node.id, { 
                    ...node.data, 
                    config: { ...node.data.config, to: e.target.value } 
                  })}
                  className="mt-1"
                  placeholder="+1234567890"
                  data-testid="input-sms-to"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Include country code (e.g., +1 for US)
                </p>
              </div>
              <div>
                <Label htmlFor="sms-message">Message Template</Label>
                <Textarea
                  id="sms-message"
                  value={node.data.config?.message || ''}
                  onChange={(e) => onUpdate(node.id, { 
                    ...node.data, 
                    config: { ...node.data.config, message: e.target.value } 
                  })}
                  className="mt-1"
                  placeholder="Alert: {{analysis}} at {{timestamp}}"
                  rows={3}
                  data-testid="input-sms-message"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Use {'{{analysis}}'} and {'{{timestamp}}'} as placeholders. Max 160 characters.
                </p>
              </div>
            </div>
          )}

          {node.type === 'call' && (
            <div className="space-y-4">
              <div>
                <Label htmlFor="call-to">Phone Number</Label>
                <Input
                  id="call-to"
                  type="tel"
                  value={node.data.config?.to || ''}
                  onChange={(e) => onUpdate(node.id, { 
                    ...node.data, 
                    config: { ...node.data.config, to: e.target.value } 
                  })}
                  className="mt-1"
                  placeholder="+1234567890"
                  data-testid="input-call-to"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Include country code (e.g., +1 for US)
                </p>
              </div>
              <div>
                <Label htmlFor="call-message">Voice Message</Label>
                <Textarea
                  id="call-message"
                  value={node.data.config?.message || ''}
                  onChange={(e) => onUpdate(node.id, { 
                    ...node.data, 
                    config: { ...node.data.config, message: e.target.value } 
                  })}
                  className="mt-1"
                  placeholder="Alert! Detection triggered. Analysis result: {{analysis}}"
                  rows={3}
                  data-testid="input-call-message"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Use {'{{analysis}}'} and {'{{timestamp}}'} as placeholders. Text-to-speech will read this message.
                </p>
              </div>
            </div>
          )}

          {node.type === 'discord' && (
            <div className="space-y-4">
              <div>
                <Label htmlFor="discord-webhook">Webhook URL</Label>
                <Input
                  id="discord-webhook"
                  type="url"
                  value={node.data.config?.webhookUrl || ''}
                  onChange={(e) => onUpdate(node.id, { 
                    ...node.data, 
                    config: { ...node.data.config, webhookUrl: e.target.value } 
                  })}
                  className="mt-1"
                  placeholder="https://discord.com/api/webhooks/..."
                  data-testid="input-discord-webhook"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Get webhook URL from Discord Server Settings → Integrations
                </p>
              </div>
              <div>
                <Label htmlFor="discord-message">Message Template</Label>
                <Textarea
                  id="discord-message"
                  value={node.data.config?.message || ''}
                  onChange={(e) => onUpdate(node.id, { 
                    ...node.data, 
                    config: { ...node.data.config, message: e.target.value } 
                  })}
                  className="mt-1"
                  placeholder="🚨 **Alert** 🚨&#10;Detection: {{analysis}}&#10;Time: {{timestamp}}"
                  rows={4}
                  data-testid="input-discord-message"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Use {'{{analysis}}'} and {'{{timestamp}}'} as placeholders. Supports Discord markdown.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}