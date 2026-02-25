import { Camera, Search, Brain, Play, Mail, MessageSquare, Phone, Send, Mic, User } from 'lucide-react';
import { Card } from '@/components/ui/card';

export function NodePalette() {
  const nodeTypes = [
    {
      type: 'camera',
      icon: Camera,
      label: 'Video Source',
      description: 'Capture from camera',
      color: 'text-emerald-500',
    },
    {
      type: 'transcription',
      icon: Mic,
      label: 'Audio Transcription',
      description: 'Speech-to-text',
      color: 'text-purple-500',
    },
    {
      type: 'detection',
      icon: Search,
      label: 'Object Detection',
      description: 'Gemini Vision detection',
      color: 'text-cyan-500',
    },
    {
      type: 'pose',
      icon: User,
      label: 'Pose Detection',
      description: 'Human pose estimation',
      color: 'text-pink-500',
    },
    {
      type: 'analysis',
      icon: Brain,
      label: 'AI Analysis',
      description: 'Gemini LLM analysis',
      color: 'text-blue-500',
    },
    {
      type: 'action',
      icon: Play,
      label: 'Save Results',
      description: 'Save to file',
      color: 'text-orange-500',
    },
    {
      type: 'email',
      icon: Mail,
      label: 'Send Email',
      description: 'Email notification',
      color: 'text-red-500',
    },
    {
      type: 'sms',
      icon: MessageSquare,
      label: 'Send SMS',
      description: 'Text message alert',
      color: 'text-yellow-500',
    },
    {
      type: 'call',
      icon: Phone,
      label: 'Make Call',
      description: 'Voice call alert',
      color: 'text-indigo-500',
    },
    {
      type: 'discord',
      icon: Send,
      label: 'Discord Message',
      description: 'Discord notification',
      color: 'text-violet-500',
    },
  ];

  return (
    <div className="p-4 space-y-4 overflow-y-auto flex-1">
      <div>
        <h3 className="text-sm font-semibold mb-3">Flow Nodes</h3>
        <div className="space-y-2">
          {nodeTypes.map((nodeType) => {
            const Icon = nodeType.icon;
            return (
              <Card
                key={nodeType.type}
                className="p-3 hover-elevate cursor-grab"
                draggable
                onDragStart={(e) => {
                  e.dataTransfer.setData('application/reactflow', nodeType.type);
                  e.dataTransfer.effectAllowed = 'move';
                }}
              >
                <div className="flex items-center gap-2">
                  <Icon className={`w-4 h-4 ${nodeType.color}`} />
                  <div>
                    <div className="text-sm font-medium">{nodeType.label}</div>
                    <div className="text-xs text-muted-foreground">{nodeType.description}</div>
                  </div>
                </div>
              </Card>
            );
          })}
        </div>
      </div>

      <div>
        <h3 className="text-sm font-semibold mb-3">Instructions</h3>
        <Card className="p-3">
          <div className="text-xs text-muted-foreground space-y-2">
            <p>1. Drag nodes to canvas</p>
            <p>2. Connect nodes</p>
            <p>3. Click nodes to configure</p>
            <p>4. Click "Run Flow"</p>
          </div>
        </Card>
      </div>
    </div>
  );
}
