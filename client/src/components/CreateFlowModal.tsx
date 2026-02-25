import { useState } from 'react';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Sparkles, FileEdit } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { apiRequest } from '@/lib/queryClient';

interface CreateFlowModalProps {
  open: boolean;
  onClose: () => void;
  onFlowCreated: (flowId: string) => void;
}

export function CreateFlowModal({ open, onClose, onFlowCreated }: CreateFlowModalProps) {
  const [mode, setMode] = useState<'choice' | 'scratch' | 'generate'>('choice');
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const { toast } = useToast();

  const handleReset = () => {
    setMode('choice');
    setPrompt('');
    setIsGenerating(false);
  };

  const handleClose = () => {
    handleReset();
    onClose();
  };

  const handleStartFromScratch = async () => {
    // Create flow with single camera node
    const flowId = `flow-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
    
    const { createFlow } = await import('@/lib/flowStorage');
    createFlow({
      id: flowId,
      name: 'New Flow',
      nodes: [
        {
          id: 'camera-1',
          type: 'camera',
          position: { x: 250, y: 200 },
          data: { label: 'Video Source', config: { inputMode: 'webcam' } },
        },
      ],
      edges: [],
    });
    
    onFlowCreated(flowId);
    handleClose();
    toast({
      title: 'Flow Created',
      description: 'Camera node added - connect more nodes to build your workflow',
    });
  };

  const handleGenerateFlow = async () => {
    if (!prompt.trim()) {
      toast({
        title: 'Prompt Required',
        description: 'Please describe the workflow you want to create',
        variant: 'destructive',
      });
      return;
    }

    setIsGenerating(true);
    try {
      const response = await apiRequest('POST', '/api/generate-flow', { 
        prompt: prompt.trim() 
      });

      const result = await response.json();

      if (result.flow) {
        const flowId = `flow-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
        
        // Save generated flow to localStorage
        const { createFlow } = await import('@/lib/flowStorage');
        createFlow({
          id: flowId,
          name: result.flow.name || 'Generated Flow',
          nodes: result.flow.nodes || [],
          edges: result.flow.edges || [],
        });

        onFlowCreated(flowId);
        handleClose();
        
        toast({
          title: 'Flow Generated',
          description: `Created "${result.flow.name}" with ${result.flow.nodes?.length || 0} nodes`,
        });
      } else {
        throw new Error('Invalid flow response');
      }
    } catch (error: any) {
      toast({
        title: 'Generation Failed',
        description: error.message || 'Failed to generate flow',
        variant: 'destructive',
      });
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={(open) => !open && handleClose()}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Create New Flow</DialogTitle>
          <DialogDescription>
            {mode === 'choice' && 'Choose how you want to create your workflow'}
            {mode === 'generate' && 'Describe the workflow you want to create'}
          </DialogDescription>
        </DialogHeader>

        {mode === 'choice' && (
          <div className="space-y-3 py-4">
            <Button
              variant="outline"
              className="w-full h-24 flex flex-col gap-2"
              onClick={handleStartFromScratch}
              data-testid="button-create-scratch"
            >
              <FileEdit className="h-6 w-6" />
              <div className="text-center">
                <div className="font-semibold">Start from Scratch</div>
                <div className="text-xs text-muted-foreground">
                  Build your workflow manually
                </div>
              </div>
            </Button>

            <Button
              variant="outline"
              className="w-full h-24 flex flex-col gap-2"
              onClick={() => setMode('generate')}
              data-testid="button-create-ai"
            >
              <Sparkles className="h-6 w-6" />
              <div className="text-center">
                <div className="font-semibold">Generate with AI</div>
                <div className="text-xs text-muted-foreground">
                  Describe your workflow and let AI build it
                </div>
              </div>
            </Button>
          </div>
        )}

        {mode === 'generate' && (
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Workflow Description</label>
              <Textarea
                placeholder="Example: Create a flow that captures my webcam, detects people, and saves results when more than 2 people are detected"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={5}
                className="resize-none"
                data-testid="input-flow-prompt"
              />
              <p className="text-xs text-muted-foreground">
                Be specific about what you want to detect, analyze, or automate
              </p>
            </div>

            <div className="flex gap-2">
              <Button
                variant="outline"
                onClick={() => setMode('choice')}
                className="flex-1"
                disabled={isGenerating}
                data-testid="button-back"
              >
                Back
              </Button>
              <Button
                onClick={handleGenerateFlow}
                className="flex-1"
                disabled={isGenerating || !prompt.trim()}
                data-testid="button-generate"
              >
                {isGenerating ? (
                  <>
                    <Sparkles className="h-4 w-4 mr-2 animate-pulse" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4 mr-2" />
                    Generate Flow
                  </>
                )}
              </Button>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
