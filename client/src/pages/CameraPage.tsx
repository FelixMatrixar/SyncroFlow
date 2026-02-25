import { useState } from 'react';
import { CameraCapture } from '@/components/CameraCapture';
import { ApprovalCard } from '@/components/ApprovalCard';
import type { DetectedObject, AnalysisResult } from '@shared/schema';
import { useToast } from '@/hooks/use-toast';
import { useMutation } from '@tanstack/react-query';
import { apiRequest } from '@/lib/queryClient';
import { Loader2 } from 'lucide-react';
import { Card } from '@/components/ui/card';

export default function CameraPage() {
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [detections, setDetections] = useState<DetectedObject[]>([]);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [showApproval, setShowApproval] = useState(false);
  const { toast } = useToast();

  const detectMutation = useMutation({
    mutationFn: async (imageData: string) => {
      // Convert base64 to blob
      const base64Data = imageData.split(',')[1];
      const byteCharacters = atob(base64Data);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: 'image/jpeg' });

      // Create form data
      const formData = new FormData();
      formData.append('image', blob, 'capture.jpg');

      // Send to backend
      const response = await fetch('/api/detect', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Detection failed');
      }

      return response.json();
    },
    onSuccess: (data) => {
      setDetections(data.detections || []);
      // Trigger analysis
      analyzeMutation.mutate({
        image: capturedImage!,
        detections: data.detections || [],
        context: 'invoice',
      });
    },
    onError: (error) => {
      toast({
        title: 'Detection Failed',
        description: error.message,
        variant: 'destructive',
      });
      setCapturedImage(null);
    },
  });

  const analyzeMutation = useMutation({
    mutationFn: async (data: { image: string; detections: DetectedObject[]; context: string }) => {
      return apiRequest('POST', '/api/analyze', data);
    },
    onSuccess: (data) => {
      setAnalysis(data.analysis);
      setShowApproval(true);
    },
    onError: (error: any) => {
      toast({
        title: 'Analysis Failed',
        description: error.message,
        variant: 'destructive',
      });
    },
  });

  const handleCapture = (image: string) => {
    setCapturedImage(image);
    toast({
      title: 'Processing Image',
      description: 'Detecting objects and analyzing...',
    });
    detectMutation.mutate(image);
  };

  const handleApprove = () => {
    toast({
      title: 'Approved',
      description: 'Processing approved. Data has been saved.',
    });
    setShowApproval(false);
    setCapturedImage(null);
    setDetections([]);
    setAnalysis(null);
  };

  const handleReject = () => {
    toast({
      title: 'Rejected',
      description: 'Processing cancelled.',
      variant: 'destructive',
    });
    setShowApproval(false);
    setCapturedImage(null);
    setDetections([]);
    setAnalysis(null);
  };

  const isProcessing = detectMutation.isPending || analyzeMutation.isPending;

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="container max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-semibold mb-2">Video Capture</h1>
          <p className="text-muted-foreground">
            Capture documents and images for automated processing
          </p>
        </div>

        <CameraCapture onCapture={handleCapture} />

        {isProcessing && capturedImage && (
          <Card className="mt-6 p-12 text-center">
            <Loader2 className="w-12 h-12 text-primary mx-auto mb-4 animate-spin" />
            <h3 className="text-lg font-medium mb-2">Processing Image</h3>
            <p className="text-sm text-muted-foreground">
              {detectMutation.isPending && 'Detecting objects...'}
              {analyzeMutation.isPending && 'Analyzing with AI...'}
            </p>
          </Card>
        )}

        {showApproval && capturedImage && !isProcessing && (
          <ApprovalCard
            image={capturedImage}
            detections={detections}
            analysis={analysis || undefined}
            onApprove={handleApprove}
            onReject={handleReject}
          />
        )}
      </div>
    </div>
  );
}
