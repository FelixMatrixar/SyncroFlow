import { Check, X, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import type { DetectedObject, AnalysisResult } from '@shared/schema';

interface ApprovalCardProps {
  image: string;
  detections: DetectedObject[];
  analysis?: AnalysisResult;
  onApprove: () => void;
  onReject: () => void;
}

export function ApprovalCard({
  image,
  detections,
  analysis,
  onApprove,
  onReject,
}: ApprovalCardProps) {
  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 z-50" data-testid="approval-modal">
      <Card className="max-w-2xl w-full p-6 shadow-2xl">
        <div className="flex items-center gap-2 mb-6">
          <AlertCircle className="w-6 h-6 text-chart-4" />
          <h2 className="text-xl font-semibold">Human Approval Required</h2>
        </div>

        <div className="space-y-6">
          {/* Image with detections */}
          <div className="relative aspect-video bg-muted rounded-lg overflow-hidden">
            <img src={image} alt="Detection preview" className="w-full h-full object-contain" data-testid="img-approval-preview" />
            
            {/* Bounding boxes overlay */}
            <svg className="absolute inset-0 w-full h-full pointer-events-none">
              {detections.map((detection, idx) => (
                <g key={idx}>
                  <rect
                    x={`${detection.bbox[0]}%`}
                    y={`${detection.bbox[1]}%`}
                    width={`${detection.bbox[2]}%`}
                    height={`${detection.bbox[3]}%`}
                    fill="none"
                    stroke="hsl(280, 100%, 70%)"
                    strokeWidth="2"
                    strokeDasharray="4 2"
                  />
                  <text
                    x={`${detection.bbox[0]}%`}
                    y={`${detection.bbox[1] - 1}%`}
                    className="text-xs font-mono fill-white"
                    style={{ textShadow: '0 0 4px rgba(0,0,0,0.8)' }}
                  >
                    {detection.class} ({Math.round(detection.confidence * 100)}%)
                  </text>
                </g>
              ))}
            </svg>
          </div>

          {/* Detected Objects */}
          <div>
            <h3 className="text-sm font-medium mb-2">Detected Objects</h3>
            <div className="flex flex-wrap gap-2">
              {detections.map((detection, idx) => (
                <Badge key={idx} variant="secondary" className="font-mono text-xs" data-testid={`badge-detection-${idx}`}>
                  {detection.class} · {Math.round(detection.confidence * 100)}%
                </Badge>
              ))}
            </div>
          </div>

          {/* Analysis Results */}
          {analysis && (
            <div>
              <h3 className="text-sm font-medium mb-2">AI Analysis</h3>
              <Card className="p-4 bg-muted">
                <p className="text-sm mb-3">{analysis.summary}</p>
                {analysis.extractedData && (
                  <div className="space-y-1">
                    {Object.entries(analysis.extractedData).map(([key, value]) => (
                      <div key={key} className="flex gap-2 text-xs">
                        <span className="text-muted-foreground font-medium">{key}:</span>
                        <span className="font-mono" data-testid={`text-extracted-${key}`}>{String(value)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </Card>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex gap-3 pt-2">
            <Button
              onClick={onApprove}
              className="flex-1"
              variant="default"
              data-testid="button-approve"
            >
              <Check className="w-4 h-4 mr-2" />
              Approve & Continue
            </Button>
            <Button
              onClick={onReject}
              className="flex-1"
              variant="destructive"
              data-testid="button-reject"
            >
              <X className="w-4 h-4 mr-2" />
              Reject
            </Button>
          </div>
        </div>
      </Card>
    </div>
  );
}
