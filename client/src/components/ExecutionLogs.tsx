import { useState } from 'react';
import { ChevronDown, ChevronRight, Clock, CheckCircle, XCircle, AlertCircle } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import type { Execution } from '@shared/schema';
import { format } from 'date-fns';

interface ExecutionLogsProps {
  executions: Execution[];
}

export function ExecutionLogs({ executions }: ExecutionLogsProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-chart-2" />;
      case 'failed':
        return <XCircle className="w-4 h-4 text-destructive" />;
      case 'running':
        return <Clock className="w-4 h-4 text-primary animate-pulse" />;
      case 'pending_approval':
        return <AlertCircle className="w-4 h-4 text-chart-4" />;
      default:
        return <Clock className="w-4 h-4 text-muted-foreground" />;
    }
  };

  const getStatusBadgeVariant = (status: string) => {
    switch (status) {
      case 'completed':
        return 'default';
      case 'failed':
        return 'destructive';
      case 'running':
        return 'default';
      case 'pending_approval':
        return 'secondary';
      default:
        return 'secondary';
    }
  };

  return (
    <div className="space-y-2" data-testid="execution-logs">
      {executions.length === 0 && (
        <Card className="p-12 text-center">
          <Clock className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
          <h3 className="text-lg font-medium mb-2">No Executions Yet</h3>
          <p className="text-sm text-muted-foreground">
            Run a workflow to see execution logs here
          </p>
        </Card>
      )}

      {executions.map((execution) => (
        <Card key={execution.id} className="p-4 hover-elevate">
          <div className="flex items-start justify-between gap-4">
            <div className="flex items-start gap-3 flex-1">
              <div className="mt-0.5">{getStatusIcon(execution.status)}</div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1 flex-wrap">
                  <h4 className="font-medium text-sm" data-testid={`text-execution-${execution.id}`}>
                    {execution.flowName}
                  </h4>
                  <Badge variant={getStatusBadgeVariant(execution.status)} className="text-xs" data-testid={`badge-status-${execution.id}`}>
                    {execution.status.replace('_', ' ')}
                  </Badge>
                </div>
                <div className="text-xs text-muted-foreground font-mono">
                  {execution.startedAt ? format(new Date(execution.startedAt), 'MMM dd, yyyy · HH:mm:ss') : 'N/A'}
                </div>
                {execution.currentNode && (
                  <div className="text-xs text-muted-foreground mt-1">
                    Current: {execution.currentNode}
                  </div>
                )}
              </div>
            </div>

            <Button
              variant="ghost"
              size="icon"
              onClick={() => setExpandedId(expandedId === execution.id ? null : execution.id)}
              data-testid={`button-toggle-${execution.id}`}
            >
              {expandedId === execution.id ? (
                <ChevronDown className="w-4 h-4" />
              ) : (
                <ChevronRight className="w-4 h-4" />
              )}
            </Button>
          </div>

          {expandedId === execution.id && (
            <div className="mt-4 pt-4 border-t border-border space-y-3">
              {execution.detections && Array.isArray(execution.detections) && execution.detections.length > 0 && (
                <div>
                  <div className="text-xs font-medium text-muted-foreground mb-2">
                    Detections
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {(execution.detections as any[]).map((det: any, idx: number) => (
                      <Badge key={idx} variant="outline" className="font-mono text-xs">
                        {det.class || 'Object'} · {Math.round((det.confidence || 0) * 100)}%
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {execution.analysis && (
                <div>
                  <div className="text-xs font-medium text-muted-foreground mb-2">
                    Analysis
                  </div>
                  <Card className="p-3 bg-muted text-xs font-mono">
                    <pre className="whitespace-pre-wrap">
                      {JSON.stringify(execution.analysis, null, 2)}
                    </pre>
                  </Card>
                </div>
              )}

              {execution.error && (
                <div>
                  <div className="text-xs font-medium text-destructive mb-2">
                    Error
                  </div>
                  <Card className="p-3 bg-destructive/10 text-xs text-destructive">
                    {execution.error}
                  </Card>
                </div>
              )}
            </div>
          )}
        </Card>
      ))}
    </div>
  );
}
