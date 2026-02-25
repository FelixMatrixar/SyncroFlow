import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { getFlowResults, clearFlowResults, exportFlowResults, type FlowResult } from '@/lib/resultsStorage';
import { Download, Trash2, CheckCircle, XCircle } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface FlowResultsProps {
  flowId: string | null;
  flowName: string;
}

export function FlowResults({ flowId, flowName }: FlowResultsProps) {
  const { toast } = useToast();
  const [results, setResults] = useState<FlowResult[]>([]);

  // Poll for new results every 2 seconds
  useEffect(() => {
    if (!flowId) {
      setResults([]);
      return;
    }

    const loadResults = () => {
      const freshResults = getFlowResults(flowId);
      setResults(freshResults);
    };

    // Load immediately
    loadResults();

    // Poll every 2 seconds for updates
    const interval = setInterval(loadResults, 2000);

    return () => clearInterval(interval);
  }, [flowId]);

  const handleExport = (format: 'json' | 'csv') => {
    if (!flowId) return;
    
    const data = exportFlowResults(flowId, format);
    const blob = new Blob([data], { type: format === 'json' ? 'application/json' : 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${flowName}_results_${new Date().toISOString().split('T')[0]}.${format}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    toast({
      title: 'Results Exported',
      description: `Downloaded ${results.length} results as ${format.toUpperCase()}`,
    });
  };

  const handleClear = () => {
    if (!flowId) return;
    
    if (confirm('Are you sure you want to clear all results for this flow?')) {
      clearFlowResults(flowId);
      toast({
        title: 'Results Cleared',
        description: 'All saved results have been removed',
      });
      window.location.reload(); // Refresh to show empty state
    }
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  if (!flowId) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Results</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">No flow selected</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="flex-row items-center justify-between space-y-0 pb-4">
        <CardTitle>Flow Results ({results.length})</CardTitle>
        <div className="flex gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={() => handleExport('json')}
            disabled={results.length === 0}
            data-testid="button-export-json"
          >
            <Download className="h-4 w-4 mr-1" />
            JSON
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => handleExport('csv')}
            disabled={results.length === 0}
            data-testid="button-export-csv"
          >
            <Download className="h-4 w-4 mr-1" />
            CSV
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={handleClear}
            disabled={results.length === 0}
            data-testid="button-clear-results"
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="flex-1 overflow-auto">
        {results.length === 0 ? (
          <div className="text-center py-8">
            <p className="text-sm text-muted-foreground">No results yet</p>
            <p className="text-xs text-muted-foreground mt-1">
              Results are saved automatically every 10 seconds during flow execution
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {results.map((result) => {
              const bgColor = result.analysis?.condition_met 
                ? 'bg-green-500/10 border-green-500/30' 
                : result.analysis?.condition_met === false 
                  ? 'bg-red-500/10 border-red-500/30'
                  : 'border';
              
              return (<div
                key={result.id}
                className={`${bgColor} rounded-lg p-3 space-y-2`}
                data-testid={`result-${result.id}`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">
                    {formatTimestamp(result.timestamp)}
                  </span>
                  {result.analysis && (
                    <div className="flex items-center gap-1">
                      {result.analysis.condition_met ? (
                        <CheckCircle className="h-4 w-4 text-green-500" />
                      ) : (
                        <XCircle className="h-4 w-4 text-red-500" />
                      )}
                      <span className="text-xs font-medium">
                        {result.analysis.condition_met ? 'MET' : 'NOT MET'}
                      </span>
                    </div>
                  )}
                </div>
                
                {result.transcript && (
                  <div className="text-sm bg-purple-500/5 border border-purple-500/20 rounded p-2" data-testid="result-transcript">
                    <div className="font-medium text-purple-600 dark:text-purple-400 mb-1">Transcript:</div>
                    <div className="text-muted-foreground whitespace-pre-wrap max-h-40 overflow-y-auto" data-testid="text-transcript-content">
                      {result.transcript}
                    </div>
                  </div>
                )}
                
                {result.analysis && (
                  <div className="text-sm text-muted-foreground">
                    {result.analysis.summary}
                  </div>
                )}
                
                {result.analysis?.extractedData && Object.keys(result.analysis.extractedData).length > 0 && (
                  <div className="text-xs">
                    <span className="font-medium">Extracted Data:</span>
                    <div className="mt-1 space-y-1">
                      {Object.entries(result.analysis.extractedData).map(([key, value]) => (
                        <div key={key} className="flex gap-2">
                          <span className="text-muted-foreground">{key}:</span>
                          <span>{String(value)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>);
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
