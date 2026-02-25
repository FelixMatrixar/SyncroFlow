import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { FileJson, RefreshCw, CheckCircle2, XCircle, Download } from 'lucide-react';
import { apiRequest } from '@/lib/queryClient';

interface SavedResult {
  filename: string;
  size: number;
  modified: string;
}

interface ResultData {
  detections: Array<{
    class: string;
    confidence: number;
    bbox: number[];
  }>;
  analysis: {
    condition_met: boolean;
    summary: string;
    extractedData?: any;
    confidence: number;
  } | null;
  timestamp: string;
  userPrompt: string;
}

export function ResultsViewer() {
  const [results, setResults] = useState<SavedResult[]>([]);
  const [selectedResult, setSelectedResult] = useState<ResultData | null>(null);
  const [loading, setLoading] = useState(false);

  const loadResults = async () => {
    setLoading(true);
    try {
      const response = await apiRequest<{ files: SavedResult[] }>('GET', '/api/results');
      setResults(response.files || []);
    } catch (error) {
      console.error('Failed to load results:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadResults();
    const interval = setInterval(loadResults, 5000);
    return () => clearInterval(interval);
  }, []);

  const viewResult = async (filename: string) => {
    try {
      const response = await fetch(`/results/${filename}`);
      const data = await response.json();
      setSelectedResult(data);
    } catch (error) {
      console.error('Failed to load result:', error);
    }
  };

  return (
    <div className="grid grid-cols-2 gap-4 h-full">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between gap-4 space-y-0 pb-4">
          <div>
            <CardTitle>Saved Results</CardTitle>
            <CardDescription>View pipeline execution results</CardDescription>
          </div>
          <Button
            size="sm"
            variant="outline"
            onClick={loadResults}
            disabled={loading}
            data-testid="button-refresh-results"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          </Button>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[400px]">
            {results.length === 0 ? (
              <p className="text-sm text-muted-foreground text-center py-8">
                No results yet. Run a flow to generate results.
              </p>
            ) : (
              <div className="space-y-2">
                {results.map((result) => (
                  <div
                    key={result.filename}
                    className="flex items-center justify-between p-3 rounded-md border hover-elevate active-elevate-2 cursor-pointer"
                    onClick={() => viewResult(result.filename)}
                    data-testid={`result-${result.filename}`}
                  >
                    <div className="flex items-center gap-3">
                      <FileJson className="h-5 w-5 text-blue-500" />
                      <div>
                        <p className="text-sm font-medium">{result.filename}</p>
                        <p className="text-xs text-muted-foreground">
                          {new Date(result.modified).toLocaleString()}
                        </p>
                      </div>
                    </div>
                    <Badge variant="secondary">{Math.round(result.size / 1024)}KB</Badge>
                  </div>
                ))}
              </div>
            )}
          </ScrollArea>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Result Details</CardTitle>
          <CardDescription>
            {selectedResult ? 'Analysis results' : 'Select a result to view details'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {selectedResult ? (
            <ScrollArea className="h-[400px]">
              <div className="space-y-4">
                <div>
                  <h4 className="text-sm font-semibold mb-2">Detections ({selectedResult.detections?.length || 0})</h4>
                  {selectedResult.detections && selectedResult.detections.length > 0 ? (
                    <div className="space-y-2">
                      {selectedResult.detections.map((det, idx) => (
                        <div key={idx} className="flex items-center gap-2 p-2 rounded bg-muted">
                          <Badge variant="outline">{det.class}</Badge>
                          <span className="text-sm text-muted-foreground">
                            {(det.confidence * 100).toFixed(0)}% confidence
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-muted-foreground">No detections</p>
                  )}
                </div>

                {selectedResult.analysis && (
                  <div>
                    <h4 className="text-sm font-semibold mb-2 flex items-center gap-2">
                      Analysis
                      {selectedResult.analysis.condition_met ? (
                        <CheckCircle2 className="h-4 w-4 text-green-500" />
                      ) : (
                        <XCircle className="h-4 w-4 text-red-500" />
                      )}
                    </h4>
                    <div className="space-y-2">
                      <div className="p-3 rounded bg-muted">
                        <p className="text-sm font-medium mb-1">Prompt:</p>
                        <p className="text-sm text-muted-foreground">{selectedResult.userPrompt}</p>
                      </div>
                      <div className="p-3 rounded bg-muted">
                        <p className="text-sm font-medium mb-1">Summary:</p>
                        <p className="text-sm">{selectedResult.analysis.summary}</p>
                      </div>
                      {selectedResult.analysis.extractedData && (
                        <div className="p-3 rounded bg-muted">
                          <p className="text-sm font-medium mb-1">Extracted Data:</p>
                          <pre className="text-xs">
                            {JSON.stringify(selectedResult.analysis.extractedData, null, 2)}
                          </pre>
                        </div>
                      )}
                      <Badge variant="secondary">
                        Confidence: {(selectedResult.analysis.confidence * 100).toFixed(0)}%
                      </Badge>
                    </div>
                  </div>
                )}

                <div>
                  <p className="text-xs text-muted-foreground">
                    Timestamp: {new Date(selectedResult.timestamp).toLocaleString()}
                  </p>
                </div>
              </div>
            </ScrollArea>
          ) : (
            <div className="flex items-center justify-center h-[400px] text-muted-foreground">
              <p className="text-sm">No result selected</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
