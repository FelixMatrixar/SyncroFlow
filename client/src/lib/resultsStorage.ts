// Results storage in localStorage per flow
export interface FlowResult {
  id: string;
  flowId: string;
  flowName: string;
  timestamp: string;
  detections?: Array<{
    class: string;
    confidence: number;
    bbox: number[];
  }>;
  transcript?: string;
  analysis?: {
    condition_met: boolean;
    summary: string;
    extractedData?: Record<string, any>;
    confidence?: number;
  };
}

const RESULTS_KEY_PREFIX = 'flow_results_';
const MAX_RESULTS_PER_FLOW = 100; // Keep last 100 results per flow

export function saveFlowResult(result: Omit<FlowResult, 'id'>): FlowResult {
  const fullResult: FlowResult = {
    ...result,
    id: `result-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
  };

  const storageKey = `${RESULTS_KEY_PREFIX}${result.flowId}`;
  const existingResults = getFlowResults(result.flowId);
  
  // Add new result at the beginning
  const updatedResults = [fullResult, ...existingResults];
  
  // Keep only the last MAX_RESULTS_PER_FLOW results
  const trimmedResults = updatedResults.slice(0, MAX_RESULTS_PER_FLOW);
  
  localStorage.setItem(storageKey, JSON.stringify(trimmedResults));
  
  console.log('[RESULTS] Saved result:', fullResult.id, 'Total:', trimmedResults.length);
  
  return fullResult;
}

export function getFlowResults(flowId: string): FlowResult[] {
  const storageKey = `${RESULTS_KEY_PREFIX}${flowId}`;
  const stored = localStorage.getItem(storageKey);
  
  if (!stored) return [];
  
  try {
    return JSON.parse(stored);
  } catch {
    return [];
  }
}

export function getAllResults(): FlowResult[] {
  const allResults: FlowResult[] = [];
  
  // Get all flow result keys
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key?.startsWith(RESULTS_KEY_PREFIX)) {
      const flowResults = JSON.parse(localStorage.getItem(key) || '[]');
      allResults.push(...flowResults);
    }
  }
  
  // Sort by timestamp (newest first)
  return allResults.sort((a, b) => 
    new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
  );
}

export function clearFlowResults(flowId: string): void {
  const storageKey = `${RESULTS_KEY_PREFIX}${flowId}`;
  localStorage.removeItem(storageKey);
  console.log('[RESULTS] Cleared results for flow:', flowId);
}

export function deleteResult(flowId: string, resultId: string): void {
  const results = getFlowResults(flowId);
  const filtered = results.filter(r => r.id !== resultId);
  
  const storageKey = `${RESULTS_KEY_PREFIX}${flowId}`;
  localStorage.setItem(storageKey, JSON.stringify(filtered));
  
  console.log('[RESULTS] Deleted result:', resultId);
}

export function exportFlowResults(flowId: string, format: 'json' | 'csv' = 'json'): string {
  const results = getFlowResults(flowId);
  
  if (format === 'json') {
    return JSON.stringify(results, null, 2);
  } else {
    // CSV format - properly escape quotes and newlines
    const escapeCSV = (value: string): string => {
      return `"${value.replace(/"/g, '""').replace(/\n/g, ' ').replace(/\r/g, '')}"`;
    };
    
    const headers = ['Timestamp', 'Detections', 'Transcript', 'Condition Met', 'Summary'];
    const rows = results.map(r => [
      r.timestamp,
      r.detections?.map(d => `${d.class} (${(d.confidence * 100).toFixed(0)}%)`).join('; ') || 'N/A',
      r.transcript || 'N/A',
      r.analysis?.condition_met ? 'YES' : 'NO',
      r.analysis?.summary || 'N/A'
    ]);
    
    return [
      headers.join(','),
      ...rows.map(row => row.map(cell => escapeCSV(cell)).join(','))
    ].join('\n');
  }
}
