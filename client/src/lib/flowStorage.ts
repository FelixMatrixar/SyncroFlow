import { Node, Edge } from '@xyflow/react';

export interface SavedFlow {
  id: string;
  name: string;
  description?: string;
  nodes: Node[];
  edges: Edge[];
  createdAt: string;
  updatedAt: string;
}

const FLOWS_STORAGE_KEY = 'SyncroFlow-flows';
const CURRENT_FLOW_ID_KEY = 'SyncroFlow-current-flow-id';

// Get all flows from localStorage
export function getAllFlows(): SavedFlow[] {
  try {
    const data = localStorage.getItem(FLOWS_STORAGE_KEY);
    if (!data) return [];
    return JSON.parse(data);
  } catch (error) {
    console.error('[FLOW_STORAGE] Error loading flows:', error);
    return [];
  }
}

// Get a single flow by ID
export function getFlow(id: string): SavedFlow | null {
  const flows = getAllFlows();
  return flows.find(f => f.id === id) || null;
}

// Create a new flow explicitly
export function createFlow(flow: { id: string; name: string; nodes: Node[]; edges: Edge[]; description?: string }): SavedFlow {
  return saveFlow(flow);
}

// Save a flow (create or update)
export function saveFlow(flow: Omit<SavedFlow, 'id' | 'createdAt' | 'updatedAt'> & { id?: string }): SavedFlow {
  const flows = getAllFlows();
  const now = new Date().toISOString();
  
  if (flow.id) {
    // Update existing flow
    const index = flows.findIndex(f => f.id === flow.id);
    if (index !== -1) {
      const updatedFlow: SavedFlow = {
        ...flows[index],
        ...flow,
        id: flow.id,
        updatedAt: now,
      };
      flows[index] = updatedFlow;
      localStorage.setItem(FLOWS_STORAGE_KEY, JSON.stringify(flows));
      console.log('[FLOW_STORAGE] Updated flow:', updatedFlow.id);
      return updatedFlow;
    }
  }
  
  // Create new flow
  const newFlow: SavedFlow = {
    ...flow,
    id: flow.id || `flow-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
    createdAt: now,
    updatedAt: now,
  };
  flows.push(newFlow);
  localStorage.setItem(FLOWS_STORAGE_KEY, JSON.stringify(flows));
  console.log('[FLOW_STORAGE] Created flow:', newFlow.id);
  return newFlow;
}

// Delete a flow
export function deleteFlow(id: string): boolean {
  const flows = getAllFlows();
  const filteredFlows = flows.filter(f => f.id !== id);
  
  if (filteredFlows.length === flows.length) {
    console.warn('[FLOW_STORAGE] Flow not found:', id);
    return false;
  }
  
  localStorage.setItem(FLOWS_STORAGE_KEY, JSON.stringify(filteredFlows));
  console.log('[FLOW_STORAGE] Deleted flow:', id);
  
  // Clear current flow ID if it was deleted
  if (getCurrentFlowId() === id) {
    clearCurrentFlowId();
  }
  
  return true;
}

// Get current flow ID
export function getCurrentFlowId(): string | null {
  return localStorage.getItem(CURRENT_FLOW_ID_KEY);
}

// Set current flow ID
export function setCurrentFlowId(id: string): void {
  localStorage.setItem(CURRENT_FLOW_ID_KEY, id);
  console.log('[FLOW_STORAGE] Set current flow ID:', id);
}

// Clear current flow ID
export function clearCurrentFlowId(): void {
  localStorage.removeItem(CURRENT_FLOW_ID_KEY);
  console.log('[FLOW_STORAGE] Cleared current flow ID');
}

// Create default flow template
export function createDefaultFlow(name: string = 'New Automation Flow'): SavedFlow {
  const defaultNodes: Node[] = [
    {
      id: 'camera-1',
      type: 'camera',
      position: { x: 100, y: 100 },
      data: { label: 'Video Source', config: {} },
    },
    {
      id: 'detect-1',
      type: 'detection',
      position: { x: 400, y: 100 },
      data: { label: 'Object Detection', config: { objectFilter: 'all' } },
    },
    {
      id: 'analyze-1',
      type: 'analysis',
      position: { x: 700, y: 100 },
      data: { label: 'AI Analysis', config: { userPrompt: 'Analyze the detected objects' } },
    },
    {
      id: 'save-1',
      type: 'action',
      position: { x: 1000, y: 100 },
      data: { label: 'Save Results', config: { outputFormat: 'json' } },
    },
  ];

  const defaultEdges: Edge[] = [
    { id: 'e1-2', source: 'camera-1', target: 'detect-1' },
    { id: 'e2-3', source: 'detect-1', target: 'analyze-1' },
    { id: 'e3-4', source: 'analyze-1', target: 'save-1' },
  ];

  return saveFlow({
    name,
    description: 'AI automation pipeline with vision and sound capabilities',
    nodes: defaultNodes,
    edges: defaultEdges,
  });
}

// Export flow to JSON file
export function exportFlow(flowId: string): void {
  const flow = getFlow(flowId);
  if (!flow) {
    console.error('[FLOW_STORAGE] Cannot export - flow not found:', flowId);
    return;
  }
  
  // Create JSON blob
  const json = JSON.stringify(flow, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  
  // Create download link
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${flow.name.replace(/[^a-z0-9]/gi, '-').toLowerCase()}-${Date.now()}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  
  console.log('[FLOW_STORAGE] Exported flow:', flowId);
}

// Import flow from JSON file
export async function importFlow(file: File): Promise<SavedFlow> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    
    reader.onload = (e) => {
      try {
        const content = e.target?.result as string;
        const importedFlow = JSON.parse(content);
        
        // Validate required fields
        if (!importedFlow.name || !importedFlow.nodes || !importedFlow.edges) {
          throw new Error('Invalid flow file: missing required fields (name, nodes, edges)');
        }
        
        // Create new flow with a new ID to avoid conflicts
        const newFlow = saveFlow({
          name: `${importedFlow.name} (Imported)`,
          description: importedFlow.description || 'Imported flow',
          nodes: importedFlow.nodes,
          edges: importedFlow.edges,
        });
        
        console.log('[FLOW_STORAGE] Imported flow:', newFlow.id);
        resolve(newFlow);
      } catch (error) {
        console.error('[FLOW_STORAGE] Import failed:', error);
        reject(error);
      }
    };
    
    reader.onerror = () => {
      reject(new Error('Failed to read file'));
    };
    
    reader.readAsText(file);
  });
}

// Migrate old storage keys to new SyncroFlow branding
export function migrateLegacyStorage(): void {
  // First migrate old camera-copilot-flows to SyncroFlow-flows
  const oldFlowsKey = 'camera-copilot-flows';
  const oldFlowsData = localStorage.getItem(oldFlowsKey);
  
  if (oldFlowsData && !localStorage.getItem(FLOWS_STORAGE_KEY)) {
    localStorage.setItem(FLOWS_STORAGE_KEY, oldFlowsData);
    localStorage.removeItem(oldFlowsKey);
    console.log('[FLOW_STORAGE] Migrated camera-copilot-flows to SyncroFlow-flows');
  }
  
  // Migrate current flow ID
  const oldCurrentFlowIdKey = 'camera-copilot-current-flow-id';
  const oldCurrentFlowId = localStorage.getItem(oldCurrentFlowIdKey);
  
  if (oldCurrentFlowId && !localStorage.getItem(CURRENT_FLOW_ID_KEY)) {
    localStorage.setItem(CURRENT_FLOW_ID_KEY, oldCurrentFlowId);
    localStorage.removeItem(oldCurrentFlowIdKey);
    console.log('[FLOW_STORAGE] Migrated camera-copilot-current-flow-id to SyncroFlow-current-flow-id');
  }
  
  // Then migrate old single-flow storage to new multi-flow storage
  const legacyKey = 'camera-copilot-flow';
  const legacyData = localStorage.getItem(legacyKey);
  
  if (legacyData && getAllFlows().length === 0) {
    try {
      const parsed = JSON.parse(legacyData);
      const migratedFlow = saveFlow({
        name: parsed.name || 'Migrated Flow',
        description: 'Automatically migrated from previous version',
        nodes: parsed.nodes || [],
        edges: parsed.edges || [],
      });
      setCurrentFlowId(migratedFlow.id);
      localStorage.removeItem(legacyKey);
      console.log('[FLOW_STORAGE] Migrated legacy flow:', migratedFlow.id);
    } catch (error) {
      console.error('[FLOW_STORAGE] Failed to migrate legacy flow:', error);
    }
  }
}
