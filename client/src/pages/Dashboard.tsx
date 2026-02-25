import { useState, useEffect, useRef } from 'react';
import { Activity, Plus, Play, Trash2, FileText, Download, Upload } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useLocation } from 'wouter';
import { CreateFlowModal } from '@/components/CreateFlowModal';
import { getAllFlows, deleteFlow, setCurrentFlowId, migrateLegacyStorage, exportFlow, importFlow, type SavedFlow } from '@/lib/flowStorage';
import { useToast } from '@/hooks/use-toast';

export default function Dashboard() {
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  const [flows, setFlows] = useState<SavedFlow[]>([]);
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [runningFlows, setRunningFlows] = useState<Set<string>>(new Set());
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load flows on mount and migrate legacy storage
  useEffect(() => {
    migrateLegacyStorage();
    loadFlows();
  }, []);

  // Check for running video upload flows
  useEffect(() => {
    const checkRunningFlows = () => {
      const running = new Set<string>();
      
      flows.forEach(flow => {
        // Check if this flow is marked as running in sessionStorage
        const isRunning = sessionStorage.getItem(`flow_${flow.id}_running`) === 'true';
        
        // Check if it's a video upload mode flow
        const cameraNode = flow.nodes.find(n => n.type === 'camera');
        const isVideoMode = (cameraNode?.data as any)?.config?.inputMode === 'video';
        
        if (isRunning && isVideoMode) {
          running.add(flow.id);
        }
      });
      
      setRunningFlows(running);
    };

    checkRunningFlows();
    
    // Poll every 2 seconds to update running status
    const interval = setInterval(checkRunningFlows, 2000);
    return () => clearInterval(interval);
  }, [flows]);

  const loadFlows = () => {
    const allFlows = getAllFlows();
    setFlows(allFlows);
  };

  const handleFlowCreated = (flowId: string) => {
    loadFlows();
    setCurrentFlowId(flowId);
    setLocation(`/flow-editor?id=${flowId}`);
  };

  const handleOpenFlow = (flowId: string) => {
    setCurrentFlowId(flowId);
    setLocation(`/flow-editor?id=${flowId}`);
  };

  const handleDeleteFlow = (flowId: string, flowName: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirm(`Are you sure you want to delete "${flowName}"?`)) {
      deleteFlow(flowId);
      loadFlows();
      toast({
        title: 'Flow Deleted',
        description: `"${flowName}" has been deleted`,
      });
    }
  };

  const handleExportFlow = (flowId: string, flowName: string, e: React.MouseEvent) => {
    e.stopPropagation();
    exportFlow(flowId);
    toast({
      title: 'Flow Exported',
      description: `"${flowName}" has been exported as JSON`,
    });
  };

  const handleImportClick = () => {
    fileInputRef.current?.click();
  };

  const handleImportFlow = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      const newFlow = await importFlow(file);
      loadFlows();
      toast({
        title: 'Flow Imported',
        description: `"${newFlow.name}" has been imported successfully`,
      });
    } catch (error) {
      toast({
        title: 'Import Failed',
        description: error instanceof Error ? error.message : 'Failed to import flow',
        variant: 'destructive',
      });
    }

    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="container max-w-7xl mx-auto p-6 space-y-8 animate-fade-in">
        {/* Header */}
        <div className="animate-slide-down">
          <h1 className="text-4xl font-bold mb-2 tracking-tight" data-testid="text-dashboard-title">
            <span className="gradient-text">Multimodal Flows</span>
          </h1>
          <p className="text-muted-foreground text-lg">
            Automation powered by vision and sound
          </p>
        </div>

        {/* Flows Section */}
        <div>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold">My Flows</h2>
            <div className="flex gap-2">
              <input
                ref={fileInputRef}
                type="file"
                accept=".json,application/json"
                onChange={handleImportFlow}
                className="hidden"
                data-testid="input-import-flow"
              />
              <Button 
                variant="secondary"
                onClick={handleImportClick}
                data-testid="button-import-flow"
              >
                <Upload className="w-4 h-4 mr-2" />
                Import Flow
              </Button>
              <Button 
                onClick={() => setIsCreateModalOpen(true)}
                data-testid="button-create-flow"
              >
                <Plus className="w-4 h-4 mr-2" />
                Create Flow
              </Button>
            </div>
          </div>

          {flows.length === 0 ? (
            <Card className="p-12 text-center animate-scale-in">
              <FileText className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-medium mb-2">No flows yet</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Create your first automation flow to get started
              </p>
              <Button onClick={() => setIsCreateModalOpen(true)} data-testid="button-create-first-flow" className="transition-spring">
                <Plus className="w-4 h-4 mr-2" />
                Create Your First Flow
              </Button>
            </Card>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {flows.map((flow, index) => (
                <Card 
                  key={flow.id} 
                  className="p-6 hover-elevate cursor-pointer transition-smooth hover:scale-[1.02] animate-slide-up"
                  style={{ animationDelay: `${index * 50}ms` }}
                  onClick={() => handleOpenFlow(flow.id)}
                  data-testid={`card-flow-${flow.id}`}
                >
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <h3 className="font-semibold text-lg" data-testid={`text-flow-name-${flow.id}`}>
                          {flow.name}
                        </h3>
                        {runningFlows.has(flow.id) && (
                          <Badge 
                            variant="default" 
                            className="bg-green-600 hover:bg-green-700 text-white animate-pulse"
                            data-testid={`badge-flow-running-${flow.id}`}
                          >
                            <Activity className="w-3 h-3 mr-1" />
                            Running
                          </Badge>
                        )}
                      </div>
                      {flow.description && (
                        <p className="text-sm text-muted-foreground line-clamp-2">
                          {flow.description}
                        </p>
                      )}
                    </div>
                    <Button
                      size="icon"
                      variant="ghost"
                      onClick={(e) => handleDeleteFlow(flow.id, flow.name, e)}
                      data-testid={`button-delete-flow-${flow.id}`}
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>

                  <div className="flex items-center justify-between text-sm text-muted-foreground">
                    <div className="flex items-center gap-4">
                      <span>{flow.nodes.length} nodes</span>
                      <span>{flow.edges.length} connections</span>
                    </div>
                  </div>

                  <div className="mt-4 pt-4 border-t border-border">
                    <p className="text-xs text-muted-foreground">
                      Updated {new Date(flow.updatedAt).toLocaleDateString()}
                    </p>
                  </div>

                  <div className="mt-4 flex gap-2">
                    <Button 
                      size="sm" 
                      className="flex-1 transition-spring hover:shadow-lg"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleOpenFlow(flow.id);
                      }}
                      data-testid={`button-open-flow-${flow.id}`}
                    >
                      <Play className="w-4 h-4 mr-2" />
                      Open Flow
                    </Button>
                    <Button 
                      size="sm"
                      variant="secondary"
                      onClick={(e) => handleExportFlow(flow.id, flow.name, e)}
                      data-testid={`button-export-flow-${flow.id}`}
                    >
                      <Download className="w-4 h-4" />
                    </Button>
                  </div>
                </Card>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Create Flow Modal */}
      <CreateFlowModal
        open={isCreateModalOpen}
        onClose={() => setIsCreateModalOpen(false)}
        onFlowCreated={handleFlowCreated}
      />
    </div>
  );
}
