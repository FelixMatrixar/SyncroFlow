import { Switch, Route, Link, useLocation } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ThemeProvider } from "@/components/ThemeProvider";
import { ThemeToggle } from "@/components/ThemeToggle";
import { LayoutDashboard, ArrowLeft } from "lucide-react";
import Dashboard from "@/pages/Dashboard";
import FlowEditor from "@/pages/FlowEditor";
import NotFound from "@/pages/not-found";
import { Button } from "@/components/ui/button";
import { useState } from "react";

function Navigation() {
  const [location, setLocation] = useLocation();
  const isFlowEditor = location.startsWith('/flow-editor') || location.startsWith('/flow/');

  const handleNavigateToDashboard = () => {
    // Check if any flow is currently running
    const allKeys = Object.keys(sessionStorage);
    const runningFlowKey = allKeys.find(key => 
      key.endsWith('_running') && sessionStorage.getItem(key) === 'true'
    );

    if (runningFlowKey) {
      const flowId = runningFlowKey.replace('flow_', '').replace('_running', '');
      
      // Check localStorage to see if this flow uses video upload mode
      const flowsData = localStorage.getItem('SyncroFlow-flows');
      let isVideoUploadMode = false;
      
      if (flowsData) {
        try {
          const flows = JSON.parse(flowsData);
          const currentFlow = flows.find((f: any) => f.id === flowId);
          if (currentFlow) {
            const cameraNode = currentFlow.nodes.find((n: any) => n.type === 'camera');
            isVideoUploadMode = cameraNode?.data?.config?.inputMode === 'video';
          }
        } catch (e) {
          console.error('[NAV] Error checking flow config:', e);
        }
      }
      
      console.log('[NAV] Navigation check:', {
        flowId,
        isVideoUploadMode
      });
      
      if (isVideoUploadMode) {
        // Video upload flows can persist - navigate freely
        console.log('[NAV] Video upload flow - allowing navigation without warning');
        setLocation('/');
      } else {
        // Webcam/screen flows need confirmation
        console.log('[NAV] Live camera flow - showing confirmation dialog');
        const confirmed = confirm(
          'A live camera flow is currently running. Navigating away will stop the execution. Do you want to continue?'
        );
        
        if (confirmed) {
          // Clear the running state for non-video flows
          sessionStorage.removeItem(`flow_${flowId}_running`);
          sessionStorage.removeItem(`flow_${flowId}_executionId`);
          setLocation('/');
        }
      }
    } else {
      setLocation('/');
    }
  };

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 h-16 border-b border-border/50 bg-card/95 backdrop-blur-xl px-6 flex items-center justify-between shadow-sm">
      <div className="flex items-center gap-8">
        <span 
          onClick={handleNavigateToDashboard}
          className="flex items-center gap-3 font-extrabold text-3xl cursor-pointer transition-smooth tracking-tight" 
          data-testid="link-home"
        >
          <span className="gradient-text">SyncroFlow</span>
        </span>
      </div>
      <ThemeToggle />
    </nav>
  );
}

function Router() {
  return (
    <Switch>
      <Route path="/" component={Dashboard} />
      <Route path="/flow-editor" component={FlowEditor} />
      <Route path="/flow/:id" component={FlowEditor} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider defaultTheme="dark">
        <TooltipProvider>
          <div className="min-h-screen bg-background">
            <Navigation />
            <div className="pt-16">
              <Router />
            </div>
          </div>
          <Toaster />
        </TooltipProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
