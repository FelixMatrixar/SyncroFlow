import {
  type Flow,
  type InsertFlow,
  type Execution,
  type InsertExecution,
  type Detection,
  type InsertDetection,
  type VideoAsset,
  type InsertVideoAsset,
} from "@shared/schema";
import { randomUUID } from "crypto";

export interface IStorage {
  // Flow operations
  getFlow(id: string): Promise<Flow | undefined>;
  getAllFlows(): Promise<Flow[]>;
  createFlow(flow: InsertFlow): Promise<Flow>;
  updateFlow(id: string, flow: Partial<InsertFlow>): Promise<Flow | undefined>;
  deleteFlow(id: string): Promise<boolean>;

  // Execution operations
  getExecution(id: string): Promise<Execution | undefined>;
  getAllExecutions(): Promise<Execution[]>;
  createExecution(execution: InsertExecution): Promise<Execution>;
  updateExecution(
    id: string,
    execution: Partial<InsertExecution>
  ): Promise<Execution | undefined>;

  // Detection operations
  getDetection(id: string): Promise<Detection | undefined>;
  getDetectionsByExecution(executionId: string): Promise<Detection[]>;
  createDetection(detection: InsertDetection): Promise<Detection>;

  // Video asset operations
  getVideoAsset(id: string): Promise<VideoAsset | undefined>;
  createVideoAsset(asset: InsertVideoAsset): Promise<VideoAsset>;
  deleteVideoAsset(id: string): Promise<boolean>;
}

export class MemStorage implements IStorage {
  private flows: Map<string, Flow>;
  private executions: Map<string, Execution>;
  private detections: Map<string, Detection>;
  private videoAssets: Map<string, VideoAsset>;

  constructor() {
    this.flows = new Map();
    this.executions = new Map();
    this.detections = new Map();
    this.videoAssets = new Map();
  }

  // Flow operations
  async getFlow(id: string): Promise<Flow | undefined> {
    return this.flows.get(id);
  }

  async getAllFlows(): Promise<Flow[]> {
    return Array.from(this.flows.values());
  }

  async createFlow(insertFlow: InsertFlow): Promise<Flow> {
    const id = randomUUID();
    const now = new Date();
    const flow: Flow = {
      ...insertFlow,
      id,
      createdAt: now,
      updatedAt: now,
    };
    this.flows.set(id, flow);
    return flow;
  }

  async updateFlow(
    id: string,
    updates: Partial<InsertFlow>
  ): Promise<Flow | undefined> {
    const flow = this.flows.get(id);
    if (!flow) return undefined;

    const updated: Flow = {
      ...flow,
      ...updates,
      updatedAt: new Date(),
    };
    this.flows.set(id, updated);
    return updated;
  }

  async deleteFlow(id: string): Promise<boolean> {
    return this.flows.delete(id);
  }

  // Execution operations
  async getExecution(id: string): Promise<Execution | undefined> {
    return this.executions.get(id);
  }

  async getAllExecutions(): Promise<Execution[]> {
    return Array.from(this.executions.values()).sort(
      (a, b) => b.startedAt.getTime() - a.startedAt.getTime()
    );
  }

  async createExecution(insertExecution: InsertExecution): Promise<Execution> {
    const id = randomUUID();
    const execution: Execution = {
      ...insertExecution,
      id,
      startedAt: new Date(),
      completedAt: null,
    };
    this.executions.set(id, execution);
    return execution;
  }

  async updateExecution(
    id: string,
    updates: Partial<InsertExecution>
  ): Promise<Execution | undefined> {
    const execution = this.executions.get(id);
    if (!execution) return undefined;

    const updated: Execution = {
      ...execution,
      ...updates,
    };
    this.executions.set(id, updated);
    return updated;
  }

  // Detection operations
  async getDetection(id: string): Promise<Detection | undefined> {
    return this.detections.get(id);
  }

  async getDetectionsByExecution(executionId: string): Promise<Detection[]> {
    return Array.from(this.detections.values()).filter(
      (d) => d.executionId === executionId
    );
  }

  async createDetection(insertDetection: InsertDetection): Promise<Detection> {
    const id = randomUUID();
    const detection: Detection = {
      ...insertDetection,
      id,
      createdAt: new Date(),
    };
    this.detections.set(id, detection);
    return detection;
  }

  // Video asset operations
  async getVideoAsset(id: string): Promise<VideoAsset | undefined> {
    return this.videoAssets.get(id);
  }

  async createVideoAsset(insertAsset: InsertVideoAsset): Promise<VideoAsset> {
    const id = randomUUID();
    const asset: VideoAsset = {
      ...insertAsset,
      id,
      createdAt: new Date(),
    };
    this.videoAssets.set(id, asset);
    return asset;
  }

  async deleteVideoAsset(id: string): Promise<boolean> {
    return this.videoAssets.delete(id);
  }
}

export const storage = new MemStorage();
