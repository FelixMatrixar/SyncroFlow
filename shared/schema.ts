import { sql } from "drizzle-orm";
import { pgTable, text, varchar, timestamp, integer, jsonb } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// Flow definition schema
export const flows = pgTable("flows", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: text("name").notNull(),
  description: text("description"),
  nodes: jsonb("nodes").notNull().default('[]'),
  edges: jsonb("edges").notNull().default('[]'),
  createdAt: timestamp("created_at").notNull().defaultNow(),
  updatedAt: timestamp("updated_at").notNull().defaultNow(),
});

// Execution log schema
export const executions = pgTable("executions", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  flowId: varchar("flow_id").references(() => flows.id),
  flowName: text("flow_name").notNull(),
  status: text("status").notNull(), // 'running', 'completed', 'failed', 'pending_approval'
  startedAt: timestamp("started_at").notNull().defaultNow(),
  completedAt: timestamp("completed_at"),
  currentNode: text("current_node"),
  detections: jsonb("detections").default('[]'),
  analysis: jsonb("analysis"),
  error: text("error"),
});

// Detection result schema
export const detections = pgTable("detections", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  executionId: varchar("execution_id").references(() => executions.id),
  imagePath: text("image_path").notNull(),
  detectedObjects: jsonb("detected_objects").notNull(),
  analysis: jsonb("analysis"),
  confidence: integer("confidence"),
  createdAt: timestamp("created_at").notNull().defaultNow(),
});

// Video assets schema
export const videoAssets = pgTable("video_assets", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  flowId: varchar("flow_id"),
  fileName: text("file_name").notNull(),
  mimeType: text("mime_type").notNull(),
  size: integer("size").notNull(),
  storagePath: text("storage_path").notNull(),
  createdAt: timestamp("created_at").notNull().defaultNow(),
});

export const insertFlowSchema = createInsertSchema(flows).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertExecutionSchema = createInsertSchema(executions).omit({
  id: true,
  startedAt: true,
  completedAt: true,
});

export const insertDetectionSchema = createInsertSchema(detections).omit({
  id: true,
  createdAt: true,
});

export const insertVideoAssetSchema = createInsertSchema(videoAssets).omit({
  id: true,
  createdAt: true,
});

export type InsertFlow = z.infer<typeof insertFlowSchema>;
export type Flow = typeof flows.$inferSelect;
export type InsertExecution = z.infer<typeof insertExecutionSchema>;
export type Execution = typeof executions.$inferSelect;
export type InsertDetection = z.infer<typeof insertDetectionSchema>;
export type Detection = typeof detections.$inferSelect;
export type InsertVideoAsset = z.infer<typeof insertVideoAssetSchema>;
export type VideoAsset = typeof videoAssets.$inferSelect;

// Frontend-only types for React Flow
export interface FlowNode {
  id: string;
  type: 'camera' | 'detection' | 'analysis' | 'action' | 'approval' | 'transcription';
  position: { x: number; y: number };
  data: {
    label: string;
    config?: Record<string, any>;
  };
}

export interface FlowEdge {
  id: string;
  source: string;
  target: string;
  animated?: boolean;
}

export interface DetectedObject {
  class: string;
  confidence: number;
  bbox: [number, number, number, number]; // [x, y, width, height]
}

export interface AnalysisResult {
  summary: string;
  extractedData?: Record<string, any>;
  suggestions?: string[];
}

export interface ExecutionUpdate {
  executionId: string;
  status: string;
  currentNode?: string;
  detections?: DetectedObject[];
  analysis?: AnalysisResult;
  error?: string;
}
