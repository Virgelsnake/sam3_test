export type JobState = 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled';

export interface Job {
  id: string;
  status: JobState;
  prompt: string;
  video_path: string;
  progress: number;
  mask_video_url?: string;
  composite_video_url?: string;
  frame_count?: number;
  objects_detected?: number;
  inventory?: Record<string, number>;  // Item name -> count
  inventory_colors?: Record<string, string>;  // Item name -> hex color
  error_message?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
}

export interface JobStatus {
  id: string;
  status: JobState;
  progress: number;
  message?: string;
}

export interface UploadResponse {
  video_id: string;
  url: string;
  filename: string;
  size_mb: number;
  path: string;
}

export interface JobCreateRequest {
  video_id: string;
  prompt: string;
}

export interface ApiError {
  detail: string;
}
