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
  size: number;
}

export interface JobCreateRequest {
  video_id: string;
  prompt: string;
}

export interface ApiError {
  detail: string;
}
