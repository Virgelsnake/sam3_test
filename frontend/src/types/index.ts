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
  inventory?: Record<string, number>;  // Item name -> count (AI-detected)
  inventory_colors?: Record<string, string>;  // Item name -> hex color
  user_inventory?: Record<string, number>;  // User-corrected counts
  error_message?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  // Image batch fields (also present on Job for unified history)
  job_type?: 'video' | 'image_batch';
  image_count?: number;
  image_paths?: string[];
  composite_images?: string[];
  per_image_results?: Array<{
    image_idx: number;
    objects_found: number;
    categories: string[];
  }>;
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

// ==================== Image Batch Types ====================

export interface ItemBbox {
  category: string;
  bbox: { x: number; y: number; width: number; height: number };
  color: string;
}

export interface PerImageResult {
  image_idx: number;
  objects_found: number;
  categories: string[];
  item_bboxes?: ItemBbox[];
}

export interface ImageJob {
  id: string;
  status: JobState;
  prompt: string;
  job_type: 'image_batch';
  image_count: number;
  image_paths: string[];
  progress: number;
  composite_images?: string[];
  objects_detected?: number;
  inventory?: Record<string, number>;
  inventory_colors?: Record<string, string>;
  user_inventory?: Record<string, number>;
  per_image_results?: PerImageResult[];
  error_message?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
}

export interface ImageUploadResponse {
  image_id: string;
  filename: string;
  size_mb: number;
  url: string;
  path: string;
}

export interface ImageJobCreateRequest {
  image_ids: string[];
  prompt: string;
}
