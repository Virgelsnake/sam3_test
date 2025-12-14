import axios from 'axios';
import type { Job, JobStatus, UploadResponse, JobCreateRequest } from '@/types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export async function uploadVideo(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post<UploadResponse>('/api/uploads', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
}

export async function createJob(videoId: string, prompt: string): Promise<Job> {
  const payload: JobCreateRequest = {
    video_id: videoId,
    prompt,
  };

  const response = await api.post<Job>('/api/jobs', payload);
  return response.data;
}

export async function getJob(jobId: string): Promise<Job> {
  const response = await api.get<Job>(`/api/jobs/${jobId}`);
  return response.data;
}

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  const response = await api.get<JobStatus>(`/api/jobs/${jobId}/status`);
  return response.data;
}

export async function cancelJob(jobId: string): Promise<void> {
  await api.delete(`/api/jobs/${jobId}`);
}

export async function listJobs(status?: string): Promise<Job[]> {
  const params = status ? { status } : {};
  const response = await api.get<Job[]>('/api/jobs', { params });
  return response.data;
}

export async function healthCheck(): Promise<{ status: string }> {
  const response = await api.get<{ status: string }>('/api/health');
  return response.data;
}

// ==================== Image Batch API ====================

import type { ImageJob, ImageUploadResponse, ImageJobCreateRequest } from '@/types';

export async function uploadImages(files: File[]): Promise<ImageUploadResponse[]> {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append('files', file);
  });

  const response = await api.post<ImageUploadResponse[]>('/api/images/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
}

export async function createImageJob(imageIds: string[], prompt: string): Promise<ImageJob> {
  const payload: ImageJobCreateRequest = {
    image_ids: imageIds,
    prompt,
  };

  const response = await api.post<ImageJob>('/api/images/jobs', payload);
  return response.data;
}

export async function getImageJob(jobId: string): Promise<ImageJob> {
  const response = await api.get<ImageJob>(`/api/images/jobs/${jobId}`);
  return response.data;
}

export async function listImageJobs(status?: string): Promise<ImageJob[]> {
  const params = status ? { status } : {};
  const response = await api.get<ImageJob[]>('/api/images/jobs', { params });
  return response.data;
}

export async function updateImageJobInventory(
  jobId: string,
  userInventory: Record<string, number>
): Promise<void> {
  await api.patch(`/api/images/jobs/${jobId}/inventory`, { user_inventory: userInventory });
}
