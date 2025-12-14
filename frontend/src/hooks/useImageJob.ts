import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { uploadImages, createImageJob, getImageJob, listImageJobs } from '@/services/api';
import type { ImageJob, ImageUploadResponse } from '@/types';

export function useImageUpload() {
  return useMutation<ImageUploadResponse[], Error, File[]>({
    mutationFn: uploadImages,
  });
}

export function useCreateImageJob() {
  const queryClient = useQueryClient();

  return useMutation<ImageJob, Error, { imageIds: string[]; prompt: string }>({
    mutationFn: ({ imageIds, prompt }) => createImageJob(imageIds, prompt),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['imageJobs'] });
    },
  });
}

export function useImageJob(jobId: string | null, enabled: boolean = true) {
  return useQuery<ImageJob>({
    queryKey: ['imageJob', jobId],
    queryFn: () => getImageJob(jobId!),
    enabled: enabled && !!jobId,
    refetchInterval: (query) => {
      const data = query.state.data;
      if (data?.status === 'completed' || data?.status === 'failed') {
        return false;
      }
      return 2000; // Poll every 2 seconds while processing
    },
  });
}

export function useImageJobs() {
  return useQuery<ImageJob[]>({
    queryKey: ['imageJobs'],
    queryFn: () => listImageJobs(),
  });
}
