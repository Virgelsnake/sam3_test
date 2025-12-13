import { useMutation, useQuery } from '@tanstack/react-query';
import { createJob, getJob, getJobStatus, cancelJob, listJobs } from '@/services/api';
import type { Job, JobStatus } from '@/types';

export function useCreateJob() {
  return useMutation<Job, Error, { videoId: string; prompt: string }>({
    mutationFn: ({ videoId, prompt }) => createJob(videoId, prompt),
  });
}

export function useJob(jobId: string | null, enabled = true) {
  return useQuery<Job, Error>({
    queryKey: ['job', jobId],
    queryFn: () => getJob(jobId!),
    enabled: enabled && !!jobId,
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data) return false;
      // Poll every 2 seconds while processing
      if (data.status === 'pending' || data.status === 'processing') {
        return 2000;
      }
      return false;
    },
  });
}

export function useJobStatus(jobId: string | null, enabled = true) {
  return useQuery<JobStatus, Error>({
    queryKey: ['jobStatus', jobId],
    queryFn: () => getJobStatus(jobId!),
    enabled: enabled && !!jobId,
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data) return false;
      // Poll every 2 seconds while processing
      if (data.status === 'pending' || data.status === 'processing') {
        return 2000;
      }
      return false;
    },
  });
}

export function useCancelJob() {
  return useMutation<void, Error, string>({
    mutationFn: cancelJob,
  });
}

export function useJobs(status?: string) {
  return useQuery<Job[], Error>({
    queryKey: ['jobs', status],
    queryFn: () => listJobs(status),
  });
}
