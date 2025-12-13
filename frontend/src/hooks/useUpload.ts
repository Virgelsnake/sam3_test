import { useMutation } from '@tanstack/react-query';
import { uploadVideo } from '@/services/api';
import type { UploadResponse } from '@/types';

export function useUpload() {
  return useMutation<UploadResponse, Error, File>({
    mutationFn: uploadVideo,
  });
}
