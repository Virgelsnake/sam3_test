import { useCallback, useState, useRef } from 'react';
import { Upload, X, Film } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { cn } from '@/lib/utils';

interface VideoUploadProps {
  onUpload: (file: File) => void;
  isUploading: boolean;
  maxSizeMB?: number;
  maxDurationSeconds?: number;
}

const ALLOWED_TYPES = ['video/mp4', 'video/webm', 'video/quicktime'];

export function VideoUpload({
  onUpload,
  isUploading,
  maxSizeMB = 100,
  maxDurationSeconds = 60,
}: VideoUploadProps) {
  const [dragActive, setDragActive] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  const validateFile = useCallback(
    async (file: File): Promise<string | null> => {
      if (!ALLOWED_TYPES.includes(file.type)) {
        return 'Please upload an MP4, WebM, or MOV file';
      }

      const maxBytes = maxSizeMB * 1024 * 1024;
      if (file.size > maxBytes) {
        return `File size must be less than ${maxSizeMB}MB`;
      }

      return new Promise((resolve) => {
        const video = document.createElement('video');
        video.preload = 'metadata';
        video.onloadedmetadata = () => {
          URL.revokeObjectURL(video.src);
          if (video.duration > maxDurationSeconds) {
            resolve(`Video must be less than ${maxDurationSeconds} seconds`);
          } else {
            resolve(null);
          }
        };
        video.onerror = () => {
          resolve('Could not read video file');
        };
        video.src = URL.createObjectURL(file);
      });
    },
    [maxSizeMB, maxDurationSeconds]
  );

  const handleFile = useCallback(
    async (file: File) => {
      setError(null);
      const validationError = await validateFile(file);
      if (validationError) {
        setError(validationError);
        return;
      }

      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreview(url);
    },
    [validateFile]
  );

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragActive(false);

      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFile(e.dataTransfer.files[0]);
      }
    },
    [handleFile]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files && e.target.files[0]) {
        handleFile(e.target.files[0]);
      }
    },
    [handleFile]
  );

  const handleClear = useCallback(() => {
    setSelectedFile(null);
    setPreview(null);
    setError(null);
    if (inputRef.current) {
      inputRef.current.value = '';
    }
  }, []);

  const handleSubmit = useCallback(() => {
    if (selectedFile) {
      onUpload(selectedFile);
    }
  }, [selectedFile, onUpload]);

  return (
    <Card className="w-full">
      <CardContent className="p-6">
        {!preview ? (
          <div
            className={cn(
              'relative flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-12 transition-colors',
              dragActive
                ? 'border-primary bg-primary/5'
                : 'border-muted-foreground/25 hover:border-primary/50',
              error && 'border-destructive'
            )}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <input
              ref={inputRef}
              type="file"
              accept="video/mp4,video/webm,video/quicktime"
              onChange={handleChange}
              className="absolute inset-0 cursor-pointer opacity-0"
            />
            <Upload className="mb-4 h-12 w-12 text-muted-foreground" />
            <p className="mb-2 text-lg font-medium">
              Drag and drop your video here
            </p>
            <p className="text-sm text-muted-foreground">
              or click to browse (MP4, WebM, MOV up to {maxSizeMB}MB, {maxDurationSeconds}s max)
            </p>
            {error && (
              <p className="mt-4 text-sm text-destructive">{error}</p>
            )}
          </div>
        ) : (
          <div className="space-y-4">
            <div className="relative aspect-video overflow-hidden rounded-lg bg-black">
              <video
                ref={videoRef}
                src={preview}
                className="h-full w-full object-contain"
                controls
                muted
              />
              <button
                onClick={handleClear}
                className="absolute right-2 top-2 rounded-full bg-black/50 p-1 text-white hover:bg-black/70"
                disabled={isUploading}
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Film className="h-4 w-4" />
                <span>{selectedFile?.name}</span>
                <span>({(selectedFile?.size ?? 0 / 1024 / 1024).toFixed(1)}MB)</span>
              </div>
              <Button onClick={handleSubmit} disabled={isUploading}>
                {isUploading ? 'Uploading...' : 'Upload Video'}
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
