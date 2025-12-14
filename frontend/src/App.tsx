import { useState, useCallback } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Video, RefreshCw, Image as ImageIcon } from 'lucide-react';
import { VideoUpload } from '@/components/VideoUpload';
import { ImageUpload } from '@/components/ImageUpload';
import { PromptInput } from '@/components/PromptInput';
import { JobStatus } from '@/components/JobStatus';
import { VideoPlayer } from '@/components/VideoPlayer';
import { ImageGallery } from '@/components/ImageGallery';
import { DownloadButtons } from '@/components/DownloadButtons';
import { JobHistory } from '@/components/JobHistory';
import { InventoryList } from '@/components/InventoryList';
import { Button } from '@/components/ui/button';
import { useUpload } from '@/hooks/useUpload';
import { useCreateJob, useJob } from '@/hooks/useJob';
import { useImageUpload, useCreateImageJob, useImageJob } from '@/hooks/useImageJob';
import type { Job } from '@/types';
import './index.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 5000,
    },
  },
});

type AppState = 'upload' | 'prompt' | 'processing' | 'results';
type InputMode = 'video' | 'images';

function AppContent() {
  const [inputMode, setInputMode] = useState<InputMode>('video');
  const [appState, setAppState] = useState<AppState>('upload');
  const [videoId, setVideoId] = useState<string | null>(null);
  const [imageIds, setImageIds] = useState<string[]>([]);
  const [jobId, setJobId] = useState<string | null>(null);
  const [originalVideoUrl, setOriginalVideoUrl] = useState<string | null>(null);

  // Video hooks
  const uploadMutation = useUpload();
  const createJobMutation = useCreateJob();
  const { data: job } = useJob(jobId, inputMode === 'video' && (appState === 'processing' || appState === 'results'));

  // Image hooks
  const imageUploadMutation = useImageUpload();
  const createImageJobMutation = useCreateImageJob();
  const { data: imageJob } = useImageJob(jobId, inputMode === 'images' && (appState === 'processing' || appState === 'results'));

  // Video handlers
  const handleVideoUpload = useCallback(
    async (file: File) => {
      try {
        const result = await uploadMutation.mutateAsync(file);
        setVideoId(result.path);
        setOriginalVideoUrl(result.url);
        setAppState('prompt');
      } catch (error) {
        console.error('Upload failed:', error);
      }
    },
    [uploadMutation]
  );

  // Image handlers
  const handleImageUpload = useCallback(
    async (files: File[]) => {
      try {
        const results = await imageUploadMutation.mutateAsync(files);
        const paths = results.map((r) => r.path);
        setImageIds(paths);
        setAppState('prompt');
      } catch (error) {
        console.error('Image upload failed:', error);
      }
    },
    [imageUploadMutation]
  );

  const handlePromptSubmit = useCallback(
    async (prompt: string) => {
      try {
        if (inputMode === 'video') {
          if (!videoId) return;
          const job = await createJobMutation.mutateAsync({ videoId, prompt });
          setJobId(job.id);
        } else {
          if (imageIds.length === 0) return;
          const job = await createImageJobMutation.mutateAsync({ imageIds, prompt });
          setJobId(job.id);
        }
        setAppState('processing');
      } catch (error) {
        console.error('Job creation failed:', error);
      }
    },
    [inputMode, videoId, imageIds, createJobMutation, createImageJobMutation]
  );

  const handleReset = useCallback(() => {
    setAppState('upload');
    setVideoId(null);
    setImageIds([]);
    setJobId(null);
    setOriginalVideoUrl(null);
    uploadMutation.reset();
    createJobMutation.reset();
    imageUploadMutation.reset();
    createImageJobMutation.reset();
  }, [uploadMutation, createJobMutation, imageUploadMutation, createImageJobMutation]);

  const handleSelectJob = useCallback((selectedJob: Job) => {
    setJobId(selectedJob.id);
    setInputMode('video'); // Assume video for job history
    setAppState('results');
  }, []);

  const handleModeChange = useCallback((mode: InputMode) => {
    setInputMode(mode);
    handleReset();
  }, [handleReset]);

  // Auto-transition to results when job completes
  const currentJob = inputMode === 'video' ? job : imageJob;
  if (currentJob?.status === 'completed' && appState === 'processing') {
    setAppState('results');
  }

  const isLoading = inputMode === 'video' 
    ? createJobMutation.isPending 
    : createImageJobMutation.isPending;

  const hasError = uploadMutation.isError || createJobMutation.isError || 
    imageUploadMutation.isError || createImageJobMutation.isError;

  const errorMessage = uploadMutation.error?.message || 
    createJobMutation.error?.message || 
    imageUploadMutation.error?.message || 
    createImageJobMutation.error?.message || 
    'An error occurred';

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/20">
      <div className="container mx-auto max-w-3xl px-4 py-8">
        <header className="mb-8 text-center">
          <div className="mb-4 flex items-center justify-center gap-2">
            {inputMode === 'video' ? (
              <Video className="h-8 w-8 text-primary" />
            ) : (
              <ImageIcon className="h-8 w-8 text-primary" />
            )}
            <h1 className="text-3xl font-bold">SAM3 Inventory System</h1>
          </div>
          <p className="text-muted-foreground">
            {inputMode === 'video' 
              ? 'Upload a video to generate an inventory'
              : 'Upload multiple images to generate an inventory'}
          </p>
        </header>

        <main className="space-y-6">
          {appState === 'upload' && (
            <>
              {/* Mode Toggle */}
              <div className="flex justify-center gap-2">
                <Button
                  variant={inputMode === 'video' ? 'default' : 'outline'}
                  onClick={() => handleModeChange('video')}
                  className="flex items-center gap-2"
                >
                  <Video className="h-4 w-4" />
                  Video
                </Button>
                <Button
                  variant={inputMode === 'images' ? 'default' : 'outline'}
                  onClick={() => handleModeChange('images')}
                  className="flex items-center gap-2"
                >
                  <ImageIcon className="h-4 w-4" />
                  Images
                </Button>
              </div>

              {inputMode === 'video' ? (
                <VideoUpload
                  onUpload={handleVideoUpload}
                  isUploading={uploadMutation.isPending}
                />
              ) : (
                <ImageUpload
                  onUpload={handleImageUpload}
                  isUploading={imageUploadMutation.isPending}
                />
              )}
              <JobHistory onSelectJob={handleSelectJob} />
            </>
          )}

          {appState === 'prompt' && (
            <>
              <div className="rounded-lg border bg-card p-4">
                <p className="text-sm text-muted-foreground">
                  {inputMode === 'video' 
                    ? 'Video uploaded successfully. Now describe what you want to inventory.'
                    : `${imageIds.length} images uploaded. Now describe what you want to inventory.`}
                </p>
              </div>
              <PromptInput
                onSubmit={handlePromptSubmit}
                isLoading={isLoading}
              />
              <Button variant="ghost" onClick={handleReset} className="w-full">
                <RefreshCw className="mr-2 h-4 w-4" />
                {inputMode === 'video' ? 'Upload a different video' : 'Upload different images'}
              </Button>
            </>
          )}

          {appState === 'processing' && currentJob && (
            <>
              <JobStatus
                status={currentJob.status}
                progress={currentJob.progress}
                errorMessage={currentJob.error_message}
              />
              {currentJob.status === 'failed' && (
                <Button variant="outline" onClick={handleReset} className="w-full">
                  <RefreshCw className="mr-2 h-4 w-4" />
                  Try again
                </Button>
              )}
            </>
          )}

          {appState === 'results' && inputMode === 'video' && job && (
            <>
              <VideoPlayer
                originalUrl={originalVideoUrl ?? undefined}
                maskUrl={job.mask_video_url}
                compositeUrl={job.composite_video_url}
              />
              
              {job.inventory && Object.keys(job.inventory).length > 0 && (
                <InventoryList 
                  jobId={job.id}
                  inventory={job.inventory} 
                  inventoryColors={job.inventory_colors}
                  userInventory={job.user_inventory}
                />
              )}
              
              <DownloadButtons
                maskUrl={job.mask_video_url}
                compositeUrl={job.composite_video_url}
              />
              <div className="rounded-lg border bg-card p-4">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">
                    Processed {job.frame_count} frames, found {job.objects_detected} object(s)
                  </span>
                  <span className="font-medium text-green-600">Completed</span>
                </div>
              </div>
              <Button variant="outline" onClick={handleReset} className="w-full">
                <RefreshCw className="mr-2 h-4 w-4" />
                Process another video
              </Button>
            </>
          )}

          {appState === 'results' && inputMode === 'images' && imageJob && (
            <>
              {imageJob.composite_images && imageJob.composite_images.length > 0 && (
                <ImageGallery 
                  images={imageJob.composite_images}
                  title="Processed Images with Detected Objects"
                />
              )}
              
              {imageJob.inventory && Object.keys(imageJob.inventory).length > 0 && (
                <InventoryList 
                  jobId={imageJob.id}
                  inventory={imageJob.inventory} 
                  inventoryColors={imageJob.inventory_colors}
                  userInventory={imageJob.user_inventory}
                />
              )}
              
              <div className="rounded-lg border bg-card p-4">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">
                    Processed {imageJob.image_count} images, found {imageJob.objects_detected} unique object(s)
                  </span>
                  <span className="font-medium text-green-600">Completed</span>
                </div>
              </div>
              <Button variant="outline" onClick={handleReset} className="w-full">
                <RefreshCw className="mr-2 h-4 w-4" />
                Process more images
              </Button>
            </>
          )}

          {hasError && (
            <div className="rounded-lg border border-destructive bg-destructive/10 p-4 text-sm text-destructive">
              {errorMessage}
            </div>
          )}
        </main>

        <footer className="mt-12 text-center text-sm text-muted-foreground">
          <p>Powered by SAM 3 (Segment Anything Model 3) from Meta AI</p>
        </footer>
      </div>
    </div>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppContent />
    </QueryClientProvider>
  );
}

export default App;
