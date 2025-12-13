import { useState, useCallback } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Video, RefreshCw } from 'lucide-react';
import { VideoUpload } from '@/components/VideoUpload';
import { PromptInput } from '@/components/PromptInput';
import { JobStatus } from '@/components/JobStatus';
import { VideoPlayer } from '@/components/VideoPlayer';
import { DownloadButtons } from '@/components/DownloadButtons';
import { Button } from '@/components/ui/button';
import { useUpload } from '@/hooks/useUpload';
import { useCreateJob, useJob } from '@/hooks/useJob';
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

function AppContent() {
  const [appState, setAppState] = useState<AppState>('upload');
  const [videoId, setVideoId] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [originalVideoUrl, setOriginalVideoUrl] = useState<string | null>(null);

  const uploadMutation = useUpload();
  const createJobMutation = useCreateJob();
  const { data: job } = useJob(jobId, appState === 'processing' || appState === 'results');

  const handleUpload = useCallback(
    async (file: File) => {
      try {
        const result = await uploadMutation.mutateAsync(file);
        setVideoId(result.video_id);
        setOriginalVideoUrl(result.url);
        setAppState('prompt');
      } catch (error) {
        console.error('Upload failed:', error);
      }
    },
    [uploadMutation]
  );

  const handlePromptSubmit = useCallback(
    async (prompt: string) => {
      if (!videoId) return;
      try {
        const job = await createJobMutation.mutateAsync({ videoId, prompt });
        setJobId(job.id);
        setAppState('processing');
      } catch (error) {
        console.error('Job creation failed:', error);
      }
    },
    [videoId, createJobMutation]
  );

  const handleReset = useCallback(() => {
    setAppState('upload');
    setVideoId(null);
    setJobId(null);
    setOriginalVideoUrl(null);
    uploadMutation.reset();
    createJobMutation.reset();
  }, [uploadMutation, createJobMutation]);

  // Auto-transition to results when job completes
  if (job?.status === 'completed' && appState === 'processing') {
    setAppState('results');
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/20">
      <div className="container mx-auto max-w-3xl px-4 py-8">
        <header className="mb-8 text-center">
          <div className="mb-4 flex items-center justify-center gap-2">
            <Video className="h-8 w-8 text-primary" />
            <h1 className="text-3xl font-bold">SAM3 Video Segmentation</h1>
          </div>
          <p className="text-muted-foreground">
            Upload a video and describe what you want to segment
          </p>
        </header>

        <main className="space-y-6">
          {appState === 'upload' && (
            <VideoUpload
              onUpload={handleUpload}
              isUploading={uploadMutation.isPending}
            />
          )}

          {appState === 'prompt' && (
            <>
              <div className="rounded-lg border bg-card p-4">
                <p className="text-sm text-muted-foreground">
                  Video uploaded successfully. Now describe what you want to segment.
                </p>
              </div>
              <PromptInput
                onSubmit={handlePromptSubmit}
                isLoading={createJobMutation.isPending}
              />
              <Button variant="ghost" onClick={handleReset} className="w-full">
                <RefreshCw className="mr-2 h-4 w-4" />
                Upload a different video
              </Button>
            </>
          )}

          {appState === 'processing' && job && (
            <>
              <JobStatus
                status={job.status}
                progress={job.progress}
                errorMessage={job.error_message}
              />
              {job.status === 'failed' && (
                <Button variant="outline" onClick={handleReset} className="w-full">
                  <RefreshCw className="mr-2 h-4 w-4" />
                  Try again
                </Button>
              )}
            </>
          )}

          {appState === 'results' && job && (
            <>
              <VideoPlayer
                originalUrl={originalVideoUrl ?? undefined}
                maskUrl={job.mask_video_url}
                compositeUrl={job.composite_video_url}
              />
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

          {(uploadMutation.isError || createJobMutation.isError) && (
            <div className="rounded-lg border border-destructive bg-destructive/10 p-4 text-sm text-destructive">
              {uploadMutation.error?.message || createJobMutation.error?.message || 'An error occurred'}
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
