import { CheckCircle2, Play } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { useJobs } from '@/hooks/useJob';
import type { Job } from '@/types';

interface JobHistoryProps {
  onSelectJob: (job: Job) => void;
}

export function JobHistory({ onSelectJob }: JobHistoryProps) {
  const { data: jobs, isLoading, error } = useJobs('completed');

  if (isLoading) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="text-lg">Previous Results</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">Loading...</p>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="text-lg">Previous Results</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-destructive">Failed to load job history</p>
        </CardContent>
      </Card>
    );
  }

  if (!jobs || jobs.length === 0) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="text-lg">Previous Results</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">No completed jobs yet</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="text-lg">Previous Results</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {jobs.map((job) => (
          <div
            key={job.id}
            className="flex items-center justify-between rounded-lg border p-3 hover:bg-muted/50 transition-colors"
          >
            <div className="flex items-center gap-3">
              <CheckCircle2 className="h-5 w-5 text-green-500" />
              <div>
                <p className="font-medium text-sm">{job.prompt}</p>
                <p className="text-xs text-muted-foreground">
                  {new Date(job.created_at).toLocaleDateString()} at{' '}
                  {new Date(job.created_at).toLocaleTimeString()}
                </p>
              </div>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => onSelectJob(job)}
            >
              <Play className="h-4 w-4 mr-1" />
              View
            </Button>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
