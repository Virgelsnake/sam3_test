import { Loader2, CheckCircle2, XCircle, Clock, AlertCircle } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import type { JobState } from '@/types';

interface JobStatusProps {
  status: JobState;
  progress: number;
  errorMessage?: string;
}

const statusConfig: Record<
  JobState,
  { icon: React.ElementType; label: string; color: string }
> = {
  pending: {
    icon: Clock,
    label: 'Waiting in queue...',
    color: 'text-yellow-500',
  },
  processing: {
    icon: Loader2,
    label: 'Processing video...',
    color: 'text-blue-500',
  },
  completed: {
    icon: CheckCircle2,
    label: 'Completed!',
    color: 'text-green-500',
  },
  failed: {
    icon: XCircle,
    label: 'Failed',
    color: 'text-destructive',
  },
  cancelled: {
    icon: AlertCircle,
    label: 'Cancelled',
    color: 'text-muted-foreground',
  },
};

export function JobStatus({ status, progress, errorMessage }: JobStatusProps) {
  const config = statusConfig[status];
  const Icon = config.icon;
  const isAnimated = status === 'processing';

  return (
    <Card className="w-full">
      <CardContent className="p-6">
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <Icon
              className={`h-6 w-6 ${config.color} ${
                isAnimated ? 'animate-spin' : ''
              }`}
            />
            <div className="flex-1">
              <p className="font-medium">{config.label}</p>
              {status === 'processing' && (
                <p className="text-sm text-muted-foreground">
                  {progress}% complete
                </p>
              )}
            </div>
          </div>

          {(status === 'pending' || status === 'processing') && (
            <Progress value={progress} className="h-2" />
          )}

          {status === 'failed' && errorMessage && (
            <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
              {errorMessage}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
