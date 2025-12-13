import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '@/lib/utils';

interface VideoPlayerProps {
  originalUrl?: string;
  maskUrl?: string;
  compositeUrl?: string;
}

type ViewMode = 'composite' | 'mask' | 'original';

export function VideoPlayer({
  originalUrl,
  maskUrl,
  compositeUrl,
}: VideoPlayerProps) {
  const [viewMode, setViewMode] = useState<ViewMode>('composite');

  const currentUrl =
    viewMode === 'composite'
      ? compositeUrl
      : viewMode === 'mask'
      ? maskUrl
      : originalUrl;

  if (!compositeUrl && !maskUrl) {
    return null;
  }

  return (
    <Card className="w-full">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg">Results</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex gap-2">
          {compositeUrl && (
            <button
              onClick={() => setViewMode('composite')}
              className={cn(
                'rounded-md px-3 py-1.5 text-sm font-medium transition-colors',
                viewMode === 'composite'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-secondary hover:bg-secondary/80'
              )}
            >
              Composite
            </button>
          )}
          {maskUrl && (
            <button
              onClick={() => setViewMode('mask')}
              className={cn(
                'rounded-md px-3 py-1.5 text-sm font-medium transition-colors',
                viewMode === 'mask'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-secondary hover:bg-secondary/80'
              )}
            >
              Mask
            </button>
          )}
          {originalUrl && (
            <button
              onClick={() => setViewMode('original')}
              className={cn(
                'rounded-md px-3 py-1.5 text-sm font-medium transition-colors',
                viewMode === 'original'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-secondary hover:bg-secondary/80'
              )}
            >
              Original
            </button>
          )}
        </div>

        <div className="relative aspect-video overflow-hidden rounded-lg bg-black">
          <video
            key={currentUrl}
            src={currentUrl}
            className="h-full w-full object-contain"
            controls
            playsInline
          />
        </div>
      </CardContent>
    </Card>
  );
}
