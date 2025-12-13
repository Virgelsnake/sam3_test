import { Download } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';

interface DownloadButtonsProps {
  maskUrl?: string;
  compositeUrl?: string;
}

export function DownloadButtons({ maskUrl, compositeUrl }: DownloadButtonsProps) {
  if (!maskUrl && !compositeUrl) {
    return null;
  }

  const handleDownload = (url: string, filename: string) => {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.target = '_blank';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <Card className="w-full">
      <CardContent className="p-6">
        <div className="flex flex-col gap-3 sm:flex-row">
          {compositeUrl && (
            <Button
              variant="outline"
              className="flex-1"
              onClick={() => handleDownload(compositeUrl, 'segmentation_composite.mp4')}
            >
              <Download className="mr-2 h-4 w-4" />
              Download Composite
            </Button>
          )}
          {maskUrl && (
            <Button
              variant="outline"
              className="flex-1"
              onClick={() => handleDownload(maskUrl, 'segmentation_mask.mp4')}
            >
              <Download className="mr-2 h-4 w-4" />
              Download Mask
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
