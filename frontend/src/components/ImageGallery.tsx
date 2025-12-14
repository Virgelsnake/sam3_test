import { useState } from 'react';
import { ChevronLeft, ChevronRight, Download, ZoomIn } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '@/lib/utils';

interface ImageGalleryProps {
  images: string[];
  title?: string;
}

export function ImageGallery({ images, title = 'Processed Images' }: ImageGalleryProps) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isZoomed, setIsZoomed] = useState(false);

  if (!images || images.length === 0) {
    return null;
  }

  const handlePrevious = () => {
    setCurrentIndex((prev) => (prev === 0 ? images.length - 1 : prev - 1));
  };

  const handleNext = () => {
    setCurrentIndex((prev) => (prev === images.length - 1 ? 0 : prev + 1));
  };

  const handleDownload = async (url: string, index: number) => {
    try {
      const response = await fetch(url);
      const blob = await response.blob();
      const downloadUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = downloadUrl;
      a.download = `inventory_image_${index + 1}.jpg`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(downloadUrl);
    } catch (error) {
      console.error('Download failed:', error);
    }
  };

  const handleDownloadAll = async () => {
    for (let i = 0; i < images.length; i++) {
      await handleDownload(images[i], i);
    }
  };

  return (
    <Card className="w-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">{title}</CardTitle>
          <Button variant="outline" size="sm" onClick={handleDownloadAll}>
            <Download className="mr-2 h-4 w-4" />
            Download All
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Main image viewer */}
        <div className="relative">
          <div
            className={cn(
              'relative overflow-hidden rounded-lg bg-black',
              isZoomed ? 'cursor-zoom-out' : 'cursor-zoom-in'
            )}
            onClick={() => setIsZoomed(!isZoomed)}
          >
            <img
              src={images[currentIndex]}
              alt={`Processed image ${currentIndex + 1}`}
              className={cn(
                'w-full transition-transform duration-200',
                isZoomed ? 'scale-150' : 'scale-100'
              )}
            />
          </div>

          {/* Navigation arrows */}
          {images.length > 1 && (
            <>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handlePrevious();
                }}
                className="absolute left-2 top-1/2 -translate-y-1/2 rounded-full bg-black/50 p-2 text-white hover:bg-black/70"
                title="Previous image"
                aria-label="Previous image"
              >
                <ChevronLeft className="h-6 w-6" />
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleNext();
                }}
                className="absolute right-2 top-1/2 -translate-y-1/2 rounded-full bg-black/50 p-2 text-white hover:bg-black/70"
                title="Next image"
                aria-label="Next image"
              >
                <ChevronRight className="h-6 w-6" />
              </button>
            </>
          )}

          {/* Image counter */}
          <div className="absolute bottom-2 left-1/2 -translate-x-1/2 rounded-full bg-black/50 px-3 py-1 text-sm text-white">
            {currentIndex + 1} / {images.length}
          </div>

          {/* Zoom indicator */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              setIsZoomed(!isZoomed);
            }}
            className="absolute right-2 top-2 rounded-full bg-black/50 p-2 text-white hover:bg-black/70"
            title="Toggle zoom"
            aria-label="Toggle zoom"
          >
            <ZoomIn className="h-4 w-4" />
          </button>
        </div>

        {/* Thumbnail strip */}
        {images.length > 1 && (
          <div className="flex gap-2 overflow-x-auto pb-2">
            {images.map((url, index) => (
              <button
                key={index}
                onClick={() => setCurrentIndex(index)}
                className={cn(
                  'relative h-16 w-16 flex-shrink-0 overflow-hidden rounded-md transition-all',
                  index === currentIndex
                    ? 'ring-2 ring-primary ring-offset-2'
                    : 'opacity-60 hover:opacity-100'
                )}
                title={`View image ${index + 1}`}
                aria-label={`View image ${index + 1}`}
              >
                <img
                  src={url}
                  alt={`Thumbnail ${index + 1}`}
                  className="h-full w-full object-cover"
                />
              </button>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
