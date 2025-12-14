import { useCallback, useState, useRef } from 'react';
import { Upload, X, Image as ImageIcon, Plus } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { cn } from '@/lib/utils';

interface ImageUploadProps {
  onUpload: (files: File[]) => void;
  isUploading: boolean;
  maxImages?: number;
  maxSizeMB?: number;
}

const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

export function ImageUpload({
  onUpload,
  isUploading,
  maxImages = 12,
  maxSizeMB = 20,
}: ImageUploadProps) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const validateFiles = useCallback(
    (files: File[]): string | null => {
      if (files.length + selectedFiles.length > maxImages) {
        return `Maximum ${maxImages} images allowed`;
      }

      for (const file of files) {
        if (!ALLOWED_TYPES.includes(file.type)) {
          return `Invalid file type: ${file.name}. Allowed: JPG, PNG, WebP`;
        }

        const sizeMB = file.size / (1024 * 1024);
        if (sizeMB > maxSizeMB) {
          return `File ${file.name} exceeds ${maxSizeMB}MB limit`;
        }
      }

      return null;
    },
    [maxImages, maxSizeMB, selectedFiles.length]
  );

  const handleFiles = useCallback(
    (files: FileList | File[]) => {
      setError(null);
      const fileArray = Array.from(files);
      
      const validationError = validateFiles(fileArray);
      if (validationError) {
        setError(validationError);
        return;
      }

      const newPreviews = fileArray.map((file) => URL.createObjectURL(file));
      
      setSelectedFiles((prev) => [...prev, ...fileArray]);
      setPreviews((prev) => [...prev, ...newPreviews]);
    },
    [validateFiles]
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

      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFiles(e.dataTransfer.files);
      }
    },
    [handleFiles]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFiles(e.target.files);
      }
    },
    [handleFiles]
  );

  const handleRemove = useCallback((index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
    setPreviews((prev) => {
      URL.revokeObjectURL(prev[index]);
      return prev.filter((_, i) => i !== index);
    });
  }, []);

  const handleClearAll = useCallback(() => {
    previews.forEach((url) => URL.revokeObjectURL(url));
    setSelectedFiles([]);
    setPreviews([]);
    setError(null);
    if (inputRef.current) {
      inputRef.current.value = '';
    }
  }, [previews]);

  const handleSubmit = useCallback(() => {
    if (selectedFiles.length > 0) {
      onUpload(selectedFiles);
    }
  }, [selectedFiles, onUpload]);

  const canAddMore = selectedFiles.length < maxImages;

  return (
    <Card className="w-full">
      <CardContent className="p-6">
        {selectedFiles.length === 0 ? (
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
              accept="image/jpeg,image/png,image/webp"
              multiple
              onChange={handleChange}
              className="absolute inset-0 cursor-pointer opacity-0"
              title="Select images to upload"
              aria-label="Select images to upload"
            />
            <Upload className="mb-4 h-12 w-12 text-muted-foreground" />
            <p className="mb-2 text-lg font-medium">
              Drag and drop your images here
            </p>
            <p className="text-sm text-muted-foreground">
              or click to browse (JPG, PNG, WebP up to {maxSizeMB}MB each, max {maxImages} images)
            </p>
            {error && (
              <p className="mt-4 text-sm text-destructive">{error}</p>
            )}
          </div>
        ) : (
          <div className="space-y-4">
            <div className="grid grid-cols-3 gap-3 sm:grid-cols-4">
              {previews.map((preview, index) => (
                <div
                  key={index}
                  className="relative aspect-square overflow-hidden rounded-lg bg-muted"
                >
                  <img
                    src={preview}
                    alt={`Preview ${index + 1}`}
                    className="h-full w-full object-cover"
                  />
                  <button
                    onClick={() => handleRemove(index)}
                    className="absolute right-1 top-1 rounded-full bg-black/50 p-1 text-white hover:bg-black/70"
                    disabled={isUploading}
                    title={`Remove image ${index + 1}`}
                    aria-label={`Remove image ${index + 1}`}
                  >
                    <X className="h-4 w-4" />
                  </button>
                  <div className="absolute bottom-1 left-1 rounded bg-black/50 px-1.5 py-0.5 text-xs text-white">
                    {index + 1}
                  </div>
                </div>
              ))}
              
              {canAddMore && (
                <label
                  className={cn(
                    'relative flex aspect-square cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed transition-colors',
                    'border-muted-foreground/25 hover:border-primary/50 hover:bg-muted/50'
                  )}
                >
                  <input
                    type="file"
                    accept="image/jpeg,image/png,image/webp"
                    multiple
                    onChange={handleChange}
                    className="absolute inset-0 cursor-pointer opacity-0"
                    disabled={isUploading}
                  />
                  <Plus className="h-8 w-8 text-muted-foreground" />
                  <span className="mt-1 text-xs text-muted-foreground">Add more</span>
                </label>
              )}
            </div>

            {error && (
              <p className="text-sm text-destructive">{error}</p>
            )}

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <ImageIcon className="h-4 w-4" />
                <span>{selectedFiles.length} image{selectedFiles.length !== 1 ? 's' : ''} selected</span>
              </div>
              <div className="flex gap-2">
                <Button variant="ghost" onClick={handleClearAll} disabled={isUploading}>
                  Clear all
                </Button>
                <Button onClick={handleSubmit} disabled={isUploading || selectedFiles.length === 0}>
                  {isUploading ? 'Uploading...' : 'Upload Images'}
                </Button>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
