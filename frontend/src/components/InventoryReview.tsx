import { useState, useMemo, useRef } from 'react';
import { Package, ChevronLeft, ChevronRight, ZoomIn, Download, Check, Save, RotateCcw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import type { PerImageResult } from '@/types';

interface InventoryItem {
  name: string;
  count: number;
  category?: string;
  appears_in_images?: number[];
}

interface InventoryReviewProps {
  jobId: string;
  compositeImages: string[];
  inventory: Record<string, number>;
  inventoryColors?: Record<string, string>;
  userInventory?: Record<string, number>;
  itemsDetail?: InventoryItem[];
  perImageResults?: PerImageResult[];
}

// Must match CATEGORY_COLORS in worker/image_batch_service.py exactly
const CATEGORY_COLORS = [
  "#22c55e",  // Green
  "#3b82f6",  // Blue
  "#ef4444",  // Red
  "#eab308",  // Yellow
  "#a855f7",  // Purple
  "#06b6d4",  // Cyan
  "#f97316",  // Orange
  "#ec4899",  // Pink
  "#14b8a6",  // Teal
  "#8b5cf6",  // Violet
  "#84cc16",  // Lime
  "#f43f5e",  // Rose
];

function getItemColor(itemName: string, sortedItems: string[], colorsFromBackend?: Record<string, string>): string {
  const normalized = itemName.toLowerCase();
  
  // Try backend colors first
  if (colorsFromBackend?.[normalized]) {
    return colorsFromBackend[normalized];
  }
  
  // Fallback: use same index-based assignment as backend (alphabetically sorted)
  const index = sortedItems.indexOf(normalized);
  if (index >= 0) {
    return CATEGORY_COLORS[index % CATEGORY_COLORS.length];
  }
  
  return CATEGORY_COLORS[0];
}

export function InventoryReview({
  jobId,
  compositeImages,
  inventory,
  inventoryColors,
  userInventory,
  itemsDetail,
  perImageResults,
}: InventoryReviewProps) {
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [isZoomed, setIsZoomed] = useState(false);
  const [hoveredItem, setHoveredItem] = useState<string | null>(null);
  const [editableInventory, setEditableInventory] = useState<Record<string, number>>(
    userInventory ?? inventory
  );
  const [hasChanges, setHasChanges] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saved' | 'error'>('idle');
  
  // Ref for image container to calculate SVG overlay dimensions
  const imageContainerRef = useRef<HTMLDivElement>(null);
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0, naturalWidth: 0, naturalHeight: 0 });

  // Sort items alphabetically to match backend color assignment
  const sortedItemNames = useMemo(() => 
    Object.keys(inventory).sort(), 
    [inventory]
  );

  // Build items with their details
  const itemsWithDetails = useMemo(() => {
    return sortedItemNames.map((name) => {
      const detail = itemsDetail?.find(
        (d) => d.name.toLowerCase() === name.toLowerCase()
      );
      return {
        name,
        aiCount: inventory[name] ?? 0,
        userCount: editableInventory[name] ?? inventory[name] ?? 0,
        appearsInImages: detail?.appears_in_images ?? [],
        category: detail?.category,
        color: getItemColor(name, sortedItemNames, inventoryColors),
      };
    });
  }, [sortedItemNames, inventory, editableInventory, itemsDetail, inventoryColors]);

  // Items visible in current image
  const itemsInCurrentImage = useMemo(() => {
    const imageNum = currentImageIndex + 1;
    return itemsWithDetails.filter(
      (item) => item.appearsInImages.includes(imageNum) || item.appearsInImages.length === 0
    );
  }, [itemsWithDetails, currentImageIndex]);

  const totalItems = Object.values(editableInventory).reduce((a, b) => a + b, 0);
  
  // Get bounding boxes for current image
  const currentImageBboxes = useMemo(() => {
    const result = perImageResults?.find(r => r.image_idx === currentImageIndex);
    return result?.item_bboxes ?? [];
  }, [perImageResults, currentImageIndex]);

  // Get bboxes for the hovered item in current image
  const hoveredItemBboxes = useMemo(() => {
    if (!hoveredItem) return [];
    return currentImageBboxes.filter(
      bbox => bbox.category.toLowerCase() === hoveredItem.toLowerCase()
    );
  }, [hoveredItem, currentImageBboxes]);

  // Handle image load to get dimensions for SVG scaling
  const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    setImageDimensions({
      width: img.clientWidth,
      height: img.clientHeight,
      naturalWidth: img.naturalWidth,
      naturalHeight: img.naturalHeight,
    });
  };

  const handleQuantityChange = (name: string, value: string) => {
    const numValue = parseInt(value, 10);
    if (!isNaN(numValue) && numValue >= 0) {
      setEditableInventory((prev) => ({ ...prev, [name]: numValue }));
      setHasChanges(true);
      setSaveStatus('idle');
    }
  };

  const saveChanges = async () => {
    setIsSaving(true);
    try {
      const response = await fetch(`http://localhost:8000/api/jobs/${jobId}/inventory`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_inventory: editableInventory }),
      });
      if (response.ok) {
        setHasChanges(false);
        setSaveStatus('saved');
        setTimeout(() => setSaveStatus('idle'), 2000);
      } else {
        setSaveStatus('error');
      }
    } catch {
      setSaveStatus('error');
    } finally {
      setIsSaving(false);
    }
  };

  const resetToAI = async () => {
    setIsSaving(true);
    try {
      const response = await fetch(`http://localhost:8000/api/jobs/${jobId}/inventory`, {
        method: 'DELETE',
      });
      if (response.ok) {
        setEditableInventory(inventory);
        setHasChanges(false);
      }
    } catch {
      // Ignore
    } finally {
      setIsSaving(false);
    }
  };

  if (!compositeImages?.length) {
    return null;
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-gray-50 border-b">
        <div className="flex items-center gap-2">
          <Package className="w-5 h-5 text-blue-600" />
          <h3 className="font-semibold text-gray-900">Inventory Review</h3>
          <span className="text-sm text-gray-500">
            {sortedItemNames.length} items • {totalItems} total
          </span>
        </div>
        <div className="flex items-center gap-2">
          {saveStatus === 'saved' && (
            <span className="flex items-center gap-1 text-sm text-green-600">
              <Check className="w-4 h-4" /> Saved
            </span>
          )}
          {hasChanges && (
            <Button size="sm" onClick={saveChanges} disabled={isSaving}>
              <Save className="w-4 h-4 mr-1" />
              Save
            </Button>
          )}
        </div>
      </div>

      {/* Main content - side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-0 divide-y lg:divide-y-0 lg:divide-x divide-gray-200">
        {/* Left: Image Viewer */}
        <div className="p-4">
          <div className="relative">
            {/* Main image with SVG overlay for bounding box highlighting */}
            <div
              ref={imageContainerRef}
              className={cn(
                'relative overflow-hidden rounded-lg bg-gray-100',
                isZoomed ? 'cursor-zoom-out' : 'cursor-zoom-in'
              )}
              onClick={() => setIsZoomed(!isZoomed)}
            >
              <img
                src={compositeImages[currentImageIndex]}
                alt={`Image ${currentImageIndex + 1} with detected objects`}
                className={cn(
                  'w-full transition-transform duration-200',
                  isZoomed ? 'scale-150' : 'scale-100'
                )}
                onLoad={handleImageLoad}
              />
              
              {/* SVG overlay for bounding box highlighting */}
              {imageDimensions.naturalWidth > 0 && hoveredItemBboxes.length > 0 && (
                <svg
                  className={cn(
                    'absolute inset-0 w-full h-full pointer-events-none',
                    isZoomed ? 'scale-150' : 'scale-100'
                  )}
                  viewBox={`0 0 ${imageDimensions.naturalWidth} ${imageDimensions.naturalHeight}`}
                  preserveAspectRatio="xMidYMid meet"
                >
                  {hoveredItemBboxes.map((item, idx) => (
                    <g key={idx}>
                      {/* Pulsing highlight rectangle */}
                      <rect
                        x={item.bbox.x}
                        y={item.bbox.y}
                        width={item.bbox.width}
                        height={item.bbox.height}
                        fill={item.color}
                        fillOpacity={0.3}
                        stroke={item.color}
                        strokeWidth={4}
                        className="animate-pulse"
                      />
                      {/* Category label */}
                      <rect
                        x={item.bbox.x}
                        y={item.bbox.y - 28}
                        width={Math.max(item.category.length * 10, 80)}
                        height={24}
                        fill={item.color}
                        rx={4}
                      />
                      <text
                        x={item.bbox.x + 8}
                        y={item.bbox.y - 10}
                        fill="white"
                        fontSize={14}
                        fontWeight="bold"
                      >
                        {item.category}
                      </text>
                    </g>
                  ))}
                </svg>
              )}
              
              {/* Zoom button */}
              <button
                onClick={(e) => { e.stopPropagation(); setIsZoomed(!isZoomed); }}
                className="absolute top-2 right-2 p-2 rounded-full bg-black/50 text-white hover:bg-black/70"
                aria-label="Toggle zoom"
              >
                <ZoomIn className="w-4 h-4" />
              </button>
            </div>

            {/* Navigation */}
            {compositeImages.length > 1 && (
              <>
                <button
                  onClick={() => setCurrentImageIndex((i) => (i === 0 ? compositeImages.length - 1 : i - 1))}
                  className="absolute left-2 top-1/2 -translate-y-1/2 p-2 rounded-full bg-black/50 text-white hover:bg-black/70"
                  aria-label="Previous image"
                >
                  <ChevronLeft className="w-5 h-5" />
                </button>
                <button
                  onClick={() => setCurrentImageIndex((i) => (i === compositeImages.length - 1 ? 0 : i + 1))}
                  className="absolute right-2 top-1/2 -translate-y-1/2 p-2 rounded-full bg-black/50 text-white hover:bg-black/70"
                  aria-label="Next image"
                >
                  <ChevronRight className="w-5 h-5" />
                </button>
                <div className="absolute bottom-2 left-1/2 -translate-x-1/2 px-3 py-1 rounded-full bg-black/50 text-white text-sm">
                  {currentImageIndex + 1} / {compositeImages.length}
                </div>
              </>
            )}
          </div>

          {/* Thumbnails */}
          {compositeImages.length > 1 && (
            <div className="flex gap-2 mt-3 overflow-x-auto pb-1">
              {compositeImages.map((url, idx) => (
                <button
                  key={idx}
                  onClick={() => setCurrentImageIndex(idx)}
                  className={cn(
                    'flex-shrink-0 w-16 h-16 rounded-md overflow-hidden border-2 transition-all',
                    idx === currentImageIndex
                      ? 'border-blue-500 ring-2 ring-blue-200'
                      : 'border-gray-200 opacity-60 hover:opacity-100'
                  )}
                >
                  <img src={url} alt={`Thumb ${idx + 1}`} className="w-full h-full object-cover" />
                </button>
              ))}
            </div>
          )}

          {/* Legend for current image */}
          <div className="mt-4">
            <h4 className="text-sm font-medium text-gray-700 mb-2">
              Items visible in this image:
            </h4>
            <div className="flex flex-wrap gap-2">
              {itemsInCurrentImage.length > 0 ? (
                itemsInCurrentImage.map((item) => (
                  <span
                    key={item.name}
                    className={cn(
                      'inline-flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium transition-all',
                      hoveredItem === item.name ? 'ring-2 ring-offset-1' : ''
                    )}
                    style={{
                      backgroundColor: `${item.color}20`,
                      color: item.color,
                      borderColor: item.color,
                      border: '1px solid',
                    }}
                    onMouseEnter={() => setHoveredItem(item.name)}
                    onMouseLeave={() => setHoveredItem(null)}
                  >
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: item.color }}
                    />
                    {item.name.replace(/_/g, ' ')}
                  </span>
                ))
              ) : (
                <span className="text-sm text-gray-500">All items may appear in this image</span>
              )}
            </div>
          </div>
        </div>

        {/* Right: Inventory List */}
        <div className="p-4 max-h-[600px] overflow-y-auto">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-medium text-gray-700">Item Inventory</h4>
            {Object.keys(editableInventory).some((k) => editableInventory[k] !== inventory[k]) && (
              <button
                onClick={resetToAI}
                disabled={isSaving}
                className="text-xs text-gray-500 hover:text-gray-700 flex items-center gap-1"
              >
                <RotateCcw className="w-3 h-3" />
                Reset to AI
              </button>
            )}
          </div>

          {/* Table header */}
          <div className="grid grid-cols-[auto_1fr_60px_70px] gap-2 px-2 py-2 bg-gray-100 rounded-t-md text-xs font-medium text-gray-600 uppercase">
            <div className="w-4"></div>
            <div>Item</div>
            <div className="text-center">AI</div>
            <div className="text-center">Your</div>
          </div>

          {/* Items */}
          <div className="divide-y divide-gray-100">
            {itemsWithDetails.map((item) => {
              const isModified = item.userCount !== item.aiCount;
              const isHovered = hoveredItem === item.name;
              const isInCurrentImage = itemsInCurrentImage.some((i) => i.name === item.name);

              return (
                <div
                  key={item.name}
                  className={cn(
                    'grid grid-cols-[auto_1fr_60px_70px] gap-2 items-center py-2.5 px-2 transition-all cursor-pointer',
                    isHovered && 'bg-blue-50',
                    isModified && !isHovered && 'bg-amber-50',
                    !isInCurrentImage && 'opacity-50'
                  )}
                  onMouseEnter={() => setHoveredItem(item.name)}
                  onMouseLeave={() => setHoveredItem(null)}
                  onClick={() => {
                    // Find first image this item appears in
                    if (item.appearsInImages.length > 0) {
                      setCurrentImageIndex(item.appearsInImages[0] - 1);
                    }
                  }}
                >
                  {/* Color dot */}
                  <div
                    className={cn(
                      'w-4 h-4 rounded-full border-2 shadow-sm transition-transform',
                      isHovered && 'scale-125'
                    )}
                    style={{ backgroundColor: item.color, borderColor: `${item.color}80` }}
                  />

                  {/* Name + image indicators */}
                  <div className="min-w-0">
                    <span className="text-sm font-medium text-gray-800 capitalize truncate block">
                      {item.name.replace(/_/g, ' ')}
                    </span>
                    {item.appearsInImages.length > 0 && (
                      <span className="text-xs text-gray-400">
                        Images: {item.appearsInImages.join(', ')}
                      </span>
                    )}
                  </div>

                  {/* AI count */}
                  <div className="text-center text-sm text-gray-500">
                    {item.aiCount}
                  </div>

                  {/* User count */}
                  <div className="flex justify-center">
                    <input
                      type="number"
                      min="0"
                      value={item.userCount}
                      onChange={(e) => handleQuantityChange(item.name, e.target.value)}
                      onClick={(e) => e.stopPropagation()}
                      aria-label={`Count for ${item.name}`}
                      className={cn(
                        'w-14 text-center text-sm font-semibold px-1.5 py-1 rounded border-2 focus:outline-none focus:ring-2 focus:ring-offset-1',
                        isModified && 'ring-2 ring-amber-300'
                      )}
                      style={{
                        backgroundColor: `${item.color}15`,
                        borderColor: isModified ? '#f59e0b' : item.color,
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Footer with download */}
      <div className="flex items-center justify-between px-4 py-3 bg-gray-50 border-t">
        <span className="text-xs text-gray-500">
          Hover over items to highlight • Click to jump to image
        </span>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            compositeImages.forEach((url, i) => {
              const a = document.createElement('a');
              a.href = url;
              a.download = `inventory_${i + 1}.jpg`;
              a.click();
            });
          }}
        >
          <Download className="w-4 h-4 mr-1" />
          Download Images
        </Button>
      </div>
    </div>
  );
}
