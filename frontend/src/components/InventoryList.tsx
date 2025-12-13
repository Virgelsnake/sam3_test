import { Package } from 'lucide-react';
import { useState, useEffect } from 'react';

interface InventoryListProps {
  inventory: Record<string, number>;
  inventoryColors?: Record<string, string>;  // Colors from backend
  onInventoryChange?: (updatedInventory: Record<string, number>) => void;
}

// Fallback colors if backend doesn't provide colors
const FALLBACK_COLORS = [
  '#00FF00', '#0000FF', '#FF0000', '#00FFFF', '#FF00FF',
  '#FFFF00', '#FF0080', '#0080FF', '#FF8000', '#00FF80',
];

function getCategoryColor(category: string, index: number, colorsFromBackend?: Record<string, string>): string {
  // Use colors from backend if available
  if (colorsFromBackend) {
    const normalized = category.toLowerCase();
    if (colorsFromBackend[normalized]) {
      return colorsFromBackend[normalized];
    }
    // Try without underscores
    const withSpaces = normalized.replace(/_/g, ' ');
    if (colorsFromBackend[withSpaces]) {
      return colorsFromBackend[withSpaces];
    }
  }
  // Fallback to index-based colors
  return FALLBACK_COLORS[index % FALLBACK_COLORS.length];
}

export function InventoryList({ inventory, inventoryColors, onInventoryChange }: InventoryListProps) {
  const [editableInventory, setEditableInventory] = useState<Record<string, number>>(inventory);
  
  useEffect(() => {
    setEditableInventory(inventory);
  }, [inventory]);

  const items = Object.entries(editableInventory);
  
  if (items.length === 0) {
    return null;
  }

  const totalItems = items.reduce((sum, [, count]) => sum + count, 0);

  const handleQuantityChange = (name: string, newValue: string) => {
    const numValue = parseInt(newValue, 10);
    if (!isNaN(numValue) && numValue >= 0) {
      const updated = { ...editableInventory, [name]: numValue };
      setEditableInventory(updated);
      onInventoryChange?.(updated);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
      <div className="flex items-center gap-2 mb-3">
        <Package className="w-5 h-5 text-blue-600" />
        <h3 className="font-semibold text-gray-900">Detected Inventory</h3>
        <span className="ml-auto text-sm text-gray-500">
          {totalItems} item{totalItems !== 1 ? 's' : ''} total
        </span>
      </div>
      
      <div className="grid gap-2">
        {items.map(([name, count], index) => {
          const color = getCategoryColor(name, index, inventoryColors);
          return (
            <div
              key={name}
              className="flex items-center justify-between py-2 px-3 bg-gray-50 rounded-md"
            >
              <div className="flex items-center gap-2">
                <div 
                  className="w-3 h-3 rounded-full" 
                  style={{ backgroundColor: color }}
                />
                <span className="text-gray-700 capitalize">
                  {name.replace(/_/g, ' ')}
                </span>
              </div>
              <input
                type="number"
                min="0"
                value={count}
                onChange={(e) => handleQuantityChange(name, e.target.value)}
                aria-label={`Quantity of ${name.replace(/_/g, ' ')}`}
                className="w-14 text-center text-sm font-medium px-2 py-0.5 rounded border-2 focus:outline-none focus:ring-2 focus:ring-offset-1"
                style={{ 
                  backgroundColor: `${color}20`,
                  borderColor: color,
                  color: '#333',
                }}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}
