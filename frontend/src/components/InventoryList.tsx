import { Package, RotateCcw, Check, Save } from 'lucide-react';
import { useState, useEffect, useCallback } from 'react';

interface InventoryListProps {
  jobId: string;
  inventory: Record<string, number>;  // AI-detected counts
  inventoryColors?: Record<string, string>;  // Colors from backend
  userInventory?: Record<string, number>;  // User-corrected counts
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

export function InventoryList({ jobId, inventory, inventoryColors, userInventory, onInventoryChange }: InventoryListProps) {
  // Use user inventory if available, otherwise AI inventory
  const [editableInventory, setEditableInventory] = useState<Record<string, number>>(
    userInventory ?? inventory
  );
  const [hasChanges, setHasChanges] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saved' | 'error'>('idle');
  
  useEffect(() => {
    setEditableInventory(userInventory ?? inventory);
    setHasChanges(false);
  }, [inventory, userInventory]);

  const handleQuantityChange = useCallback((name: string, newValue: string) => {
    const numValue = parseInt(newValue, 10);
    if (!isNaN(numValue) && numValue >= 0) {
      setEditableInventory(prev => {
        const updated = { ...prev, [name]: numValue };
        onInventoryChange?.(updated);
        return updated;
      });
      setHasChanges(true);
      setSaveStatus('idle');
    }
  }, [onInventoryChange]);

  const saveChanges = useCallback(async () => {
    setIsSaving(true);
    setSaveStatus('idle');
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
  }, [jobId, editableInventory]);

  const resetToAI = useCallback(async () => {
    setIsSaving(true);
    try {
      const response = await fetch(`http://localhost:8000/api/jobs/${jobId}/inventory`, {
        method: 'DELETE',
      });
      if (response.ok) {
        setEditableInventory(inventory);
        setHasChanges(false);
        setSaveStatus('idle');
      }
    } catch {
      // Ignore errors on reset
    } finally {
      setIsSaving(false);
    }
  }, [jobId, inventory]);

  // Computed values - after all hooks
  const items = Object.entries(editableInventory);
  const totalItems = items.reduce((sum, [, count]) => sum + count, 0);
  const hasUserCorrections = Object.keys(editableInventory).some(
    key => editableInventory[key] !== inventory[key]
  );

  // Early return after all hooks
  if (items.length === 0) {
    return null;
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 mt-4">
      <div className="flex items-center gap-2 mb-4">
        <Package className="w-5 h-5 text-blue-600" />
        <h3 className="font-semibold text-gray-900">Detected Inventory</h3>
        <span className="ml-auto text-sm text-gray-500">
          {totalItems} item{totalItems !== 1 ? 's' : ''} total
        </span>
      </div>
      
      {/* Table Header */}
      <div className="grid grid-cols-[auto_1fr_80px_80px] gap-2 px-3 py-2 bg-gray-100 rounded-t-md text-xs font-medium text-gray-600 uppercase tracking-wide">
        <div className="w-4"></div>
        <div>Item</div>
        <div className="text-center">AI Count</div>
        <div className="text-center">Your Count</div>
      </div>
      
      {/* Table Body */}
      <div className="divide-y divide-gray-100">
        {items.map(([name, count]: [string, number], index: number) => {
          const color = getCategoryColor(name, index, inventoryColors);
          const aiCount = inventory[name] ?? 0;
          const isModified = count !== aiCount;
          
          return (
            <div
              key={name}
              className={`grid grid-cols-[auto_1fr_80px_80px] gap-2 items-center py-3 px-3 hover:bg-gray-50 transition-colors ${
                isModified ? 'bg-amber-50' : ''
              }`}
            >
              {/* Color indicator */}
              <div 
                className="w-4 h-4 rounded-full border-2 shadow-sm" 
                style={{ backgroundColor: color, borderColor: `${color}80` }}
                title={`Color for ${name}`}
              />
              
              {/* Item name */}
              <span className="text-gray-800 font-medium capitalize">
                {name.replace(/_/g, ' ')}
              </span>
              
              {/* AI count (read-only) */}
              <div className="text-center text-gray-500 text-sm">
                {aiCount}
              </div>
              
              {/* Editable user count */}
              <div className="flex justify-center">
                <input
                  type="number"
                  min="0"
                  value={count}
                  onChange={(e) => handleQuantityChange(name, e.target.value)}
                  aria-label={`Your count for ${name.replace(/_/g, ' ')}`}
                  className={`w-16 text-center text-sm font-semibold px-2 py-1 rounded-md border-2 focus:outline-none focus:ring-2 focus:ring-offset-1 transition-all ${
                    isModified ? 'ring-2 ring-amber-300' : ''
                  }`}
                  style={{ 
                    backgroundColor: `${color}15`,
                    borderColor: isModified ? '#f59e0b' : color,
                    color: '#1f2937',
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
      
      {/* Action buttons */}
      <div className="flex items-center justify-between mt-4 pt-3 border-t border-gray-200">
        <div className="flex items-center gap-2">
          {hasUserCorrections && (
            <button
              onClick={resetToAI}
              disabled={isSaving}
              className="flex items-center gap-1.5 px-3 py-1.5 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-md transition-colors disabled:opacity-50"
            >
              <RotateCcw className="w-4 h-4" />
              Reset to AI
            </button>
          )}
        </div>
        
        <div className="flex items-center gap-2">
          {saveStatus === 'saved' && (
            <span className="flex items-center gap-1 text-sm text-green-600">
              <Check className="w-4 h-4" />
              Saved
            </span>
          )}
          {saveStatus === 'error' && (
            <span className="text-sm text-red-600">Failed to save</span>
          )}
          {hasChanges && (
            <button
              onClick={saveChanges}
              disabled={isSaving}
              className="flex items-center gap-1.5 px-4 py-1.5 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-md transition-colors disabled:opacity-50"
            >
              <Save className="w-4 h-4" />
              {isSaving ? 'Saving...' : 'Save Changes'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
