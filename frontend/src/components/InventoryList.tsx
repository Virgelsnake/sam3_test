import { Package } from 'lucide-react';

interface InventoryListProps {
  inventory: Record<string, number>;
}

export function InventoryList({ inventory }: InventoryListProps) {
  const items = Object.entries(inventory);
  
  if (items.length === 0) {
    return null;
  }

  const totalItems = items.reduce((sum, [, count]) => sum + count, 0);

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
        {items.map(([name, count]) => (
          <div
            key={name}
            className="flex items-center justify-between py-2 px-3 bg-gray-50 rounded-md"
          >
            <span className="text-gray-700 capitalize">{name}</span>
            <span className="bg-blue-100 text-blue-800 text-sm font-medium px-2.5 py-0.5 rounded">
              {count}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
