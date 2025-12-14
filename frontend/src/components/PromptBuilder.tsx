import { useState, useMemo } from 'react';
import { Home, Truck, Building2, DollarSign, ClipboardList, ChevronDown, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface PromptBuilderProps {
  onPromptGenerated: (prompt: string) => void;
}

interface Preset {
  id: string;
  name: string;
  icon: React.ReactNode;
  description: string;
  prompt: string;
}

const PRESETS: Preset[] = [
  {
    id: 'insurance',
    name: 'Insurance',
    icon: <Home className="w-4 h-4" />,
    description: 'Comprehensive inventory for insurance claims',
    prompt: 'Insurance valuation inventory. Include all furniture, electronics, appliances, and valuable items. Provide detailed descriptions with condition assessment and estimated replacement value where possible.',
  },
  {
    id: 'moving',
    name: 'Moving',
    icon: <Truck className="w-4 h-4" />,
    description: 'Inventory for moving or relocation',
    prompt: 'Moving inventory. List all items that would need to be packed and transported. Include furniture dimensions where visible, and note any fragile or special handling items.',
  },
  {
    id: 'office',
    name: 'Office Audit',
    icon: <Building2 className="w-4 h-4" />,
    description: 'Office equipment and asset tracking',
    prompt: 'Office asset audit. Focus on furniture, IT equipment, monitors, computers, and office supplies. Include serial numbers or model information if visible.',
  },
  {
    id: 'selling',
    name: 'Selling',
    icon: <DollarSign className="w-4 h-4" />,
    description: 'Items for sale or valuation',
    prompt: 'Items for sale inventory. Focus on items with resale value. Include detailed descriptions, visible condition, brand names, and estimated market value.',
  },
  {
    id: 'general',
    name: 'General',
    icon: <ClipboardList className="w-4 h-4" />,
    description: 'Complete room inventory',
    prompt: 'Complete room inventory. List all visible items including furniture, electronics, décor, and miscellaneous items.',
  },
];

type DetailLevel = 'basic' | 'standard' | 'detailed';
type ItemScope = 'all' | 'major' | 'custom';

const CATEGORIES = [
  { id: 'furniture', label: 'Furniture', default: true },
  { id: 'electronics', label: 'Electronics', default: true },
  { id: 'appliances', label: 'Appliances', default: true },
  { id: 'decor', label: 'Décor & Art', default: true },
  { id: 'storage', label: 'Storage & Bags', default: false },
  { id: 'supplies', label: 'Office Supplies', default: false },
  { id: 'cables', label: 'Cables & Small Items', default: false },
  { id: 'fixtures', label: 'Fixtures (radiators, blinds)', default: false },
];

export function PromptBuilder({ onPromptGenerated }: PromptBuilderProps) {
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);
  const [showCustomize, setShowCustomize] = useState(false);
  const [detailLevel, setDetailLevel] = useState<DetailLevel>('standard');
  const [itemScope, setItemScope] = useState<ItemScope>('all');
  const [selectedCategories, setSelectedCategories] = useState<string[]>(
    CATEGORIES.filter(c => c.default).map(c => c.id)
  );
  const [customNote, setCustomNote] = useState('');

  // Compute generated prompt using useMemo (no useState needed)
  const generatedPrompt = useMemo((): string => {
    const preset = PRESETS.find(p => p.id === selectedPreset);
    
    if (!showCustomize && preset) {
      return preset.prompt;
    }

    const parts: string[] = [];

    // Base from preset or default
    if (preset) {
      parts.push(preset.prompt);
    } else {
      parts.push('Room inventory.');
    }

    // Item scope
    if (itemScope === 'major') {
      parts.push('Focus on major items only - skip small items like cables, pens, and minor accessories.');
    } else if (itemScope === 'custom' && selectedCategories.length > 0) {
      const categoryLabels = CATEGORIES
        .filter(c => selectedCategories.includes(c.id))
        .map(c => c.label.toLowerCase());
      parts.push(`Include only: ${categoryLabels.join(', ')}.`);
    }

    // Detail level
    if (detailLevel === 'basic') {
      parts.push('Provide simple item names and counts only.');
    } else if (detailLevel === 'detailed') {
      parts.push('Include detailed descriptions with colors, materials, brands, condition, and estimated values where visible.');
    }

    // Custom note
    if (customNote.trim()) {
      parts.push(customNote.trim());
    }

    return parts.join(' ');
  }, [selectedPreset, showCustomize, itemScope, selectedCategories, detailLevel, customNote]);

  const handlePresetClick = (presetId: string) => {
    setSelectedPreset(presetId);
    if (!showCustomize) {
      const preset = PRESETS.find(p => p.id === presetId);
      if (preset) {
        onPromptGenerated(preset.prompt);
      }
    }
  };

  const handleCategoryToggle = (categoryId: string) => {
    setSelectedCategories(prev => 
      prev.includes(categoryId)
        ? prev.filter(id => id !== categoryId)
        : [...prev, categoryId]
    );
  };

  const handleApplyCustom = () => {
    onPromptGenerated(generatedPrompt);
  };

  return (
    <div className="w-full max-w-2xl mx-auto space-y-4">
      {/* Presets */}
      <div className="space-y-2">
        <label className="text-sm font-medium text-gray-700">Quick Start - Select Purpose:</label>
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
          {PRESETS.map((preset) => (
            <button
              key={preset.id}
              onClick={() => handlePresetClick(preset.id)}
              className={`
                flex flex-col items-center gap-1 p-3 rounded-lg border-2 transition-all
                ${selectedPreset === preset.id 
                  ? 'border-blue-500 bg-blue-50 text-blue-700' 
                  : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                }
              `}
              title={preset.description}
            >
              {preset.icon}
              <span className="text-xs font-medium">{preset.name}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Customize Toggle */}
      <button
        onClick={() => setShowCustomize(!showCustomize)}
        className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900"
      >
        <ChevronDown className={`w-4 h-4 transition-transform ${showCustomize ? 'rotate-180' : ''}`} />
        <span>{showCustomize ? 'Hide' : 'Show'} customization options</span>
      </button>

      {/* Customization Panel */}
      {showCustomize && (
        <div className="space-y-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
          {/* Detail Level */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700">Detail Level:</label>
            <div className="flex gap-2">
              {[
                { id: 'basic', label: 'Basic', desc: 'Names only' },
                { id: 'standard', label: 'Standard', desc: 'Names + counts' },
                { id: 'detailed', label: 'Detailed', desc: 'Full descriptions' },
              ].map((level) => (
                <button
                  key={level.id}
                  onClick={() => setDetailLevel(level.id as DetailLevel)}
                  className={`
                    flex-1 py-2 px-3 rounded-md text-sm transition-all
                    ${detailLevel === level.id
                      ? 'bg-blue-500 text-white'
                      : 'bg-white border border-gray-300 hover:bg-gray-50'
                    }
                  `}
                >
                  <div className="font-medium">{level.label}</div>
                  <div className={`text-xs ${detailLevel === level.id ? 'text-blue-100' : 'text-gray-500'}`}>
                    {level.desc}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Item Scope */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700">What to Include:</label>
            <div className="flex gap-2">
              {[
                { id: 'all', label: 'Everything' },
                { id: 'major', label: 'Major Items Only' },
                { id: 'custom', label: 'Custom Selection' },
              ].map((scope) => (
                <button
                  key={scope.id}
                  onClick={() => setItemScope(scope.id as ItemScope)}
                  className={`
                    flex-1 py-2 px-3 rounded-md text-sm font-medium transition-all
                    ${itemScope === scope.id
                      ? 'bg-blue-500 text-white'
                      : 'bg-white border border-gray-300 hover:bg-gray-50'
                    }
                  `}
                >
                  {scope.label}
                </button>
              ))}
            </div>
          </div>

          {/* Category Selection (when custom) */}
          {itemScope === 'custom' && (
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-700">Select Categories:</label>
              <div className="flex flex-wrap gap-2">
                {CATEGORIES.map((category) => (
                  <button
                    key={category.id}
                    onClick={() => handleCategoryToggle(category.id)}
                    className={`
                      py-1 px-3 rounded-full text-sm transition-all
                      ${selectedCategories.includes(category.id)
                        ? 'bg-blue-500 text-white'
                        : 'bg-white border border-gray-300 hover:bg-gray-50'
                      }
                    `}
                  >
                    {category.label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Custom Note */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700">Additional Instructions (optional):</label>
            <input
              type="text"
              value={customNote}
              onChange={(e) => setCustomNote(e.target.value)}
              placeholder="e.g., Focus on items near the window, Include brand names..."
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>

          {/* Preview */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700">Generated Prompt Preview:</label>
            <div className="p-3 bg-white border border-gray-200 rounded-md text-sm text-gray-700 min-h-[60px]">
              {generatedPrompt || 'Select options to generate a prompt...'}
            </div>
          </div>

          {/* Apply Button */}
          <Button 
            onClick={handleApplyCustom}
            className="w-full"
            disabled={!generatedPrompt}
          >
            <Sparkles className="w-4 h-4 mr-2" />
            Use This Prompt
          </Button>
        </div>
      )}

      {/* Selected prompt indicator (when not customizing) */}
      {!showCustomize && selectedPreset && (
        <div className="p-3 bg-blue-50 border border-blue-200 rounded-md">
          <div className="text-xs text-blue-600 font-medium mb-1">Selected Prompt:</div>
          <div className="text-sm text-blue-800">
            {PRESETS.find(p => p.id === selectedPreset)?.prompt}
          </div>
        </div>
      )}
    </div>
  );
}
