import { useState } from 'react';
import { Sparkles, ChevronDown } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent } from '@/components/ui/card';

interface PromptInputProps {
  onSubmit: (prompt: string) => void;
  isLoading: boolean;
  disabled?: boolean;
}

const EXAMPLE_PROMPTS = [
  'person',
  'car',
  'dog',
  'cat',
  'ball',
  'hand',
  'face',
  'bird',
];

export function PromptInput({ onSubmit, isLoading, disabled }: PromptInputProps) {
  const [prompt, setPrompt] = useState('');
  const [showExamples, setShowExamples] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (prompt.trim()) {
      onSubmit(prompt.trim());
    }
  };

  const handleExampleClick = (example: string) => {
    setPrompt(example);
    setShowExamples(false);
  };

  const maxLength = 100;
  const remaining = maxLength - prompt.length;

  return (
    <Card className="w-full">
      <CardContent className="p-6">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <label htmlFor="prompt" className="text-sm font-medium">
              What do you want to segment?
            </label>
            <div className="relative">
              <Input
                id="prompt"
                type="text"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value.slice(0, maxLength))}
                placeholder="e.g., person, car, dog..."
                disabled={disabled || isLoading}
                className="pr-20"
              />
              <span
                className={`absolute right-3 top-1/2 -translate-y-1/2 text-xs ${
                  remaining < 20 ? 'text-destructive' : 'text-muted-foreground'
                }`}
              >
                {remaining}
              </span>
            </div>
          </div>

          <div className="relative">
            <button
              type="button"
              onClick={() => setShowExamples(!showExamples)}
              className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
            >
              <span>Example prompts</span>
              <ChevronDown
                className={`h-4 w-4 transition-transform ${
                  showExamples ? 'rotate-180' : ''
                }`}
              />
            </button>
            {showExamples && (
              <div className="mt-2 flex flex-wrap gap-2">
                {EXAMPLE_PROMPTS.map((example) => (
                  <button
                    key={example}
                    type="button"
                    onClick={() => handleExampleClick(example)}
                    className="rounded-full bg-secondary px-3 py-1 text-sm hover:bg-secondary/80"
                  >
                    {example}
                  </button>
                ))}
              </div>
            )}
          </div>

          <Button
            type="submit"
            disabled={!prompt.trim() || disabled || isLoading}
            className="w-full"
          >
            <Sparkles className="mr-2 h-4 w-4" />
            {isLoading ? 'Processing...' : 'Start Segmentation'}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
