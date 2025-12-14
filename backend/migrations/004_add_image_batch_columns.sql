-- Migration: Add columns for image batch processing
-- This allows the jobs table to support both video and image batch jobs

-- Add job_type to distinguish between video and image batch jobs
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS job_type TEXT DEFAULT 'video';

-- Add image-specific columns
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS image_paths JSONB DEFAULT '[]'::jsonb;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS image_count INTEGER DEFAULT 0;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS composite_images JSONB DEFAULT '[]'::jsonb;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS per_image_results JSONB DEFAULT NULL;

-- Create index for job_type queries
CREATE INDEX IF NOT EXISTS idx_jobs_job_type ON jobs(job_type);

-- Update existing rows to have job_type = 'video'
UPDATE jobs SET job_type = 'video' WHERE job_type IS NULL;
