-- Migration: Create jobs table
-- Run this in Supabase SQL Editor

CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'queued', 'processing', 'completed', 'failed', 'cancelled')),
    prompt TEXT NOT NULL,
    video_path TEXT NOT NULL,
    mask_video_url TEXT,
    composite_video_url TEXT,
    frame_count INTEGER,
    objects_detected INTEGER,
    error_message TEXT,
    progress INTEGER DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    created_at TIMESTAMPTZ DEFAULT now(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

-- Create index for status queries
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);

-- Create index for created_at queries
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC);

-- Enable Row Level Security (optional, for multi-tenant scenarios)
-- ALTER TABLE jobs ENABLE ROW LEVEL SECURITY;

-- Grant access to authenticated users (adjust as needed)
-- CREATE POLICY "Users can view their own jobs" ON jobs FOR SELECT USING (auth.uid() = user_id);

COMMENT ON TABLE jobs IS 'Video segmentation jobs';
COMMENT ON COLUMN jobs.status IS 'Current job status: pending, queued, processing, completed, failed, cancelled';
COMMENT ON COLUMN jobs.prompt IS 'Text prompt describing the object to segment';
COMMENT ON COLUMN jobs.video_path IS 'Path to the uploaded video in storage';
COMMENT ON COLUMN jobs.mask_video_url IS 'URL to the generated mask video';
COMMENT ON COLUMN jobs.composite_video_url IS 'URL to the composite video with overlay';
COMMENT ON COLUMN jobs.progress IS 'Processing progress percentage (0-100)';
