-- Migration: Add inventory column to jobs table
-- Run this in Supabase SQL Editor

ALTER TABLE jobs ADD COLUMN IF NOT EXISTS inventory JSONB;

COMMENT ON COLUMN jobs.inventory IS 'JSON inventory of detected items from classification (item_name: count)';
