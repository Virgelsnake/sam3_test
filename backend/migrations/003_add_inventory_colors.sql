-- Migration: Add inventory_colors column to jobs table
-- Run this in Supabase SQL Editor

ALTER TABLE jobs ADD COLUMN IF NOT EXISTS inventory_colors JSONB;

COMMENT ON COLUMN jobs.inventory_colors IS 'JSON mapping of category name to hex color string for UI display';
