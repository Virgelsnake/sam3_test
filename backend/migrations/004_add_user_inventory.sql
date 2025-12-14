-- Migration: Add user_inventory column to jobs table for user-corrected quantities
-- Run this in Supabase SQL Editor

ALTER TABLE jobs ADD COLUMN IF NOT EXISTS user_inventory JSONB;

COMMENT ON COLUMN jobs.user_inventory IS 'User-corrected inventory counts. When NULL, AI inventory is used. Allows users to override detected quantities.';
