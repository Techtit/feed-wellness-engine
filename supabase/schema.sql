-- Feed Wellness Engine — Supabase Schema
-- Run this in Supabase SQL Editor

-- Session-level wellness reports
CREATE TABLE IF NOT EXISTS wellness_sessions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL UNIQUE,
    session_start TIMESTAMPTZ NOT NULL,
    session_end TIMESTAMPTZ NOT NULL,
    post_count INTEGER NOT NULL DEFAULT 0,
    toxicity_score FLOAT NOT NULL DEFAULT 0,
    addictiveness_score FLOAT NOT NULL DEFAULT 0,
    positivity_score FLOAT NOT NULL DEFAULT 0,
    emotional_intensity FLOAT NOT NULL DEFAULT 0,
    circadian_score FLOAT NOT NULL DEFAULT 0,
    wellness_index FLOAT NOT NULL DEFAULT 50,
    recommendation TEXT NOT NULL DEFAULT 'NO_ACTION',
    feed_embedding_json JSONB,  -- stored as JSON array for future analysis
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Daily rollups for historical trends
CREATE TABLE IF NOT EXISTS daily_aggregates (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id TEXT NOT NULL,
    date DATE NOT NULL,
    avg_wellness_index FLOAT NOT NULL DEFAULT 50,
    avg_toxicity FLOAT NOT NULL DEFAULT 0,
    avg_positivity FLOAT NOT NULL DEFAULT 0,
    avg_addictiveness FLOAT NOT NULL DEFAULT 0,
    total_dwell_seconds FLOAT NOT NULL DEFAULT 0,
    night_usage_ratio FLOAT NOT NULL DEFAULT 0,
    session_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, date)
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_wellness_sessions_user ON wellness_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_wellness_sessions_created ON wellness_sessions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_daily_aggregates_user_date ON daily_aggregates(user_id, date DESC);

-- RLS Policies (user already has RLS enabled)
ALTER TABLE wellness_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE daily_aggregates ENABLE ROW LEVEL SECURITY;

-- Users can only read/write their own data
CREATE POLICY "Users can insert own sessions"
    ON wellness_sessions FOR INSERT
    WITH CHECK (true);  -- Backend inserts with service key; adjust if using user auth

CREATE POLICY "Users can read own sessions"
    ON wellness_sessions FOR SELECT
    USING (true);  -- Adjust to (auth.uid()::text = user_id) if using Supabase Auth

CREATE POLICY "Users can insert own aggregates"
    ON daily_aggregates FOR INSERT
    WITH CHECK (true);

CREATE POLICY "Users can update own aggregates"
    ON daily_aggregates FOR UPDATE
    USING (true);

CREATE POLICY "Users can read own aggregates"
    ON daily_aggregates FOR SELECT
    USING (true);
