-- ============================================================================
-- EMBEDDING AGENT METRICS VIEWS - FINAL WORKING VERSION
-- PostgreSQL 12+ Compatible - No problematic type casts
-- ============================================================================

-- 1. Overall Summary
CREATE OR REPLACE VIEW DEMODB.v_agent_summary AS
SELECT 
    (SELECT COUNT(*) FROM DEMODB.chunked_files) AS total_files,
    (SELECT COUNT(*) FILTER (WHERE status = 'completed') FROM DEMODB.chunked_files) AS completed_files,
    (SELECT COUNT(*) FILTER (WHERE status = 'failed') FROM DEMODB.chunked_files) AS failed_files,
    (SELECT COUNT(*) FILTER (WHERE status = 'processing') FROM DEMODB.chunked_files) AS processing_files,
    (SELECT COALESCE(SUM(num_chunks), 0) FROM DEMODB.chunked_files WHERE status = 'completed') AS total_chunks,
    (SELECT COUNT(*) FROM DEMODB.file_chunks WHERE status = 'completed') AS embedded_chunks,
    (SELECT ROUND(AVG(duration_ms)::numeric, 2) FROM DEMODB.chunked_files WHERE status = 'completed' AND duration_ms IS NOT NULL) AS avg_chunking_time_ms,
    (SELECT ROUND(AVG(duration_ms)::numeric, 2) FROM DEMODB.file_chunks WHERE status = 'completed' AND duration_ms IS NOT NULL) AS avg_embedding_time_ms;

-- 2. Status Distribution
CREATE OR REPLACE VIEW DEMODB.v_agent_status_dist AS
SELECT 
    status,
    COUNT(*) AS count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS percentage
FROM DEMODB.chunked_files
GROUP BY status;

-- 3. File Type Analysis
CREATE OR REPLACE VIEW DEMODB.v_agent_file_types AS
SELECT 
    COALESCE(file_type, 'unknown') AS file_type,
    COUNT(*) AS total_files,
    COUNT(*) FILTER (WHERE status = 'completed') AS completed,
    COUNT(*) FILTER (WHERE status = 'failed') AS failed,
    COALESCE(SUM(num_chunks), 0) AS total_chunks,
    ROUND(100.0 * COUNT(*) FILTER (WHERE status = 'completed') / NULLIF(COUNT(*), 0), 2) AS success_rate_pct
FROM DEMODB.chunked_files
GROUP BY file_type
ORDER BY total_files DESC;

-- 4. Chunking Strategy Performance
CREATE OR REPLACE VIEW DEMODB.v_agent_strategy_perf AS
SELECT 
    chunking_strategy,
    COUNT(*) AS files_processed,
    COUNT(*) FILTER (WHERE status = 'completed') AS completed,
    COUNT(*) FILTER (WHERE status = 'failed') AS failed,
    COALESCE(SUM(num_chunks), 0) AS total_chunks,
    ROUND(100.0 * COUNT(*) FILTER (WHERE status = 'completed') / NULLIF(COUNT(*), 0), 2) AS success_rate_pct,
    ROUND(COALESCE(AVG(duration_ms), 0)::numeric, 2) AS avg_duration_ms
FROM DEMODB.chunked_files
GROUP BY chunking_strategy
ORDER BY files_processed DESC;

-- 5. Hourly Metrics
CREATE OR REPLACE VIEW DEMODB.v_agent_hourly AS
SELECT 
    DATE_TRUNC('hour', created_at) AS time_bucket,
    COUNT(*) AS files_processed,
    COUNT(*) FILTER (WHERE status = 'completed') AS files_completed,
    COUNT(*) FILTER (WHERE status = 'failed') AS files_failed,
    COALESCE(SUM(num_chunks) FILTER (WHERE status = 'completed'), 0) AS chunks_created
FROM DEMODB.chunked_files
GROUP BY DATE_TRUNC('hour', created_at)
ORDER BY time_bucket DESC;

-- 6. Daily Metrics
CREATE OR REPLACE VIEW DEMODB.v_agent_daily AS
SELECT 
    DATE_TRUNC('day', created_at)::DATE AS process_date,
    COUNT(*) AS files_processed,
    COUNT(*) FILTER (WHERE status = 'completed') AS files_completed,
    COUNT(*) FILTER (WHERE status = 'failed') AS files_failed,
    COALESCE(SUM(num_chunks) FILTER (WHERE status = 'completed'), 0) AS chunks_created,
    ROUND(100.0 * COUNT(*) FILTER (WHERE status = 'completed') / NULLIF(COUNT(*), 0), 2) AS success_rate_pct
FROM DEMODB.chunked_files
GROUP BY DATE_TRUNC('day', created_at)
ORDER BY process_date DESC;

-- 7. Recent Files
CREATE OR REPLACE VIEW DEMODB.v_agent_recent_files AS
SELECT 
    id,
    file_name,
    file_type,
    num_chunks,
    status,
    duration_ms,
    error_message,
    created_at
FROM DEMODB.chunked_files
ORDER BY created_at DESC
LIMIT 100;

-- 8. Error Summary (SIMPLE - no date calculations)
CREATE OR REPLACE VIEW DEMODB.v_agent_errors AS
SELECT 
    error_message,
    COUNT(*) AS error_count,
    COUNT(DISTINCT file_name) AS unique_files_affected,
    MAX(created_at) AS last_occurrence
FROM DEMODB.chunked_files
WHERE status = 'failed' AND error_message IS NOT NULL
GROUP BY error_message
ORDER BY error_count DESC;

-- 9. Performance Stats
CREATE OR REPLACE VIEW DEMODB.v_agent_perf_stats AS
SELECT 
    COUNT(*) FILTER (WHERE status = 'completed') AS total_successful,
    COUNT(*) FILTER (WHERE status = 'failed') AS total_failed,
    ROUND(100.0 * COUNT(*) FILTER (WHERE status = 'completed') / NULLIF(COUNT(*), 0), 2) AS overall_success_rate_pct,
    ROUND(MIN(NULLIF(duration_ms, 0))::numeric, 2) AS min_duration_ms,
    ROUND(AVG(NULLIF(duration_ms, 0))::numeric, 2) AS avg_duration_ms,
    ROUND(MAX(duration_ms)::numeric, 2) AS max_duration_ms
FROM DEMODB.chunked_files
WHERE status = 'completed' AND duration_ms IS NOT NULL;

-- 10. Solace Queue Metrics
CREATE OR REPLACE VIEW DEMODB.v_agent_queue_metrics AS
SELECT 
    solace_queue,
    COUNT(*) AS total_messages,
    COUNT(*) FILTER (WHERE status = 'completed') AS completed,
    COUNT(*) FILTER (WHERE status = 'failed') AS failed,
    ROUND(100.0 * COUNT(*) FILTER (WHERE status = 'completed') / NULLIF(COUNT(*), 0), 2) AS success_rate_pct,
    MAX(created_at) AS last_message_time
FROM DEMODB.processed_files
WHERE solace_queue IS NOT NULL
GROUP BY solace_queue;

-- 11. Chunk Status
CREATE OR REPLACE VIEW DEMODB.v_agent_chunk_status AS
SELECT 
    status,
    COUNT(*) AS chunk_count,
    ROUND(AVG(duration_ms)::numeric, 2) AS avg_embedding_time_ms,
    ROUND(MAX(duration_ms)::numeric, 2) AS max_embedding_time_ms,
    ROUND(MIN(duration_ms)::numeric, 2) AS min_embedding_time_ms
FROM DEMODB.file_chunks
GROUP BY status;

-- 12. Bucket Metrics
CREATE OR REPLACE VIEW DEMODB.v_agent_bucket_metrics AS
SELECT 
    bucket_name,
    COUNT(*) AS files_in_bucket,
    COUNT(*) FILTER (WHERE status = 'completed') AS completed,
    COALESCE(SUM(num_chunks), 0) AS total_chunks,
    ROUND(100.0 * COUNT(*) FILTER (WHERE status = 'completed') / NULLIF(COUNT(*), 0), 2) AS success_rate_pct
FROM DEMODB.chunked_files
WHERE bucket_name IS NOT NULL
GROUP BY bucket_name;

-- 13. Qdrant Status
CREATE OR REPLACE VIEW DEMODB.v_agent_qdrant_status AS
SELECT 
    COUNT(*) FILTER (WHERE qdrant_point_id IS NOT NULL) AS chunks_in_qdrant,
    COUNT(*) FILTER (WHERE qdrant_point_id IS NULL) AS chunks_not_in_qdrant,
    COUNT(*) FILTER (WHERE status = 'completed') AS completed_chunks
FROM DEMODB.file_chunks;

-- 14. Transaction Summary
CREATE OR REPLACE VIEW DEMODB.v_agent_transaction_summary AS
SELECT 
    action,
    status,
    COUNT(*) AS transaction_count,
    ROUND(AVG(duration_ms)::numeric, 2) AS avg_duration_ms,
    MAX(created_at) AS last_transaction
FROM DEMODB.processed_files
WHERE action IS NOT NULL
GROUP BY action, status;

-- 15. File Hashing (Duplicates)
CREATE OR REPLACE VIEW DEMODB.v_agent_file_hash_tracking AS
SELECT 
    file_hash,
    COUNT(*) AS duplicate_count,
    STRING_AGG(DISTINCT file_name, ', ') AS files,
    MAX(created_at) AS last_processed
FROM DEMODB.chunked_files
GROUP BY file_hash
HAVING COUNT(*) > 1;

-- 16. Skip Summary - Overall Duplicate and Skip Metrics
CREATE OR REPLACE VIEW DEMODB.v_agent_skip_summary AS
SELECT 
    (SELECT COUNT(*) FROM DEMODB.processed_files WHERE action = 'skip') AS total_skipped,
    (SELECT COUNT(DISTINCT file_hash) FROM DEMODB.processed_files WHERE action = 'skip') AS unique_skipped_files,
    (SELECT COUNT(*) FROM DEMODB.processed_files WHERE action = 'skip' AND file_hash IN (
        SELECT file_hash FROM DEMODB.processed_files WHERE action = 'skip' GROUP BY file_hash HAVING COUNT(*) > 1
    )) AS duplicate_skip_submissions,
    (SELECT COUNT(DISTINCT file_hash) FROM DEMODB.processed_files 
     WHERE file_hash IS NOT NULL AND file_hash IN (
        SELECT file_hash FROM DEMODB.processed_files WHERE action = 'skip' GROUP BY file_hash HAVING COUNT(*) > 1
    )) AS files_submitted_multiple_times,
    (SELECT COUNT(*) FROM DEMODB.processed_files WHERE action != 'skip' AND status = 'completed') AS files_successfully_processed;

-- 17. Skip Details by Action
CREATE OR REPLACE VIEW DEMODB.v_agent_skip_by_action AS
SELECT 
    action,
    COUNT(*) AS submission_count,
    COUNT(DISTINCT file_hash) AS unique_files,
    COUNT(DISTINCT file_name) AS unique_file_names,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM DEMODB.processed_files WHERE action = 'skip'), 2) AS pct_of_total_skips,
    MAX(created_at) AS last_occurrence
FROM DEMODB.processed_files
WHERE action = 'skip'
GROUP BY action
ORDER BY submission_count DESC;

-- 18. Duplicate File Submissions (Files submitted more than once)
CREATE OR REPLACE VIEW DEMODB.v_agent_duplicate_submissions AS
SELECT 
    file_hash,
    file_name,
    COUNT(*) AS submission_count,
    COUNT(*) FILTER (WHERE action = 'skip') AS skip_count,
    COUNT(*) FILTER (WHERE action != 'skip') AS process_count,
    STRING_AGG(DISTINCT action, ', ' ORDER BY action) AS actions_taken,
    MIN(created_at) AS first_submitted,
    MAX(created_at) AS last_submitted
FROM DEMODB.processed_files
WHERE file_hash IS NOT NULL
GROUP BY file_hash, file_name
HAVING COUNT(*) > 1
ORDER BY submission_count DESC;

-- 19. Skip Trends - Hourly
CREATE OR REPLACE VIEW DEMODB.v_agent_skip_hourly AS
SELECT 
    DATE_TRUNC('hour', created_at) AS time_bucket,
    COUNT(*) AS total_submissions,
    COUNT(*) FILTER (WHERE action = 'skip') AS skipped,
    COUNT(*) FILTER (WHERE action != 'skip') AS processed,
    ROUND(100.0 * COUNT(*) FILTER (WHERE action = 'skip') / NULLIF(COUNT(*), 0), 2) AS skip_rate_pct
FROM DEMODB.processed_files
GROUP BY DATE_TRUNC('hour', created_at)
ORDER BY time_bucket DESC;

-- 20. Skip Trends - Daily
CREATE OR REPLACE VIEW DEMODB.v_agent_skip_daily AS
SELECT 
    DATE_TRUNC('day', created_at)::DATE AS process_date,
    COUNT(*) AS total_submissions,
    COUNT(*) FILTER (WHERE action = 'skip') AS skipped,
    COUNT(*) FILTER (WHERE action != 'skip') AS processed,
    COUNT(DISTINCT file_hash) FILTER (WHERE action = 'skip') AS unique_files_skipped,
    ROUND(100.0 * COUNT(*) FILTER (WHERE action = 'skip') / NULLIF(COUNT(*), 0), 2) AS skip_rate_pct
FROM DEMODB.processed_files
GROUP BY DATE_TRUNC('day', created_at)
ORDER BY process_date DESC;

-- 21. Action Summary - All Actions in Processed Files
CREATE OR REPLACE VIEW DEMODB.v_agent_action_summary AS
SELECT 
    action,
    COUNT(*) AS total_count,
    COUNT(*) FILTER (WHERE status = 'completed') AS completed,
    COUNT(*) FILTER (WHERE status = 'failed') AS failed,
    COUNT(*) FILTER (WHERE status = 'processing') AS processing,
    COUNT(*) FILTER (WHERE status = 'queued') AS queued,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM DEMODB.processed_files), 2) AS pct_of_total,
    ROUND(COALESCE(AVG(duration_ms), 0)::numeric, 2) AS avg_duration_ms
FROM DEMODB.processed_files
GROUP BY action
ORDER BY total_count DESC;

-- 22. Duplicate Files Timeline
CREATE OR REPLACE VIEW DEMODB.v_agent_duplicate_timeline AS
SELECT 
    file_hash,
    file_name,
    COUNT(*) AS resubmission_count,
    MIN(created_at) AS first_submission,
    MAX(created_at) AS latest_submission,
    EXTRACT(EPOCH FROM (MAX(created_at) - MIN(created_at))) / 3600 AS hours_span
FROM DEMODB.processed_files
WHERE file_hash IS NOT NULL
GROUP BY file_hash, file_name
HAVING COUNT(*) > 1
ORDER BY resubmission_count DESC;

-- ============================================================================
-- INDIVIDUAL GRANT STATEMENTS (Simple and Compatible)
-- ============================================================================
GRANT USAGE ON SCHEMA DEMODB TO postgres;
GRANT SELECT ON DEMODB.v_agent_summary TO postgres;
GRANT SELECT ON DEMODB.v_agent_status_dist TO postgres;
GRANT SELECT ON DEMODB.v_agent_file_types TO postgres;
GRANT SELECT ON DEMODB.v_agent_strategy_perf TO postgres;
GRANT SELECT ON DEMODB.v_agent_hourly TO postgres;
GRANT SELECT ON DEMODB.v_agent_daily TO postgres;
GRANT SELECT ON DEMODB.v_agent_recent_files TO postgres;
GRANT SELECT ON DEMODB.v_agent_errors TO postgres;
GRANT SELECT ON DEMODB.v_agent_perf_stats TO postgres;
GRANT SELECT ON DEMODB.v_agent_queue_metrics TO postgres;
GRANT SELECT ON DEMODB.v_agent_chunk_status TO postgres;
GRANT SELECT ON DEMODB.v_agent_bucket_metrics TO postgres;
GRANT SELECT ON DEMODB.v_agent_qdrant_status TO postgres;
GRANT SELECT ON DEMODB.v_agent_transaction_summary TO postgres;
GRANT SELECT ON DEMODB.v_agent_file_hash_tracking TO postgres;
GRANT SELECT ON DEMODB.v_agent_skip_summary TO postgres;
GRANT SELECT ON DEMODB.v_agent_skip_by_action TO postgres;
GRANT SELECT ON DEMODB.v_agent_duplicate_submissions TO postgres;
GRANT SELECT ON DEMODB.v_agent_skip_hourly TO postgres;
GRANT SELECT ON DEMODB.v_agent_skip_daily TO postgres;
GRANT SELECT ON DEMODB.v_agent_action_summary TO postgres;
GRANT SELECT ON DEMODB.v_agent_duplicate_timeline TO postgres;

-- ============================================================================
-- INDEXES for Performance
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_chunked_files_status_created 
    ON DEMODB.chunked_files(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chunked_files_created 
    ON DEMODB.chunked_files(created_at DESC);

-- ============================================================================
-- SUCCESS - All views created
-- ============================================================================