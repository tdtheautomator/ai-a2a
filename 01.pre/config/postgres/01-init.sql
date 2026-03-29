-- PostgreSQL Schema for Chunking & Embedding Agent
-- Creates tables for tracking chunked documents, chunk embeddings, and transaction logs

-- Create DEMODB schema if not exists
CREATE SCHEMA IF NOT EXISTS DEMODB;

-- ============================================================================
-- CHUNKED_DOCUMENTS TABLE
-- Tracks all documents that have been chunked
-- ============================================================================
CREATE TABLE IF NOT EXISTS DEMODB.chunked_files (
    id SERIAL PRIMARY KEY,
    file_hash VARCHAR(64) UNIQUE NOT NULL,                      -- SHA256 hash of original file
    file_name VARCHAR(255) NOT NULL,                            -- Original filename
    file_type VARCHAR(20),                                       -- File type (txt, pdf, docx, md)
    file_path TEXT,                                             -- Path in bucket
    bucket_name VARCHAR(255),                                   -- MinIO bucket name
    num_chunks INTEGER NOT NULL,                                -- Number of chunks created
    chunking_strategy VARCHAR(50) NOT NULL,                     -- Strategy used (fixed_size, recursive, etc)
    chunk_size INTEGER NOT NULL,                                -- Chunk size parameter
    chunk_overlap INTEGER NOT NULL,                             -- Overlap parameter
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,             -- When chunking started
    end_time TIMESTAMP,                                         -- When chunking completed
    duration_ms INTEGER,                                        -- Duration in milliseconds
    status VARCHAR(20) DEFAULT 'pending',                       -- pending, processing, completed, failed, skipped
    error_message TEXT,                                         -- Error details if failed
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CHECK (num_chunks >= 0),
    CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'skipped'))
);

-- Indexes on chunked_files
CREATE INDEX IF NOT EXISTS idx_chunked_files_file_hash 
    ON DEMODB.chunked_files(file_hash);
CREATE INDEX IF NOT EXISTS idx_chunked_files_status 
    ON DEMODB.chunked_files(status);
CREATE INDEX IF NOT EXISTS idx_chunked_files_created_at 
    ON DEMODB.chunked_files(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chunked_files_file_name 
    ON DEMODB.chunked_files(file_name);
CREATE INDEX IF NOT EXISTS idx_chunked_files_file_type 
    ON DEMODB.chunked_files(file_type);

-- ============================================================================
-- CHUNK_EMBEDDINGS TABLE
-- Tracks embeddings for individual chunks
-- ============================================================================
CREATE TABLE IF NOT EXISTS DEMODB.file_chunks (
    id SERIAL PRIMARY KEY,
    chunk_hash VARCHAR(64) NOT NULL UNIQUE,                     -- SHA256 hash of chunk
    chunked_doc_id INTEGER NOT NULL,                            -- FK to chunked_files
    chunk_index INTEGER NOT NULL,                               -- Index of chunk (0-based)
    chunk_preview TEXT,                                         -- First 500 chars of chunk
    vector_dimension INTEGER,                                   -- Dimension of embedding vector
    duration_ms INTEGER,                                        -- Time to generate embedding
    qdrant_point_id BIGINT,                                     -- ID in Qdrant
    status VARCHAR(20) DEFAULT 'pending',                       -- pending, completed, failed
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (chunked_doc_id) REFERENCES DEMODB.chunked_files(id) ON DELETE CASCADE,
    CHECK (chunk_index >= 0),
    CHECK (status IN ('pending', 'completed', 'failed'))
);

-- Indexes on file_chunks
CREATE INDEX IF NOT EXISTS idx_file_chunks_chunked_doc_id 
    ON DEMODB.file_chunks(chunked_doc_id);
CREATE INDEX IF NOT EXISTS idx_file_chunks_chunk_hash 
    ON DEMODB.file_chunks(chunk_hash);
CREATE INDEX IF NOT EXISTS idx_file_chunks_qdrant_point_id 
    ON DEMODB.file_chunks(qdrant_point_id);
CREATE INDEX IF NOT EXISTS idx_file_chunks_status 
    ON DEMODB.file_chunks(status);

-- ============================================================================
-- PROCESSING_TRANSACTIONS TABLE
-- Logs all processing transactions for audit and debugging
-- ============================================================================
CREATE TABLE IF NOT EXISTS DEMODB.processed_files (
    id SERIAL PRIMARY KEY,
    message_id VARCHAR(255) NOT NULL,                           -- Solace message ID
    file_name VARCHAR(255) NOT NULL,
    file_hash VARCHAR(64),
    solace_queue VARCHAR(100),                                  -- Which queue
    status VARCHAR(20),                                         -- queued, processing, completed, failed
    action VARCHAR(100),                                        -- chunk, embed, skip, etc
    duration_ms INTEGER,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CHECK (status IN ('queued', 'processing', 'completed', 'failed'))
);

-- Indexes on processed_files
CREATE INDEX IF NOT EXISTS idx_processed_files_message_id 
    ON DEMODB.processed_files(message_id);
CREATE INDEX IF NOT EXISTS idx_processed_files_file_hash 
    ON DEMODB.processed_files(file_hash);
CREATE INDEX IF NOT EXISTS idx_processed_files_created_at 
    ON DEMODB.processed_files(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_processed_files_status 
    ON DEMODB.processed_files(status);

-- ============================================================================
-- TRIGGER FOR UPDATED_AT COLUMNS
-- ============================================================================

CREATE OR REPLACE FUNCTION DEMODB.update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_chunked_files_timestamp
    BEFORE UPDATE ON DEMODB.chunked_files
    FOR EACH ROW
    EXECUTE FUNCTION DEMODB.update_timestamp();

CREATE TRIGGER update_file_chunks_timestamp
    BEFORE UPDATE ON DEMODB.file_chunks
    FOR EACH ROW
    EXECUTE FUNCTION DEMODB.update_timestamp();

CREATE TRIGGER update_processed_files_timestamp
    BEFORE UPDATE ON DEMODB.processed_files
    FOR EACH ROW
    EXECUTE FUNCTION DEMODB.update_timestamp();
