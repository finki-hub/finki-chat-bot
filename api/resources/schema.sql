CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS question (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid (),
    name TEXT NOT NULL UNIQUE,
    content TEXT NOT NULL,
    user_id TEXT,
    links JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS question_name_idx ON question (name);

CREATE TABLE IF NOT EXISTS link (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid (),
    name TEXT NOT NULL UNIQUE,
    url TEXT NOT NULL,
    description TEXT,
    user_id TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS link_name_idx ON link (name);

-- Embeddings

ALTER TABLE question
ADD COLUMN IF NOT EXISTS embedding_llama3_3_70b vector (8192);

-- No indexing for llama3_3_70b because indexes support up to 2000 dimensions

ALTER TABLE question
ADD COLUMN IF NOT EXISTS embedding_bge_m3 vector (1024);

CREATE INDEX IF NOT EXISTS question_embedding_bge_m3_idx ON question USING hnsw (
    embedding_bge_m3 vector_cosine_ops
);

ALTER TABLE question
ADD COLUMN IF NOT EXISTS embedding_text_embedding_3_large vector (3072);

-- No indexing for text_embedding_3_large because indexes support up to 2000 dimensions

ALTER TABLE question
ADD COLUMN IF NOT EXISTS embedding_text_embedding_004 vector (768);

CREATE INDEX IF NOT EXISTS question_embedding_text_embedding_004_idx ON question USING hnsw (
    embedding_text_embedding_004 vector_cosine_ops
);
