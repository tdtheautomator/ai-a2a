#!/usr/bin/env python3
"""
Chunking & Embedding Agent with Solace, MinIO, and LiteLLM Integration
──────────────────────────────────────────────────────────────────────
Consumes file upload messages from Solace queue, chunks documents
(supporting TXT, MD, PDF, DOCX), generates embeddings via LiteLLM,
and stores vectors in Qdrant with full metadata tracking.

Supported file types: .txt, .md, .pdf, .docx, .doc
Chunking strategies: fixed_size, recursive, semantic, markdown

Installation:
    pip install -r requirements.txt

Usage:
    python agent.py --vector-size 1024                         
    python agent.py --chunking-strategy recursive --vector-size 1024
    python agent.py --collection my-documents --vector-size 1024
    python agent.py --re-chunk                    # Force re-chunking & delete existing embeddings
    python agent.py --vector-size 1024 --recreate-collection  # Delete & recreate collection for new vector size
"""

import argparse
import json
import sys
import logging
import importlib
import time
import hashlib
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from io import BytesIO

import requests
from minio import Minio
from minio.error import S3Error
import psycopg2
from psycopg2.extras import Json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from requests.auth import HTTPBasicAuth

# Try to import document libraries
try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

# ── Configure logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── ANSI colours ──────────────────────────────────────────────────────────────
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"
DIM = "\033[2m"

SEP = DIM + "─" * 80 + RESET
DSEP = BOLD + "=" * 80 + RESET
# ── Chunking Strategies ───────────────────────────────────────────────────────

class ChunkingStrategy(Enum):
    """Supported chunking strategies"""
    FIXED_SIZE = "fixed_size"           # Fixed size chunks with overlap
    RECURSIVE = "recursive"             # Recursive split (paragraphs → sentences → chars)
    SEMANTIC = "semantic"               # Split by semantic meaning (paragraphs)
    MARKDOWN = "markdown"               # Split by markdown headers


class DocumentChunker:
    """Handles document chunking with multiple strategies"""
    
    def __init__(self, strategy: str = "recursive", chunk_size: int = 500, chunk_overlap: int = 100):
        try:
            self.strategy = ChunkingStrategy(strategy)
        except ValueError:
            logger.warning(f"Unknown strategy {strategy}, using recursive")
            self.strategy = ChunkingStrategy.RECURSIVE
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, filename: str) -> Tuple[List[str], Dict[str, Any]]:
        """
        Chunk text using selected strategy
        
        Returns:
            Tuple of (chunks list, metadata dict)
        """
        metadata = {
            "strategy": self.strategy.value,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        
        if self.strategy == ChunkingStrategy.FIXED_SIZE:
            chunks = self._chunk_fixed_size(text)
        elif self.strategy == ChunkingStrategy.RECURSIVE:
            chunks = self._chunk_recursive(text)
        elif self.strategy == ChunkingStrategy.SEMANTIC:
            chunks = self._chunk_semantic(text)
        elif self.strategy == ChunkingStrategy.MARKDOWN:
            chunks = self._chunk_markdown(text)
        else:
            chunks = self._chunk_fixed_size(text)
        
        # Filter empty chunks
        chunks = [c.strip() for c in chunks if c.strip()]
        metadata["num_chunks"] = len(chunks)
        
        return chunks, metadata
    
    def _chunk_fixed_size(self, text: str) -> List[str]:
        """Fixed size chunks with overlap"""
        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        
        for i in range(0, len(text), step):
            chunk = text[i:i + self.chunk_size]
            if len(chunk.strip()) > 0:
                chunks.append(chunk)
            if i + self.chunk_size >= len(text):
                break
        
        return chunks
    
    def _chunk_recursive(self, text: str) -> List[str]:
        """Recursive split: paragraphs → sentences → chars"""
        separators = ["\n\n", "\n", ". ", " ", ""]
        chunks = self._recursive_split(text, separators, 0)
        return chunks
    
    def _recursive_split(self, text: str, separators: List[str], separator_index: int) -> List[str]:
        """Recursively split text using separators"""
        chunks = []
        
        if separator_index >= len(separators):
            return [text] if len(text) > 0 else []
        
        separator = separators[separator_index]
        
        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)
        
        good_splits = []
        other_splits = []
        
        for s in splits:
            if len(s) > self.chunk_size:
                good_splits.append(s)
            elif len(s) > 0:
                other_splits.append(s)
        
        # Process good splits recursively
        for good_split in good_splits:
            sub_chunks = self._recursive_split(good_split, separators, separator_index + 1)
            chunks.extend(sub_chunks)
        
        # Merge other splits
        if other_splits:
            separator_str = separator if separator != "" else ""
            merged = separator_str.join(other_splits)
            
            if len(merged) > self.chunk_size and separator_index < len(separators) - 1:
                sub_chunks = self._recursive_split(merged, separators, separator_index + 1)
                chunks.extend(sub_chunks)
            elif len(merged) > 0:
                chunks.append(merged)
        
        return chunks
    
    def _chunk_semantic(self, text: str) -> List[str]:
        """Simple semantic chunking by paragraphs"""
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current_chunk) + len(para) + 2 < self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _chunk_markdown(self, text: str) -> List[str]:
        """Split by markdown headers"""
        chunks = []
        current_chunk = ""
        lines = text.split("\n")
        
        for line in lines:
            # Check if line is a markdown header
            if line.startswith("#"):
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks


class DocumentProcessor:
    """Handles document type detection and text extraction"""
    
    @staticmethod
    def get_file_type(filename: str) -> str:
        """Get file type from filename"""
        return Path(filename).suffix.lower().strip(".")
    
    @staticmethod
    def extract_text(file_data: bytes, filename: str) -> Optional[str]:
        """Extract text from various file types"""
        extension = Path(filename).suffix.lower()
        
        try:
            if extension in ['.txt']:
                return file_data.decode('utf-8')
            elif extension == '.md':
                return file_data.decode('utf-8')
            elif extension == '.pdf':
                return DocumentProcessor._extract_pdf(file_data)
            elif extension in ['.docx', '.doc']:
                return DocumentProcessor._extract_docx(file_data)
            else:
                logger.warning(f"Unsupported file type: {extension}, treating as text")
                try:
                    return file_data.decode('utf-8')
                except UnicodeDecodeError:
                    return file_data.decode('utf-8', errors='ignore')
        except Exception as e:
            logger.error(f"Error extracting text from {filename}: {e}")
            return None
    
    @staticmethod
    def _extract_pdf(file_data: bytes) -> Optional[str]:
        """Extract text from PDF"""
        if not HAS_PDF:
            logger.warning("PyPDF2 not installed, cannot process PDF files. Install: pip install PyPDF2")
            return None
        
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_data))
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.extract_text()
            return text
        except Exception as e:
            logger.error(f"Failed to extract PDF: {e}")
            return None
    
    @staticmethod
    def _extract_docx(file_data: bytes) -> Optional[str]:
        """Extract text from DOCX/DOC"""
        if not HAS_DOCX:
            logger.warning("python-docx not installed, cannot process DOCX files. Install: pip install python-docx")
            return None
        
        try:
            doc = DocxDocument(BytesIO(file_data))
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Failed to extract DOCX: {e}")
            return None


# ── Solace SDK imports ────────────────────────────────────────────────────────

def _import_solace():
    """Import Solace SDK with error handling"""
    errors = []

    try:
        from solace.messaging.messaging_service import MessagingService
    except ImportError as e:
        errors.append(f"  MessagingService  → {e}")
        MessagingService = None

    try:
        from solace.messaging.resources.queue import Queue
    except ImportError as e:
        errors.append(f"  Queue             → {e}")
        Queue = None

    if errors:
        print("❌  solace-pubsubplus import errors:")
        for msg in errors:
            print(msg)
        print("\n  Try:  pip install --upgrade solace-pubsubplus")
        sys.exit(1)

    SolaceError = Exception
    for candidate in (
        "solace.messaging.errors.pubsubplus_client_error.PubSubPlusClientException",
        "solace.messaging.errors.PubSubPlusClientException",
    ):
        module_path, _, class_name = candidate.rpartition(".")
        try:
            mod = importlib.import_module(module_path)
            SolaceError = getattr(mod, class_name)
            break
        except (ImportError, AttributeError):
            continue

    return MessagingService, Queue, SolaceError


MessagingService, Queue, SolaceError = _import_solace()


# ── Chunking & Embedding Agent ────────────────────────────────────────────────

class ChunkingEmbeddingAgent:
    """Agent for chunking documents and generating embeddings"""

    def __init__(
        self,
        solace_host: str = "tcp://solace-broker:55555",
        solace_rest_url: str = "http://solace-broker:9000",
        solace_vpn: str = "default",
        solace_user: str = "admin",
        solace_password: str = "admin",
        minio_host: str = "minio:9000",
        minio_user: str = "minioadmin",
        minio_password: str = "minioadmin",
        qdrant_host: str = "qdrant",
        qdrant_port: int = 6333,
        qdrant_collection: str = "documents",
        qdrant_vector_size: int = 1024,
        litellm_url: str = "http://litellm:4000",
        litellm_key: str = "sk-1234",
        embedding_model: str = "qwen3-embedding:0.6b",
        postgres_host: str = "postgres",
        postgres_port: int = 5432,
        postgres_db: str = "DEMODB",
        postgres_user: str = "postgres",
        postgres_password: str = "postgres",
        chunking_strategy: str = "recursive",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        poll_interval: int = 15,
        batch_size: int = 5,
        re_chunk: bool = False,
        recreate_collection: bool = False,
    ):
        self.solace_host = solace_host
        self.solace_rest_url = solace_rest_url
        self.solace_vpn = solace_vpn
        self.solace_user = solace_user
        self.solace_password = solace_password
        
        self.minio_host = minio_host
        self.minio_user = minio_user
        self.minio_password = minio_password
        
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.qdrant_collection = qdrant_collection
        self.qdrant_vector_size = qdrant_vector_size
        
        self.litellm_url = litellm_url
        self.litellm_key = litellm_key
        self.embedding_model = embedding_model
        
        self.postgres_host = postgres_host
        self.postgres_port = postgres_port
        self.postgres_db = postgres_db
        self.postgres_user = postgres_user
        self.postgres_password = postgres_password
        
        self.chunking_strategy = chunking_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self.re_chunk = re_chunk
        self.recreate_collection = recreate_collection
        
        # Initialize chunker
        self.chunker = DocumentChunker(chunking_strategy, chunk_size, chunk_overlap)
        
        # Connection instances
        self.messaging_service = None
        self.receiver = None
        self.minio_client = None
        self.qdrant_client = None
        self.postgres_conn = None

    def connect_all(self) -> bool:
        """Connect to all services"""
        print(f"\n{DSEP}")
        print(f"{BOLD}  Chunking & Embedding Agent - Initializing{RESET}")
        print(f"{BOLD}  Chunking: {self.chunking_strategy} | Size: {self.chunk_size} | Overlap: {self.chunk_overlap}{RESET}")
        print(f"{BOLD}  Collection: {self.qdrant_collection} | Vectors: {self.qdrant_vector_size}D{RESET}")
        print(DSEP)
        
        if not self._connect_solace():
            return False
        if not self._connect_minio():
            return False
        if not self._connect_qdrant():
            return False
        if not self._connect_postgres():
            return False
        if not self._connect_litellm():
            return False
        
        print(f"\n{GREEN}✅ All connections established!{RESET}\n")
        return True

    def _connect_solace(self) -> bool:
        """Connect to Solace broker"""
        try:
            print(f"\n{DIM}  Connecting to Solace…{RESET}")
            
            broker_props = {
                "solace.messaging.transport.host": self.solace_host,
                "solace.messaging.service.vpn-name": self.solace_vpn,
                "solace.messaging.authentication.scheme.basic.username": self.solace_user,
                "solace.messaging.authentication.scheme.basic.password": self.solace_password,
            }
            
            self.messaging_service = (
                MessagingService.builder()
                .from_properties(broker_props)
                .build()
            )
            self.messaging_service.connect()
            
            print(f"  {GREEN}✅ Connected to Solace {self.solace_host}{RESET}")
            return True
        except Exception as e:
            print(f"  {RED}❌ Solace connection failed: {e}{RESET}")
            return False

    def _connect_minio(self) -> bool:
        """Connect to MinIO"""
        try:
            print(f"{DIM}  Connecting to MinIO…{RESET}")
            
            self.minio_client = Minio(
                self.minio_host,
                access_key=self.minio_user,
                secret_key=self.minio_password,
                secure=False,
            )
            
            self.minio_client.list_buckets()
            print(f"  {GREEN}✅ Connected to MinIO {self.minio_host}{RESET}")
            return True
        except Exception as e:
            print(f"  {RED}❌ MinIO connection failed: {e}{RESET}")
            return False

    def _connect_qdrant(self) -> bool:
        """Connect to Qdrant"""
        try:
            print(f"{DIM}  Connecting to Qdrant…{RESET}")
            
            self.qdrant_client = QdrantClient(
                host=self.qdrant_host,
                port=self.qdrant_port,
                timeout=10.0
            )
            
            self.qdrant_client.get_collections()
            print(f"  {GREEN}✅ Connected to Qdrant {self.qdrant_host}:{self.qdrant_port}{RESET}")
            return True
        except Exception as e:
            print(f"  {RED}❌ Qdrant connection failed: {e}{RESET}")
            return False

    def _connect_postgres(self) -> bool:
        """Connect to PostgreSQL"""
        try:
            print(f"{DIM}  Connecting to PostgreSQL…{RESET}")
            
            self.postgres_conn = psycopg2.connect(
                host=self.postgres_host,
                port=self.postgres_port,
                database=self.postgres_db,
                user=self.postgres_user,
                password=self.postgres_password,
                connect_timeout=5,
            )
            
            self.postgres_conn.autocommit = True
            print(f"  {GREEN}✅ Connected to PostgreSQL {self.postgres_host}:{self.postgres_port}{RESET}")
            return True
        except Exception as e:
            print(f"  {RED}❌ PostgreSQL connection failed: {e}{RESET}")
            return False

    def _connect_litellm(self) -> bool:
        """Test LiteLLM connectivity"""
        try:
            print(f"{DIM}  Connecting to LiteLLM…{RESET}")
            
            headers = {
                "Authorization": f"Bearer {self.litellm_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                f"{self.litellm_url}/health/liveliness",
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            
            print(f"  {GREEN}✅ Connected to LiteLLM {self.litellm_url}{RESET}")
            print(f"  {DIM}     Model: {self.embedding_model}{RESET}")
            return True
        except Exception as e:
            print(f"  {RED}❌ LiteLLM connection failed: {e}{RESET}")
            return False

    def setup_solace_receivers(self) -> bool:
        """Setup Solace queue receiver"""
        try:
            print(f"\n{DIM}  Setting up Solace receivers…{RESET}")
            
            # Build receiver for 'uploads' queue
            queue = Queue.durable_non_exclusive_queue("uploads")
            self.receiver = (
                self.messaging_service
                .create_persistent_message_receiver_builder()
                .build(queue)
            )
            self.receiver.start()
            print(f"  {GREEN}✅ Receiver ready on queue 'uploads'{RESET}")
            print(f"  {GREEN}✅ Response queue 'embed-agent/response' ready (via REST API){RESET}")

            return True
        except Exception as e:
            print(f"  {RED}❌ Solace receiver setup failed: {e}{RESET}")
            return False

    def setup_qdrant_collection(self) -> bool:
        """Ensure Qdrant collection exists with proper schema"""
        try:
            print(f"\n{DIM}  Checking Qdrant collection…{RESET}")
            
            collection_exists = False
            try:
                info = self.qdrant_client.get_collection(self.qdrant_collection)
                collection_exists = True
                vector_size = info.config.params.vectors.size
            except Exception:
                collection_exists = False
            
            # Handle recreate flag
            if collection_exists and self.recreate_collection:
                print(f"  {YELLOW}🔄 Deleting existing collection '{self.qdrant_collection}'…{RESET}")
                try:
                    self.qdrant_client.delete_collection(collection_name=self.qdrant_collection)
                    print(f"  {GREEN}✅ Collection deleted{RESET}")
                    collection_exists = False
                except Exception as e:
                    print(f"  {RED}❌ Failed to delete collection: {e}{RESET}")
                    return False
            
            # Create or verify collection
            if not collection_exists:
                print(f"  {DIM}  Creating collection '{self.qdrant_collection}' ({self.qdrant_vector_size}D, cosine)…{RESET}")
                
                self.qdrant_client.recreate_collection(
                    collection_name=self.qdrant_collection,
                    vectors_config=VectorParams(size=self.qdrant_vector_size, distance=Distance.COSINE),
                )
                
                print(f"  {GREEN}✅ Collection '{self.qdrant_collection}' created{RESET}")
            else:
                # Collection exists and no recreate flag
                # Verify vector size matches
                try:
                    info = self.qdrant_client.get_collection(self.qdrant_collection)
                    vector_size = info.config.params.vectors.size
                    if vector_size != self.qdrant_vector_size:
                        print(f"  {YELLOW}⚠️  Collection has vector size {vector_size}D, but {self.qdrant_vector_size}D requested{RESET}")
                        print(f"  {YELLOW}     Use --recreate-collection to delete and recreate with correct size{RESET}")
                        return False
                    print(f"  {GREEN}✅ Collection '{self.qdrant_collection}' exists ({vector_size}D){RESET}")
                except Exception as e:
                    print(f"  {RED}❌ Failed to verify collection: {e}{RESET}")
                    return False
            
            return True
                
        except Exception as e:
            print(f"  {RED}❌ Qdrant collection setup failed: {e}{RESET}")
            return False

    def _download_from_minio(self, bucket: str, object_name: str) -> Optional[bytes]:
        """Download file from MinIO"""
        try:
            response = self.minio_client.get_object(bucket, object_name)
            data = response.read()
            response.close()
            return data
        except S3Error as e:
            logger.error(f"Failed to download {object_name} from {bucket}: {e}")
            return None

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from LiteLLM"""
        try:
            headers = {
                "Authorization": f"Bearer {self.litellm_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{self.litellm_url}/embeddings",
                json={"model": self.embedding_model, "input": text},
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            if "data" in data and len(data["data"]) > 0:
                embedding = data["data"][0].get("embedding")
                if embedding:
                    return embedding
            
            return None
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return None

    def _is_already_chunked(self, file_hash: str) -> bool:
        """Check if file is already chunked"""
        try:
            with self.postgres_conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id FROM DEMODB.chunked_files 
                    WHERE file_hash = %s AND status = 'completed'
                    """,
                    (file_hash,)
                )
                result = cur.fetchone()
                return result is not None
        except Exception as e:
            logger.error(f"Error checking chunk status: {e}")
            return False

    def _delete_file_embeddings(self, file_hash: str) -> bool:
        """Delete all embeddings for a file from Qdrant"""
        try:
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="file_hash",
                        match=MatchValue(value=file_hash)
                    )
                ]
            )
            
            delete_result = self.qdrant_client.delete(
                collection_name=self.qdrant_collection,
                points_selector=filter_condition
            )
            
            logger.info(f"Deleted embeddings for file_hash {file_hash[:16]}…")
            return True
        except Exception as e:
            logger.error(f"Error deleting embeddings: {e}")
            return False

    def _delete_file_chunks(self, file_hash: str) -> bool:
        """Delete all chunk records for a file from PostgreSQL"""
        try:
            with self.postgres_conn.cursor() as cur:
                # First, get the chunked_doc_id(s)
                cur.execute(
                    """
                    SELECT id FROM DEMODB.chunked_files
                    WHERE file_hash = %s
                    """,
                    (file_hash,)
                )
                doc_ids = [row[0] for row in cur.fetchall()]
                
                # Delete chunk embeddings for these documents
                if doc_ids:
                    for doc_id in doc_ids:
                        cur.execute(
                            """
                            DELETE FROM DEMODB.file_chunks
                            WHERE chunked_doc_id = %s
                            """,
                            (doc_id,)
                        )
                    
                    # Delete the chunked documents
                    cur.execute(
                        """
                        DELETE FROM DEMODB.chunked_files
                        WHERE file_hash = %s
                        """,
                        (file_hash,)
                    )
                    
                    self.postgres_conn.commit()
                    logger.info(f"Deleted {len(doc_ids)} chunk record(s) for file_hash {file_hash[:16]}…")
                    return True
            return True
        except Exception as e:
            logger.error(f"Error deleting chunk records: {e}")
            return False

    def _record_chunking(
        self,
        file_hash: str,
        file_name: str,
        file_type: str,
        bucket: str,
        object_name: str,
        num_chunks: int,
        chunking_strategy: str,
        chunk_size: int,
        chunk_overlap: int,
        duration_ms: int,
        status: str = "completed",
        error_msg: Optional[str] = None
    ) -> Optional[int]:
        """Record chunking in PostgreSQL, return chunked_doc_id"""
        try:
            start_time = datetime.utcnow()
            end_time = datetime.utcfromtimestamp(start_time.timestamp() + duration_ms / 1000.0)
            
            with self.postgres_conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO DEMODB.chunked_files 
                    (file_hash, file_name, file_type, file_path, bucket_name, 
                     num_chunks, chunking_strategy, chunk_size, chunk_overlap,
                     start_time, end_time, duration_ms, status, error_message)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        file_hash, file_name, file_type, object_name, bucket,
                        num_chunks, chunking_strategy, chunk_size, chunk_overlap,
                        start_time, end_time, duration_ms, status, error_msg
                    )
                )
                result = cur.fetchone()
                self.postgres_conn.commit()
                return result[0] if result else None
        except Exception as e:
            logger.error(f"Error recording chunking: {e}")
            return None

    def _record_chunk_embedding(
        self,
        chunk_hash: str,
        chunked_doc_id: int,
        chunk_index: int,
        chunk_text: str,
        vector_dimension: int,
        duration_ms: int,
        qdrant_point_id: int,
        status: str = "completed"
    ) -> bool:
        """Record chunk embedding in PostgreSQL"""
        try:
            with self.postgres_conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO DEMODB.file_chunks 
                    (chunk_hash, chunked_doc_id, chunk_index, chunk_preview, 
                     vector_dimension, duration_ms, qdrant_point_id, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        chunk_hash, chunked_doc_id, chunk_index,
                        chunk_text[:500],
                        vector_dimension, duration_ms, qdrant_point_id, status
                    )
                )
                self.postgres_conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error recording chunk embedding: {e}")
            return False

    def _record_processing_transaction(
        self,
        message_id: str,
        file_name: str,
        file_hash: str,
        solace_queue: str,
        status: str,
        action: str,
        duration_ms: int,
        error_message: Optional[str] = None
    ) -> bool:
        """Record processing transaction in PostgreSQL"""
        try:
            with self.postgres_conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO DEMODB.processed_files
                    (message_id, file_name, file_hash, solace_queue, status, action, duration_ms, error_message)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        message_id, file_name, file_hash, solace_queue,
                        status, action, duration_ms, error_message
                    )
                )
                self.postgres_conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error recording processing transaction: {e}")
            return False

    def _send_embed_response(self, source_message_id: str, file_name: str, status: str) -> bool:
        """
        Send embedding response message to Solace queue 'embed-response'
        
        Args:
            source_message_id: Original message ID from the file upload
            file_name: Name of the processed file
            status: Processing status ('processed' or 'skipped')
        
        Returns:
            True if message sent successfully, False otherwise
        """
        try:
            # Generate new message ID with embed- prefix
            embed_message_id = f"embed-{random.randint(1000, 9999)}"
            
            queue_name = "embed-response"
            print(f"\n{DIM}  Sending response to Solace queue '{queue_name}'…{RESET}")
            
            # Build minimal queue message
            queue_message = {
                "message_id": embed_message_id,
                "source_message_id": source_message_id,
                "file_name": file_name,
                "status": status,
                "timestamp": datetime.utcnow().isoformat(),
            }
            message_body = json.dumps(queue_message, indent=2)
            
            print(f"   Payload:")
            for key, val in queue_message.items():
                print(f"      {key}: {val}")
            
             # ── Attempt 1: Solace REST Messaging – /QUEUE/<name> ──────────────
            queue_url = f"{self.solace_rest_url}/QUEUE/{queue_name}"
            try:
                print(f"\n   Trying REST Messaging: {queue_url}")
                response = requests.post(
                    queue_url,
                    data=message_body,
                    auth=HTTPBasicAuth(self.solace_user, self.solace_password),
                    headers={
                        "Content-Type": "application/json",
                        "Solace-Message-Type": "Persistent",
                    },
                    timeout=5,
                )
                print(f"   Status: {response.status_code}")
                if response.status_code < 300:
                    print(f"✅ Message sent to queue '{queue_name}'!")
                    return True
            except requests.exceptions.ConnectionError:
                print(f"   ❌ Connection refused")

            # ── Attempt 2: SEMP v2 action publish with destinationType=queue ──
            semp_url = f"{self.solace_rest_url}/SEMP/v2/action/msgVpns/default/publish"
            semp_payload = {
                "destinationName": queue_name,
                "destinationType":  "queue",
                "messageBody":      message_body,
            }
            try:
                print(f"\n   Trying SEMP v2 action: {semp_url}")
                response = requests.post(
                    semp_url,
                    json=semp_payload,
                    auth=HTTPBasicAuth(self.solace_user, self.solace_password),
                    timeout=5,
                )
                print(f"   Status: {response.status_code}")
                if response.status_code < 300:
                    print(f"✅ Message sent to queue '{queue_name}' via SEMP!")
                    return True
            except requests.exceptions.ConnectionError:
                print(f"   ❌ Connection refused")
            except Exception as e:
                print(f"   ❌ Error: {type(e).__name__}: {e}")

            # ── Attempt 3: REST Messaging on common alternate port 9001 ────────
            alt_host = self.solace_host.rsplit(":", 1)[0] + ":9001"
            alt_url = f"{alt_host}/QUEUE/{queue_name}"
            try:
                print(f"\n   Trying alternate port: {alt_url}")
                response = requests.post(
                    alt_url,
                    data=message_body,
                    auth=HTTPBasicAuth(self.solace_user, self.solace_password),
                    headers={
                        "Content-Type": "application/json",
                        "Solace-Message-Type": "Persistent",
                    },
                    timeout=5,
                )
                print(f"   Status: {response.status_code}")
                if response.status_code < 300:
                    print(f"✅ Message sent to queue '{queue_name}'!")
                    return True
            except requests.exceptions.ConnectionError:
                print(f"   ❌ Connection refused")
            except Exception as e:
                print(f"   ❌ Error: {type(e).__name__}: {e}")

            print(f"\n⚠️  Could not send message to queue '{queue_name}'")
            print(f"   (File was still chunked and embedded successfully)")
            print(f"   💡 Check queue exists in Solace console: http://localhost:8080")
            print(f"      Queues > Create Queue > Name: {queue_name}")
            return False

        except Exception as e:
            print(f"⚠️  Unexpected error sending to queue: {e}")
            print(f"   (File was still chunked and embedded successfully)")
            return False
            
        except Exception as e:
            logger.error(f"Error sending embed response: {e}")
            return False

    def process_message(self, msg_data: Dict[str, Any]) -> bool:
        """Process a single file upload message"""
        file_name = msg_data.get("file_name", "unknown")
        file_size = msg_data.get("file_size", 0)
        location = msg_data.get("location", "")
        message_id = msg_data.get("message_id", "")
        
        logger.info(f"\n{SEP}")
        logger.info(f"Processing: {file_name} ({file_size} bytes) | Message: {message_id}")
        
        # Parse location
        try:
            parts = location.split("/browser/")
            if len(parts) != 2:
                logger.error(f"Invalid location format: {location}")
                return False
            
            bucket = parts[1].split("/")[0]
            object_name = "/".join(parts[1].split("/")[1:])
        except Exception as e:
            logger.error(f"Failed to parse location: {e}")
            return False
        
        start_time = datetime.utcnow()
        file_hash = None
        
        try:
            # Download file
            print(f"  {DIM}Downloading from MinIO…{RESET}")
            file_data = self._download_from_minio(bucket, object_name)
            if file_data is None:
                duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                self._record_processing_transaction(
                    message_id, file_name, "", bucket,
                    "failed", "download", duration_ms,
                    "Failed to download file from MinIO"
                )
                return False
            
            file_hash = hashlib.sha256(file_data).hexdigest()
            file_type = DocumentProcessor.get_file_type(file_name)
            print(f"  {CYAN}Hash: {file_hash[:16]}… | Type: .{file_type}{RESET}")
            
            # Check if already chunked
            if self._is_already_chunked(file_hash):
                if not self.re_chunk:
                    logger.warning(f"File already chunked (use --re-chunk to reprocess)")
                    print(f"  {YELLOW}⭕ Skipping (already processed){RESET}")
                    # Send skipped status to embed-response queue
                    self._send_embed_response(message_id, file_name, "skipped")
                    duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                    self._record_processing_transaction(
                        message_id, file_name, file_hash, bucket,
                        "completed", "skip", duration_ms,
                        None
                    )
                    return True
                else:
                    # Re-chunking requested: delete existing embeddings and chunks
                    logger.warning(f"Re-chunking requested: deleting existing embeddings…")
                    print(f"  {YELLOW}🔄 Re-chunking: deleting existing embeddings…{RESET}")
                    
                    # Delete from Qdrant
                    if not self._delete_file_embeddings(file_hash):
                        logger.error("Failed to delete existing embeddings from Qdrant")
                    
                    # Delete from PostgreSQL
                    if not self._delete_file_chunks(file_hash):
                        logger.error("Failed to delete existing chunks from PostgreSQL")
                    
                    print(f"  {GREEN}✅ Old embeddings deleted, proceeding with re-chunking…{RESET}")
            
            # Extract text
            print(f"  {DIM}Extracting text from .{file_type}…{RESET}")
            text_content = DocumentProcessor.extract_text(file_data, file_name)
            if text_content is None:
                logger.error("Failed to extract text")
                duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                self._record_processing_transaction(
                    message_id, file_name, file_hash, bucket,
                    "failed", "extract", duration_ms,
                    f"Failed to extract text from .{file_type}"
                )
                return False
            
            print(f"  {CYAN}Text length: {len(text_content)} characters{RESET}")
            
            # Chunk document
            print(f"  {DIM}Chunking with {self.chunking_strategy}…{RESET}")
            chunks, chunk_metadata = self.chunker.chunk_text(text_content, file_name)
            print(f"  {GREEN}✅ Created {len(chunks)} chunks{RESET}")
            
            # Record chunking
            chunking_duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            chunked_doc_id = self._record_chunking(
                file_hash, file_name, file_type, bucket, object_name,
                len(chunks), self.chunking_strategy, self.chunk_size, self.chunk_overlap,
                chunking_duration_ms
            )
            
            if chunked_doc_id is None:
                logger.error("Failed to record chunking")
                duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                self._record_processing_transaction(
                    message_id, file_name, file_hash, bucket,
                    "failed", "chunk_record", duration_ms,
                    "Failed to record chunking in database"
                )
                return False
            
            # Embed each chunk
            print(f"  {DIM}Embedding {len(chunks)} chunks…{RESET}")
            embedded_count = 0
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_start = time.time()
                
                # Get embedding
                embedding = self._get_embedding(chunk)
                if embedding is None:
                    logger.warning(f"Failed to embed chunk {chunk_idx}")
                    continue
                
                embedding_duration_ms = int((time.time() - chunk_start) * 1000)
                
                # Store in Qdrant
                chunk_hash = hashlib.sha256(chunk.encode()).hexdigest()
                point_id = int(hashlib.md5(chunk_hash.encode()).hexdigest(), 16) % (2**31)
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "file_name": file_name,
                        "file_type": file_type,
                        "chunk_index": chunk_idx,
                        "chunk_hash": chunk_hash,
                        "file_hash": file_hash,
                        "chunking_strategy": self.chunking_strategy,
                        "chunk_preview": chunk[:200],
                        "text": chunk,               # FIX: full chunk text for retrieval
                    }
                )
                
                # Use REST API directly to ensure vector is properly stored
                try:
                    import requests
                    
                    qdrant_url = f"http://{self.qdrant_host}:{self.qdrant_port}/collections/{self.qdrant_collection}/points?wait=true"
                    
                    point_data = {
                        "id": point_id,
                        "vector": embedding,
                        "payload": {
                            "file_name": file_name,
                            "file_type": file_type,
                            "chunk_index": chunk_idx,
                            "chunk_hash": chunk_hash,
                            "file_hash": file_hash,
                            "chunking_strategy": self.chunking_strategy,
                            "chunk_preview": chunk[:200],
                            "text": chunk,               # FIX: full chunk text for retrieval
                        }
                    }
                    
                    response = requests.put(
                        qdrant_url,
                        json={"points": [point_data]},
                        timeout=30
                    )
                    
                    if response.status_code not in [200, 201]:
                        logger.error(f"Vector upsert failed: {response.status_code} - {response.text}")
                        continue
                        
                except Exception as e:
                    logger.error(f"Vector upsert error: {e}")
                    continue
                
                # Record embedding
                if self._record_chunk_embedding(
                    chunk_hash, chunked_doc_id, chunk_idx, chunk,
                    len(embedding), embedding_duration_ms, point_id
                ):
                    embedded_count += 1
            
            print(f"  {GREEN}✅ Embedded {embedded_count}/{len(chunks)} chunks{RESET}")
            
            # Send processed status to embed-response queue
            self._send_embed_response(message_id, file_name, "processed")
            
            # Record successful transaction
            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            self._record_processing_transaction(
                message_id, file_name, file_hash, bucket,
                "completed", "chunk_embed", duration_ms,
                None
            )
            
            logger.info(f"{GREEN}✅ SUCCESS: {file_name} | {len(chunks)} chunks | {embedded_count} embedded{RESET}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            self._record_processing_transaction(
                message_id, file_name, file_hash or "", bucket,
                "failed", "process", duration_ms,
                str(e)
            )
            return False

    def run(self) -> None:
        """Main agent loop"""
        print(f"\n{DSEP}")
        print(f"{BOLD}  Chunking & Embedding Agent - Running{RESET}")
        print(f"{BOLD}  Poll Interval: {self.poll_interval}s | Batch Size: {self.batch_size}{RESET}")
        print(DSEP)
        
        try:
            while True:
                try:
                    print(f"\n{DIM}{datetime.utcnow().isoformat()} - Polling for messages…{RESET}")
                    
                    count = 0
                    for _ in range(self.batch_size):
                        msg = self.receiver.receive_message(timeout=1000)
                        if msg is None:
                            break
                        
                        try:
                            payload = msg.get_payload_as_string() or ""
                            msg_data = json.loads(payload)
                            
                            success = self.process_message(msg_data)
                            self.receiver.ack(msg)
                            count += 1
                            
                        except json.JSONDecodeError as e:
                            logger.error(f"Invalid JSON in message: {e}")
                            self.receiver.ack(msg)
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                            self.receiver.ack(msg)
                    
                    if count == 0:
                        print(f"  {YELLOW}⭕ No messages{RESET}")
                    else:
                        print(f"  {GREEN}✅ Processed {count} message(s){RESET}")
                    
                    time.sleep(self.poll_interval)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    time.sleep(self.poll_interval)
        
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Gracefully shutdown"""
        print(f"\n{DIM}  Shutting down…{RESET}")
        
        try:
            if self.receiver:
                self.receiver.terminate()
        except Exception as e:
            logger.debug(f"Error terminating receiver: {e}")
        
        try:
            if self.messaging_service:
                self.messaging_service.disconnect()
        except Exception as e:
            logger.debug(f"Error disconnecting from Solace: {e}")
        
        try:
            if self.postgres_conn:
                self.postgres_conn.close()
        except Exception as e:
            logger.debug(f"Error closing PostgreSQL: {e}")
        
        print(f"  {GREEN}✅ Shutdown complete{RESET}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Chunking & Embedding Agent with LiteLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent.py
  python agent.py --chunking-strategy recursive --chunk-size 500
  python agent.py --collection my-docs --vector-size 1024
  python agent.py --re-chunk  # Force re-chunking
        """,
    )
    
    # Solace options
    parser.add_argument("--host", default="tcp://solace-broker:55555")
    parser.add_argument("--solace-rest-url", default="http://solace-broker:9000", help="Solace REST API URL for queue messaging")
    parser.add_argument("--vpn", default="default")
    parser.add_argument("--username", default="admin")
    parser.add_argument("--password", default="admin")
    
    # MinIO options
    parser.add_argument("--minio-host", default="minio:9000")
    parser.add_argument("--minio-user", default="minioadmin")
    parser.add_argument("--minio-password", default="minioadmin")
    
    # Qdrant options
    parser.add_argument("--qdrant-host", default="qdrant")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--collection", default="documents", help="Qdrant collection name")
    parser.add_argument("--vector-size", type=int, default=1024, help="Vector size (dims) - qwen3-embedding:0.6b outputs 1024d vectors")
    
    # LiteLLM options
    parser.add_argument("--litellm-url", default="http://litellm:4000")
    parser.add_argument("--litellm-key", default="sk-1234", help="LiteLLM API key for authentication")
    parser.add_argument("--model", default="qwen3-embedding:0.6b")
    
    # PostgreSQL options
    parser.add_argument("--postgres-host", default="postgres")
    parser.add_argument("--postgres-port", type=int, default=5432)
    parser.add_argument("--postgres-db", default="DEMODB")
    parser.add_argument("--postgres-user", default="postgres")
    parser.add_argument("--postgres-password", default="postgres")
    
    # Chunking options
    parser.add_argument("--chunking-strategy", default="recursive",
                       choices=["fixed_size", "recursive", "semantic", "markdown"])
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    
    # Agent options
    parser.add_argument("--poll-interval", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--re-chunk", action="store_true", help="Force re-chunking of existing files")
    parser.add_argument("--recreate-collection", action="store_true", help="Delete and recreate Qdrant collection (use when changing vector size)")
    
    args = parser.parse_args()
    
    agent = ChunkingEmbeddingAgent(
        solace_host=args.host,
        solace_rest_url=args.solace_rest_url,
        solace_vpn=args.vpn,
        solace_user=args.username,
        solace_password=args.password,
        minio_host=args.minio_host,
        minio_user=args.minio_user,
        minio_password=args.minio_password,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        qdrant_collection=args.collection,
        qdrant_vector_size=args.vector_size,
        litellm_url=args.litellm_url,
        litellm_key=args.litellm_key,
        embedding_model=args.model,
        postgres_host=args.postgres_host,
        postgres_port=args.postgres_port,
        postgres_db=args.postgres_db,
        postgres_user=args.postgres_user,
        postgres_password=args.postgres_password,
        chunking_strategy=args.chunking_strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        poll_interval=args.poll_interval,
        batch_size=args.batch_size,
        re_chunk=args.re_chunk,
        recreate_collection=args.recreate_collection,
    )
    
    if agent.connect_all():
        if agent.setup_solace_receivers():
            if agent.setup_qdrant_collection():
                agent.run()
            else:
                logger.error("Failed to setup Qdrant collection")
        else:
            logger.error("Failed to setup Solace receivers")
    else:
        logger.error("Failed to connect to services")
        sys.exit(1)


if __name__ == "__main__":
    main()
