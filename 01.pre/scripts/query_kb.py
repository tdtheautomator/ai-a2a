#!/usr/bin/env python3
"""Query Qdrant with LLM formatting and file filtering."""

import requests
import json
import sys
import argparse
from datetime import datetime
from typing import List, Dict, Any

class QdrantLLMQuery:
    def __init__(self, qdrant_url="http://localhost:6333", litellm_url="http://localhost:4000", 
                 collection="documents", embed_key="sk-1234", llm_key="sk-1234", llm_model="qwen3:0.6b",
                 query_file_name=None):
        self.qdrant_url = qdrant_url.rstrip("/")
        self.litellm_url = litellm_url.rstrip("/")
        self.collection = collection
        self.embed_key = embed_key
        self.llm_key = llm_key
        self.llm_model = llm_model
        self.query_file_name = query_file_name

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding from LiteLLM."""
        try:
            response = requests.post(
                f"{self.litellm_url}/embeddings",
                json={"model": "qwen3-embedding:0.6b", "input": text},
                headers={"Authorization": f"Bearer {self.embed_key}"},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            if "data" in data and data["data"]:
                return data["data"][0]["embedding"]
            return data.get("embedding", data)
        except Exception as e:
            raise Exception(f"Embedding failed: {str(e)}")

    def search_qdrant(self, query: str, limit: int = 5) -> List[Dict]:
        """Search Qdrant for similar documents and optionally filter by file_name."""
        try:
            embedding = self.get_embedding(query)
            
            response = requests.post(
                f"{self.qdrant_url}/collections/{self.collection}/points/search",
                json={"vector": embedding, "limit": limit * 2, "with_vectors": False, "with_payload": True},
                timeout=10
            )
            
            if response.status_code != 200:
                raise Exception(f"Search failed with status {response.status_code}")
            
            results = response.json().get("result", [])
            
            # Filter by file_name if query_file_name is provided
            if self.query_file_name:
                results = [r for r in results if r.get("payload", {}).get("file_name") == self.query_file_name]
                # Limit to original limit after filtering
                results = results[:limit]
            else:
                results = results[:limit]
            
            return results
        except Exception as e:
            raise Exception(f"Qdrant search failed: {str(e)}")

    def format_with_llm(self, question: str, results: List[Dict]) -> str:
        """Format search results using LLM."""
        try:
            results_text = ""
            for idx, result in enumerate(results, 1):
                score = result.get("score", 0)
                payload = result.get("payload", {})
                text = payload.get("text", "")
                file_name = payload.get("file_name", "Unknown")
                
                results_text += f"\n[Result {idx}] (Similarity: {score:.4f}) [From: {file_name}]\n{text}\n"
            
            prompt = f"""Question: {question}

Knowledge Base Results:
{results_text}

Provide a clear, professional answer in English using the above results. Include key points and mention the relevance scores. Also mention which documents/files the information came from."""

            response = requests.post(
                f"{self.litellm_url}/chat/completions",
                json={
                    "model": self.llm_model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant providing clear answers."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1000
                },
                headers={"Authorization": f"Bearer {self.llm_key}"},
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]
            return "No response from LLM"
        except Exception as e:
            raise Exception(f"LLM formatting failed: {str(e)}")

    def display_result(self, num: int, question: str, results: List[Dict], answer: str):
        """Display formatted result with file_name metadata."""
        print("\n" + "="*150)
        print(f"QUERY #{num}")
        print("="*150)
        print(f"\nQUESTION: {question}\n")
        
        if self.query_file_name:
            print(f"FILTER: Showing results only from file: {self.query_file_name}")
            print("-"*150)
        
        if results:
            print("SOURCE DOCUMENTS (with Similarity Scores and File Information):")
            print("-"*150)
            for idx, result in enumerate(results, 1):
                score = result.get("score", 0)
                doc_id = result.get("id")
                payload = result.get("payload", {})
                text = payload.get("text", "")[:95]
                file_name = payload.get("file_name", "Unknown File")
                
                if score >= 0.85:
                    quality = "[EXCELLENT]"
                elif score >= 0.75:
                    quality = "[VERY GOOD]"
                elif score >= 0.65:
                    quality = "[GOOD]"
                elif score >= 0.55:
                    quality = "[FAIR]"
                else:
                    quality = "[WEAK]"
                
                print(f"\n[{idx}] {quality} Score: {score:.6f} | File: {file_name} | ID: {doc_id}")
                print(f"    {text}...")
        else:
            if self.query_file_name:
                print(f"No results found in file: {self.query_file_name}")
            else:
                print("No results found.")
        
        print("\n" + "="*150)
        print("FORMATTED ANSWER (from LLM)")
        print("="*150)
        print(f"\n{answer}\n")
        print("="*150 + "\n")

    def process_file(self, filename: str, limit: int = 5, output: str = None):
        """Process questions from file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                questions = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"\nERROR: File '{filename}' not found")
            sys.exit(1)
        
        if not questions:
            print(f"\nERROR: No questions in '{filename}'")
            sys.exit(1)
        
        print("\n" + "="*150)
        print("QDRANT SEMANTIC SEARCH WITH LLM FORMATTING")
        print("="*150)
        print(f"\nFile: {filename}")
        print(f"Questions: {len(questions)}")
        print(f"Results per Query: {limit}")
        print(f"Collection: {self.collection}")
        print(f"LLM Model: {self.llm_model}")
        if self.query_file_name:
            print(f"FILE FILTER: {self.query_file_name}")
        print(f"Qdrant: {self.qdrant_url}")
        print(f"LiteLLM: {self.litellm_url}\n")
        
        all_results = {
            "metadata": {
                "file": filename,
                "total": len(questions),
                "limit": limit,
                "collection": self.collection,
                "llm_model": self.llm_model,
                "file_filter": self.query_file_name if self.query_file_name else "None (All Files)",
                "timestamp": datetime.now().isoformat()
            },
            "results": []
        }
        
        for idx, question in enumerate(questions, 1):
            preview = question[:70] + "..." if len(question) > 70 else question
            filter_info = f" [Filter: {self.query_file_name}]" if self.query_file_name else ""
            print(f"[{idx}/{len(questions)}] {preview}{filter_info}")
            
            try:
                results = self.search_qdrant(question, limit)
                answer = self.format_with_llm(question, results) if results else "No results found."
                
                self.display_result(idx, question, results, answer)
                
                search_results_list = []
                for r in results:
                    payload = r.get("payload", {})
                    search_results_list.append({
                        "id": r.get("id"),
                        "score": r.get("score", 0),
                        "file_name": payload.get("file_name", "Unknown"),
                        "text": payload.get("text", ""),
                        "metadata": {k: v for k, v in payload.items() if k not in ["text", "file_name"]}
                    })
                
                all_results["results"].append({
                    "query_id": idx,
                    "question": question,
                    "num_results": len(results),
                    "search_results": search_results_list,
                    "llm_answer": answer
                })
            except Exception as e:
                print(f"ERROR: {str(e)}\n")
                all_results["results"].append({
                    "query_id": idx,
                    "question": question,
                    "error": str(e)
                })
        
        print("\n" + "="*150)
        print("SUMMARY")
        print("="*150)
        successful = sum(1 for r in all_results['results'] if 'error' not in r)
        total_results = sum(r.get('num_results', 0) for r in all_results['results'] if 'error' not in r)
        print(f"\nProcessed: {len(questions)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(questions) - successful}")
        print(f"Total Results Found: {total_results}\n")
        
        if self.query_file_name:
            print(f"Results filtered for file: {self.query_file_name}\n")
        
        if output:
            try:
                with open(output, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
                print(f"Results saved to: {output}\n")
            except Exception as e:
                print(f"Error saving: {e}\n")

def main():
    parser = argparse.ArgumentParser(description="Query Qdrant with LLM formatting and file filtering")
    parser.add_argument("file", nargs="?", help="Questions file (one per line)")
    parser.add_argument("--limit", type=int, default=5, help="Results per query (default: 5)")
    parser.add_argument("--collection", default="documents", help="Collection name (default: documents)")
    parser.add_argument("--output", help="Save to JSON file")
    parser.add_argument("--qdrant-url", default="http://localhost:6333", help="Qdrant URL (default: http://localhost:6333)")
    parser.add_argument("--litellm-url", default="http://localhost:4000", help="LiteLLM URL (default: http://localhost:4000)")
    parser.add_argument("--embed-key", default="sk-1234", help="Embedding API key (default: sk-1234)")
    parser.add_argument("--llm-key", default="sk-1234", help="LLM API key (default: sk-1234)")
    parser.add_argument("--llm-model", default="qwen3:0.6b", help="LLM model name (default: qwen3:0.6b)")
    parser.add_argument("--query-file-name", help="Filter results by file_name (optional, if not provided searches all files)")
    
    args = parser.parse_args()
    
    if not args.file:
        parser.print_help()
        sys.exit(1)
    
    engine = QdrantLLMQuery(
        qdrant_url=args.qdrant_url,
        litellm_url=args.litellm_url,
        collection=args.collection,
        embed_key=args.embed_key,
        llm_key=args.llm_key,
        llm_model=args.llm_model,
        query_file_name=args.query_file_name
    )
    
    try:
        engine.process_file(args.file, args.limit, args.output)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
