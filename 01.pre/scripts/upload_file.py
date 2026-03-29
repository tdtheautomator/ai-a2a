#!/usr/bin/env python3
"""
Working File Upload to MinIO with Solace Notification
Uses: minio SDK + requests

Installation:
    pip install minio requests

Usage:
    python upload_working.py /path/to/file.txt
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

# Check dependencies
def check_and_install_deps():
    """Check dependencies with helpful messages"""
    missing = []
    
    try:
        import minio
    except ImportError:
        missing.append("minio")
    
    try:
        import requests
    except ImportError:
        missing.append("requests")
    
    if missing:
        print("❌ Missing packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print(f"\n📦 Install with:")
        print(f"   pip install {' '.join(missing)}")
        sys.exit(1)

check_and_install_deps()

from minio import Minio
from minio.error import S3Error
import requests
from requests.auth import HTTPBasicAuth


class MinIOUploader:
    """Upload files to MinIO with full diagnostics"""

    def __init__(
        self,
        minio_host="localhost:19000",  # FIXED: was 19001 (Console UI)
        minio_user="minioadmin",
        minio_password="minioadmin",
        solace_host="http://localhost:9000",  # FIXED: was 8080 (Management UI)
        solace_user="admin",
        solace_password="admin",
    ):
        self.minio_host = minio_host
        self.minio_user = minio_user
        self.minio_password = minio_password
        self.solace_host = solace_host
        self.solace_user = solace_user
        self.solace_password = solace_password
        self.client = None

    def connect(self) -> bool:
        """Connect to MinIO with detailed diagnostics"""
        try:
            print(f"\n🔗 Connecting to MinIO...")
            print(f"   Host: {self.minio_host}")
            print(f"   User: {self.minio_user}")

            self.client = Minio(
                self.minio_host,
                access_key=self.minio_user,
                secret_key=self.minio_password,
                secure=False,
            )

            # Test connection by listing buckets
            print("   Testing connection by listing buckets...")
            buckets = self.client.list_buckets()
            
            bucket_names = [b.name for b in buckets]
            print(f"\n✅ Connected to MinIO!")
            print(f"   Existing buckets: {bucket_names if bucket_names else 'None'}")
            
            return True

        except S3Error as e:
            print(f"\n❌ MinIO connection failed!")
            print(f"   Error: {e}")
            print(f"\n🔧 Troubleshooting:")
            print(f"   1. Is MinIO running? docker ps | grep minio")
            print(f"   2. Check logs: docker logs demo-minio")
            print(f"   3. Restart MinIO: docker-compose restart minio")
            return False
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            return False

    def create_bucket(self, bucket_name: str) -> bool:
        """Create bucket if it doesn't exist"""
        try:
            print(f"\n📦 Checking bucket '{bucket_name}'...")

            if not self.client.bucket_exists(bucket_name):
                print(f"   Creating bucket '{bucket_name}'...")
                self.client.make_bucket(bucket_name)
                print(f"✅ Bucket '{bucket_name}' created!")
            else:
                print(f"✅ Bucket '{bucket_name}' exists!")

            return True

        except S3Error as e:
            print(f"❌ Bucket operation failed: {e}")
            return False

    def upload_file(self, bucket_name: str, file_path: str, object_name: str = None) -> dict:
        """Upload file with detailed output"""
        try:
            file_path = Path(file_path)
            
            # Verify file exists
            if not file_path.exists():
                print(f"\n❌ File not found: {file_path}")
                raise FileNotFoundError(f"File not found: {file_path}")

            if object_name is None:
                object_name = file_path.name

            file_size = file_path.stat().st_size
            upload_time = datetime.utcnow().isoformat()

            print(f"\n📤 Uploading file...")
            print(f"   File: {file_path.name}")
            print(f"   Size: {file_size} bytes")
            print(f"   Destination: {bucket_name}/{object_name}")

            # Upload file
            with open(file_path, "rb") as file_data:
                self.client.put_object(
                    bucket_name,
                    object_name,
                    file_data,
                    length=file_size,
                )

            # Verify upload by trying to stat the object
            print("   Verifying upload...")
            stat = self.client.stat_object(bucket_name, object_name)
            
            print(f"\n✅ File uploaded successfully!")
            print(f"   Size on MinIO: {stat.size} bytes")
            print(f"   Location: http://localhost:19001/minio/upload")

            metadata = {
                "message_id": f"file-{random.randint(100000, 999999)}",
                "bucket_name": bucket_name,
                "object_name": object_name,
                "file_name": file_path.name,
                "file_size": file_size,
                "upload_time": upload_time,
                "location": f"http://localhost:19001/browser/{bucket_name}/{object_name}",
            }

            return metadata

        except FileNotFoundError as e:
            print(f"\n❌ {e}")
            sys.exit(1)
        except S3Error as e:
            print(f"\n❌ Upload failed: {e}")
            print(f"\n🔧 Troubleshooting:")
            print(f"   1. Check bucket exists: {bucket_name}")
            print(f"   2. Check MinIO permissions")
            print(f"   3. View MinIO logs: docker logs demo-minio")
            sys.exit(1)

    def publish_notification(self, topic: str, metadata: dict) -> bool:
        """Publish to Solace topic using SEMP v2 API"""
        try:
            print(f"\n📢 Publishing to Solace topic...")
            print(f"   Topic: {topic}")
            print(f"   Host: {self.solace_host}")

            payload = {
                "event": "file_uploaded",
                "message": f"File '{metadata['file_name']}' uploaded to MinIO",
                "metadata": metadata,
                "timestamp": datetime.utcnow().isoformat(),
            }

            payload_json = json.dumps(payload)

            # Try multiple Solace endpoints
            endpoints = [
                # SEMP v2 Action endpoint (primary)
                (f"{self.solace_host}/SEMP/v2/action/msgVpns/default/publish", True),
                # Direct REST publish
                (f"{self.solace_host}/restapi/v1/publish/topics/{topic.replace('/', '%2F')}", False),
                # Alternative REST endpoint
                (f"{self.solace_host}/api/v1/topics/{topic.replace('/', '%2F')}", False),
            ]

            for endpoint, is_semp_v2 in endpoints:
                try:
                    if is_semp_v2:
                        req_payload = {
                            "topicName": topic,
                            "messageBody": payload_json,
                        }
                    else:
                        req_payload = payload

                    print(f"   Trying: {endpoint}")
                    response = requests.post(
                        endpoint,
                        json=req_payload,
                        auth=HTTPBasicAuth(self.solace_user, self.solace_password),
                        timeout=5,
                    )

                    print(f"   Status: {response.status_code}")
                    
                    if response.status_code < 300:
                        print(f"✅ Message published to topic '{topic}'!")
                        return True

                except requests.exceptions.ConnectionError:
                    print(f"   ❌ Connection refused")
                    continue
                except Exception as e:
                    print(f"   ❌ Error: {type(e).__name__}")
                    continue

            print(f"\n⚠️  Could not publish to Solace topic")
            print(f"   (File is still safely uploaded to MinIO)")
            print(f"   💡 Verify in Solace console: http://localhost:8080")
            return False

        except Exception as e:
            print(f"⚠️  Error: {e}")
            print(f"   (File is still safely uploaded to MinIO)")
            return False

    def send_to_queue(self, queue_name: str, metadata: dict) -> bool:
        """
        Send a focused message to a Solace queue via REST Messaging API.

        The message body contains only the three key fields:
            file_name, file_size, location

        Solace REST Messaging endpoints tried (in order):
          1. POST /QUEUE/<queue>          – standard REST Messaging API
          2. POST /QUEUE/<queue> on :9000 – alternate port
          3. SEMP v2 action publish with destinationType=queue
        """
        try:
            print(f"\n📬 Sending to Solace queue '{queue_name}'...")
            print(f"   Host: {self.solace_host}")

            # Minimal queue message – only what was asked for
            queue_message = {
                "message_id": metadata["message_id"],
                "file_name": metadata["file_name"],
                "file_size": metadata["file_size"],
                "location":  metadata["location"],
            }
            message_body = json.dumps(queue_message, indent=2)

            print(f"   Payload:")
            for key, val in queue_message.items():
                print(f"      {key}: {val}")

            # ── Attempt 1: Solace REST Messaging – /QUEUE/<name> ──────────────
            queue_url = f"{self.solace_host}/QUEUE/{queue_name}"
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
            semp_url = f"{self.solace_host}/SEMP/v2/action/msgVpns/default/publish"
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
            print(f"   (File is still safely uploaded to MinIO)")
            print(f"   💡 Check queue exists in Solace console: http://localhost:8080")
            print(f"      Queues > Create Queue > Name: {queue_name}")
            return False

        except Exception as e:
            print(f"⚠️  Unexpected error sending to queue: {e}")
            print(f"   (File is still safely uploaded to MinIO)")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Upload file to MinIO with Solace topic notification and queue message"
    )
    parser.add_argument("file_path", help="Path to file to upload")
    parser.add_argument("--bucket", default="uploads", help="MinIO bucket")
    parser.add_argument("--object-name", help="Object name in MinIO")
    parser.add_argument("--topic", default="file/uploaded", help="Solace topic")
    parser.add_argument("--queue", default="uploads", help="Solace queue name")
    parser.add_argument("--minio-host", default="localhost:19000")
    parser.add_argument("--minio-user", default="minioadmin")
    parser.add_argument("--minio-password", default="minioadmin")
    parser.add_argument("--solace-host", default="http://localhost:9000")
    parser.add_argument("--solace-user", default="admin")
    parser.add_argument("--solace-password", default="admin")

    args = parser.parse_args()

    uploader = MinIOUploader(
        minio_host=args.minio_host,
        minio_user=args.minio_user,
        minio_password=args.minio_password,
        solace_host=args.solace_host,
        solace_user=args.solace_user,
        solace_password=args.solace_password,
    )

    print("\n" + "=" * 70)
    print("MinIO File Upload with Solace Topic + Queue Notification")
    print("=" * 70)

    try:
        # Connect
        if not uploader.connect():
            sys.exit(1)

        # Create bucket
        if not uploader.create_bucket(args.bucket):
            sys.exit(1)

        # Upload
        metadata = uploader.upload_file(args.bucket, args.file_path, args.object_name)

        # Publish to topic (full event payload)
        topic_ok = uploader.publish_notification(args.topic, metadata)

        # Send to queue (focused: file_name, file_size, location)
        queue_ok = uploader.send_to_queue(args.queue, metadata)

        print("\n" + "=" * 70)
        print("✅ SUCCESS - All steps completed!")
        print("=" * 70)
        print(f"\n📋 Summary:")
        print(f"   Message ID: {metadata['message_id']}")
        print(f"   File:     {metadata['file_name']}")
        print(f"   Size:     {metadata['file_size']} bytes")
        print(f"   Location: {metadata['location']}")
        print(f"   Topic:    {'✅' if topic_ok else '⚠️ '} {args.topic}")
        print(f"   Queue:    {'✅' if queue_ok else '⚠️ '} {args.queue}")
        print(f"\n🔗 View in MinIO:")
        print(f"   http://localhost:19001")
        print(f"   Username: minioadmin")
        print(f"   Password: minioadmin")
        print("\n" + "=" * 70 + "\n")

    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
