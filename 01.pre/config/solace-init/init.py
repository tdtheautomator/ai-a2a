#!/usr/bin/env python3
"""
Solace Broker Initialization Script
Initializes a Solace message broker with configuration from JSON file
"""

import json
import sys
import logging
from typing import Dict, Any
import requests
from requests.auth import HTTPBasicAuth

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SolaceInitializer:
    """Initialize Solace broker with configuration"""

    def __init__(self, broker_host: str, admin_username: str, admin_password: str):
        """
        Initialize Solace broker connector
        
        Args:
            broker_host: Solace broker host (e.g., 'localhost:8080')
            admin_username: Admin username for SEMP API
            admin_password: Admin password for SEMP API
        """
        self.broker_host = broker_host
        self.base_url = f"http://{broker_host}/SEMP/v2/config"
        self.auth = HTTPBasicAuth(admin_username, admin_password)
        self.headers = {'Content-Type': 'application/json'}

    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> requests.Response:
        """Make HTTP request to SEMP API"""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            if method == 'GET':
                response = requests.get(url, auth=self.auth, headers=self.headers)
            elif method == 'POST':
                response = requests.post(url, auth=self.auth, headers=self.headers, json=data)
            elif method == 'PATCH':
                response = requests.patch(url, auth=self.auth, headers=self.headers, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Request failed for {method} {endpoint}: {e}")
            raise

    def check_connectivity(self, timeout: int = 5) -> bool:
        """Check connectivity to the Solace SEMP API using provided credentials.

        Performs a simple GET request against the configured SEMP config base
        path and returns True if authentication and connection succeed.
        """
        try:
            resp = requests.get(self.base_url, auth=self.auth, headers=self.headers, timeout=timeout)
            resp.raise_for_status()
            logger.info(f"✅ Connected to Solace SEMP API at {self.broker_host} (HTTP {resp.status_code})")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Connectivity check failed for {self.broker_host}: {e}")
            return False
    
    def create_msg_vpn(self, vpn_config: Dict[str, Any]) -> bool:
        """Create a message VPN"""
        vpn_name = vpn_config['msgVpnName']
        payload = {
            'msgVpnName': vpn_name,
            'enabled': vpn_config.get('enabled', True),
            'authenticationBasicEnabled': vpn_config.get('authenticationBasicEnabled', True),
            'maxMsgSpoolUsage': vpn_config.get('maxMsgSpoolUsage', 5000)
        }
        
        endpoint = "msgVpns"
        
        try:
            response = self._make_request('POST', endpoint, payload)
            logger.info(f"✅ Message VPN '{vpn_name}' created successfully")
            return True
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400 and 'already exists' in e.response.text:
                logger.info(f"⭕ Message VPN '{vpn_name}' already exists")
                return True
            else:
                logger.error(f"❌ Failed to create message VPN '{vpn_name}': {e}")
                logger.error(f"Response body: {e.response.text}")
                return False
        
    def create_queue(self, vpn_name: str, queue_config: Dict[str, Any]) -> bool:
        """Create a queue in the message VPN"""
        queue_name = queue_config['queueName']
        
        payload = {
            'queueName': queue_name,
            'ingressEnabled': queue_config.get('ingressEnabled', True),
            'egressEnabled': queue_config.get('egressEnabled', True),
            'accessType': queue_config.get('accessType', 'exclusive'),
            'owner': queue_config.get('owner', 'admin'),
            'permission': queue_config.get('permission', 'consume')  
        }
        
     
        endpoint = f"msgVpns/{vpn_name}/queues"

        # Attempt to create the queue. If it already exists, continue and still
        # try to add subscriptions.
        try:
            response = self._make_request('POST', endpoint, payload)
            logger.info(f"✅ Queue '{queue_name}' created successfully")
            created_or_exists = True
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400 and 'already exists' in e.response.text:
                logger.info(f"⭕ Queue '{queue_name}' already exists")
                created_or_exists = True
            else:
                logger.error(f"❌ Failed to create queue '{queue_name}': {e}")
                logger.error(f"Response body: {e.response.text}")
                return False

        # If queue was created or already exists, add any subscriptions provided
        if created_or_exists:
           subscription = queue_config['subscription']
           ok = self.add_queue_subscription(vpn_name, queue_name, subscription)
           all_ok = True
           if not ok:
             all_ok = False
           if not all_ok:
                logger.warning(f"⚠️ Queue '{queue_name}' created/existed but some subscriptions failed")
           return all_ok
        return True


    def add_queue_subscription(self, vpn_name: str, queue_name: str, subscription_topic: str) -> bool:
        """Add topic subscription to a queue"""
        payload = {
            'subscriptionTopic': subscription_topic
        }
        
        endpoint = f"msgVpns/{vpn_name}/queues/{queue_name}/subscriptions"
        
        try:
            response = self._make_request('POST', endpoint, payload)
            logger.info(f"✅ Subscription '{subscription_topic}' added to queue '{queue_name}'")
            return True
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400 and 'already exists' in e.response.text:
                logger.info(f"⭕ Subscription '{subscription_topic}' already exists for queue '{queue_name}'")
                return True
            else:
                logger.error(f"❌ Failed to add subscription to queue '{queue_name}': {e}")
                logger.error(f"Response body: {e.response.text}")
                return False
            

    def initialize(self, config: Dict[str, Any]) -> bool:
        success = True
        
        # Create message VPN
        vpns = config.get('msgVpn', [])
        if vpns:
            logger.info(f"ℹ️ Creating {len(vpns)} message VPN(s)...")
            for vpn_config in vpns:
                if not self.create_msg_vpn(vpn_config):
                    success = False
        
        # Setting default VPN name for queues and topic endpoints if not specified in config
        vpn_name = config.get('msgVpn', [{}])[0].get('msgVpnName', 'default')

        # Create queues
        queues = config.get('queues', [])
        if queues:
            logger.info(f"ℹ️ Creating {len(queues)} queue(s) on VPN '{vpn_name}'...")
            for queue_config in queues:
                if not self.create_queue(vpn_name, queue_config):
                    success = False
        
        endpoints = config.get('topicEndpoints', [])
        if endpoints:
            logger.warning("⚠️ topicEndpoints defined in config but create_topic_endpoint() is not implemented — skipping")
        # if endpoints:
        #     logger.info(f"ℹ️ Creating {len(endpoints)} topic endpoint(s) on VPN '{vpn_name}'...")
        #     for endpoint_config in endpoints:
        #         if not self.create_topic_endpoint(vpn_name, endpoint_config):
        #             success = False
        
        subscriptions = config.get('topicSubscriptions', [])
        if subscriptions:
            logger.warning("⚠️ topicSubscriptions defined in config but add_topic_subscription() is not implemented — skipping")
        # if subscriptions:
        #     logger.info(f"ℹ️ Creating {len(subscriptions)} topic subscription(s) on VPN '{vpn_name}'...")
        #     for sub_config in subscriptions:
        #         if not self.add_topic_subscription(vpn_name, sub_config):
        #             success = False
        
        if success:
            logger.info("✅ Solace broker initialization completed successfully!")
        else:
            logger.warning("⚠️ Solace broker initialization completed with some errors")

        return success


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            logger.info(f"✅ Configuration loaded from {config_file}")
            return config
    except FileNotFoundError:
        logger.error(f"❌ Configuration file not found: {config_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"⚠️ Invalid JSON in configuration file: {e}")
        sys.exit(1)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Initialize Solace broker with configuration'
    )
    parser.add_argument(
        'config_file',
        help='Path to configuration JSON file'
    )
    parser.add_argument(
        '--host',
        default='localhost:8080',
        help='Solace broker host (default: localhost:8080)'
    )
    parser.add_argument(
        '--username',
        default='admin',
        help='Admin username (default: admin)'
    )
    parser.add_argument(
        '--password',
        default='admin',
        help='Admin password (default: admin)'
    )
    
    args = parser.parse_args()
    """Initialize Solace broker with configuration"""
    logger.info("=" * 60)
    logger.info(f"Starting Solace broker initialization on {args.host}...")
    logger.info("=" * 60)
    # Create initializer and verify connectivity before making changes
    initializer = SolaceInitializer(args.host, args.username, args.password)

    logger.info(f"ℹ️ Checking connectivity to Solace broker at {args.host}...")
    ok = initializer.check_connectivity()
    if not ok:
        logger.error(f"❌ Cannot reach Solace broker at {args.host} with provided credentials. Aborting initialization.")
        sys.exit(2)

    # Load configuration
    config = load_config(args.config_file)

    # Initialize Solace broker
    success = initializer.initialize(config)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()