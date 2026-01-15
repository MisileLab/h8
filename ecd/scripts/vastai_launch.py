#!/usr/bin/env python3
"""
vast.ai Programmatic Instance Launcher for ECD

This script uses the vast.ai CLI/API to:
1. Search for suitable GPU instances
2. Launch an instance with the onstart script
3. Optionally wait for completion and download results

Prerequisites:
    pip install vastai
    vastai set api-key YOUR_API_KEY

Usage:
    python scripts/vastai_launch.py --help
    python scripts/vastai_launch.py --gpu-name "RTX 4090" --disk 50
    python scripts/vastai_launch.py --min-gpu-ram 24 --max-price 0.50
    python scripts/vastai_launch.py --quick --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class InstanceConfig:
    """Configuration for vast.ai instance."""
    gpu_name: Optional[str] = None
    min_gpu_ram: float = 16.0  # GB
    min_ram: float = 32.0  # GB
    min_disk: float = 30.0  # GB
    max_price: float = 1.0  # $/hr
    num_gpus: int = 1
    cuda_version: float = 12.0
    image: str = "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
    

def check_vastai_cli() -> bool:
    """Check if vast.ai CLI is installed and configured."""
    try:
        result = subprocess.run(
            ["vastai", "show", "user"],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def search_instances(config: InstanceConfig, limit: int = 10) -> list[dict]:
    """Search for available instances matching criteria."""
    
    # Build search query
    query_parts = [
        f"gpu_ram >= {config.min_gpu_ram}",
        f"cpu_ram >= {config.min_ram}",
        f"disk_space >= {config.min_disk}",
        f"dph_total <= {config.max_price}",
        f"num_gpus >= {config.num_gpus}",
        f"cuda_vers >= {config.cuda_version}",
        "verified = true",
        "rentable = true",
    ]
    
    if config.gpu_name:
        query_parts.append(f'gpu_name = "{config.gpu_name}"')
    
    query = " ".join(query_parts)
    
    cmd = [
        "vastai", "search", "offers",
        query,
        "--order", "dph_total",
        "--limit", str(limit),
        "--raw"
    ]
    
    print(f"Searching: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Search failed: {result.stderr}", file=sys.stderr)
        return []
    
    try:
        offers = json.loads(result.stdout)
        return offers if isinstance(offers, list) else []
    except json.JSONDecodeError:
        print(f"Failed to parse offers: {result.stdout}", file=sys.stderr)
        return []


def format_offer(offer: dict) -> str:
    """Format an offer for display."""
    return (
        f"ID: {offer.get('id', 'N/A'):8} | "
        f"GPU: {offer.get('gpu_name', 'N/A'):20} | "
        f"RAM: {offer.get('gpu_ram', 0):.0f}GB | "
        f"Price: ${offer.get('dph_total', 0):.3f}/hr | "
        f"CUDA: {offer.get('cuda_max_good', 'N/A')}"
    )


def generate_onstart_script(
    repo_url: str,
    branch: str = "main",
    quick_mode: bool = False,
    skip_data: bool = False,
    steps: int = 20000,
    seeds: str = "0,1,2"
) -> str:
    """Generate the onstart script for the instance."""
    
    quick_flag = "--quick" if quick_mode else ""
    skip_data_flag = "--skip-data" if skip_data else ""
    
    script = f"""#!/bin/bash
set -euo pipefail

echo "=========================================="
echo "vast.ai Auto Deploy - ECD Project"
echo "=========================================="

export TRACK_B_STEPS={steps}
export TRACK_B_SEEDS="{seeds}"

# Install git
apt-get update -qq && apt-get install -y -qq git curl

# Clone repository
cd /workspace
if [[ -d "ecd/.git" ]]; then
    cd ecd && git pull origin {branch}
else
    git clone --branch {branch} {repo_url} ecd
    cd ecd
fi

# Run deployment
bash scripts/vastai_deploy.sh {quick_flag} {skip_data_flag}

echo "Deployment complete!"
"""
    return script


def create_instance(
    offer_id: int,
    config: InstanceConfig,
    onstart_script: str,
    disk_gb: float = 50.0,
    dry_run: bool = False
) -> Optional[int]:
    """Create a new instance from an offer."""
    
    # Write onstart script to temp file
    script_path = Path("/tmp/vastai_onstart.sh")
    script_path.write_text(onstart_script)
    
    cmd = [
        "vastai", "create", "instance",
        str(offer_id),
        "--image", config.image,
        "--disk", str(disk_gb),
        "--onstart-cmd", f"bash -c '{onstart_script}'",
        "--raw"
    ]
    
    if dry_run:
        print("\n[DRY RUN] Would execute:")
        print(f"  {' '.join(cmd[:6])} ...")
        print(f"\nOnstart script:\n{onstart_script[:500]}...")
        return None
    
    print(f"\nCreating instance from offer {offer_id}...")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Failed to create instance: {result.stderr}", file=sys.stderr)
        return None
    
    try:
        response = json.loads(result.stdout)
        instance_id = response.get("new_contract")
        print(f"Instance created: {instance_id}")
        return instance_id
    except (json.JSONDecodeError, KeyError):
        # Try to parse as plain instance ID
        try:
            return int(result.stdout.strip())
        except ValueError:
            print(f"Unexpected response: {result.stdout}")
            return None


def get_instance_status(instance_id: int) -> Optional[dict]:
    """Get the status of an instance."""
    cmd = ["vastai", "show", "instance", str(instance_id), "--raw"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        return None
    
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def wait_for_instance(instance_id: int, timeout: int = 3600) -> bool:
    """Wait for instance to be running."""
    print(f"\nWaiting for instance {instance_id} to be ready...")
    
    start = time.time()
    while time.time() - start < timeout:
        if status := get_instance_status(instance_id):
            state = status.get("actual_status", "unknown")
            print(f"  Status: {state}")
            
            if state == "running":
                ssh_host = status.get("ssh_host")
                ssh_port = status.get("ssh_port")
                print(f"\nInstance ready!")
                print(f"  SSH: ssh -p {ssh_port} root@{ssh_host}")
                return True
            elif state in ["error", "stopped", "destroyed"]:
                print(f"Instance failed: {state}")
                return False
        
        time.sleep(30)
    
    print("Timeout waiting for instance")
    return False


def destroy_instance(instance_id: int) -> bool:
    """Destroy an instance."""
    cmd = ["vastai", "destroy", "instance", str(instance_id)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Launch vast.ai instance for ECD experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search for RTX 4090 instances
  python scripts/vastai_launch.py --gpu-name "RTX 4090" --search-only
  
  # Launch cheapest instance with 24GB+ VRAM
  python scripts/vastai_launch.py --min-gpu-ram 24 --max-price 0.50
  
  # Quick test run
  python scripts/vastai_launch.py --quick --min-gpu-ram 16
  
  # Dry run to see what would happen
  python scripts/vastai_launch.py --dry-run
"""
    )
    
    # Instance selection
    parser.add_argument("--gpu-name", type=str, help="Specific GPU name (e.g., 'RTX 4090')")
    parser.add_argument("--min-gpu-ram", type=float, default=16.0, help="Minimum GPU RAM in GB")
    parser.add_argument("--min-ram", type=float, default=32.0, help="Minimum system RAM in GB")
    parser.add_argument("--min-disk", type=float, default=30.0, help="Minimum disk space in GB")
    parser.add_argument("--max-price", type=float, default=1.0, help="Maximum price per hour")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--disk", type=float, default=50.0, help="Disk size to allocate in GB")
    
    # Repository settings
    parser.add_argument("--repo-url", type=str, 
                        default="https://github.com/YOUR_USERNAME/ecd.git",
                        help="Git repository URL")
    parser.add_argument("--branch", type=str, default="main", help="Git branch")
    
    # Experiment settings
    parser.add_argument("--steps", type=int, default=20000, help="Training steps")
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Seeds (comma-separated)")
    parser.add_argument("--quick", action="store_true", help="Quick mode (10 steps)")
    parser.add_argument("--skip-data", action="store_true", help="Skip data preparation")
    
    # Actions
    parser.add_argument("--search-only", action="store_true", help="Only search, don't launch")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen")
    parser.add_argument("--wait", action="store_true", help="Wait for instance to be ready")
    parser.add_argument("--offer-id", type=int, help="Use specific offer ID instead of searching")
    
    args = parser.parse_args()
    
    # Check CLI
    if not check_vastai_cli():
        print("vast.ai CLI not installed or not configured.", file=sys.stderr)
        print("Install with: pip install vastai", file=sys.stderr)
        print("Configure with: vastai set api-key YOUR_API_KEY", file=sys.stderr)
        sys.exit(1)
    
    # Build config
    config = InstanceConfig(
        gpu_name=args.gpu_name,
        min_gpu_ram=args.min_gpu_ram,
        min_ram=args.min_ram,
        min_disk=args.min_disk,
        max_price=args.max_price,
        num_gpus=args.num_gpus,
    )
    
    # Search for instances
    if args.offer_id:
        offers = [{"id": args.offer_id}]
        print(f"Using specified offer: {args.offer_id}")
    else:
        print("Searching for available instances...")
        offers = search_instances(config)
        
        if not offers:
            print("No instances found matching criteria.", file=sys.stderr)
            sys.exit(1)
        
        print(f"\nFound {len(offers)} matching instances:")
        print("-" * 80)
        for offer in offers[:10]:
            print(format_offer(offer))
        print("-" * 80)
    
    if args.search_only:
        print("\n--search-only specified, not launching.")
        sys.exit(0)
    
    # Generate onstart script
    onstart = generate_onstart_script(
        repo_url=args.repo_url,
        branch=args.branch,
        quick_mode=args.quick,
        skip_data=args.skip_data,
        steps=args.steps,
        seeds=args.seeds,
    )
    
    # Select best offer
    best_offer = offers[0]
    offer_id = best_offer.get("id") if isinstance(best_offer, dict) else args.offer_id
    
    print(f"\nSelected offer: {offer_id}")
    if isinstance(best_offer, dict):
        print(format_offer(best_offer))
    
    # Create instance
    instance_id = create_instance(
        offer_id=offer_id,
        config=config,
        onstart_script=onstart,
        disk_gb=args.disk,
        dry_run=args.dry_run
    )
    
    if args.dry_run:
        print("\n[DRY RUN] No instance created.")
        sys.exit(0)
    
    if not instance_id:
        print("Failed to create instance.", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nInstance {instance_id} created successfully!")
    print("Monitor at: https://vast.ai/console/instances/")
    
    if args.wait:
        success = wait_for_instance(instance_id)
        sys.exit(0 if success else 1)
    else:
        print("\nTo wait for instance: vastai show instance", instance_id)
        print("To SSH: vastai ssh-url", instance_id)
        print("To destroy: vastai destroy instance", instance_id)


if __name__ == "__main__":
    main()
