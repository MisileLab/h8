#!/usr/bin/env python3
"""
Cloud Deployment Script for ECD (Embedding Compression DB)

A unified launcher for deploying Track-B experiments to cloud GPU providers.

Supported Providers:
  - vast.ai   (SSH-based VM instances)
  - SaladCloud (Container-based deployment)

Usage:
    python scripts/cloud_deploy.py --help
    python scripts/cloud_deploy.py vast --search
    python scripts/cloud_deploy.py vast --launch --gpu "RTX 4090"
    python scripts/cloud_deploy.py salad --create --image myregistry/ecd:latest
    python scripts/cloud_deploy.py local  # Run locally

Prerequisites:
    vast.ai:
        pip install vastai
        vastai set api-key YOUR_API_KEY
    
    SaladCloud:
        export SALAD_API_KEY=your_key
        export SALAD_ORG_NAME=your_org
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

# Optional imports
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    steps: int = 20000
    seeds: str = "0,1,2"
    quick_mode: bool = False
    skip_data: bool = False
    

@dataclass 
class VastConfig:
    """vast.ai instance configuration."""
    gpu_name: Optional[str] = None
    min_gpu_ram: float = 16.0
    min_ram: float = 32.0
    min_disk: float = 30.0
    max_price: float = 1.0
    num_gpus: int = 1
    disk_size: float = 50.0
    image: str = "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"


@dataclass
class SaladConfig:
    """SaladCloud container configuration."""
    image: str = ""
    project: str = "ecd-experiments"
    gpu_class: str = "rtx4090"
    cpu: int = 4
    memory: int = 32768
    results_bucket: str = ""


# =============================================================================
# Utilities
# =============================================================================

class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    BOLD = '\033[1m'
    NC = '\033[0m'


def log(msg: str, color: str = Colors.GREEN):
    print(f"{color}[{time.strftime('%H:%M:%S')}]{Colors.NC} {msg}")


def warn(msg: str):
    print(f"{Colors.YELLOW}[WARN]{Colors.NC} {msg}", file=sys.stderr)


def error(msg: str):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}", file=sys.stderr)


def info(msg: str):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {msg}")


def run_command(cmd: list[str], capture: bool = False) -> tuple[int, str]:
    """Run a shell command."""
    try:
        if capture:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return result.returncode, result.stdout
        else:
            result = subprocess.run(cmd)
            return result.returncode, ""
    except subprocess.TimeoutExpired:
        return -1, "Command timed out"
    except FileNotFoundError:
        return -1, f"Command not found: {cmd[0]}"


def check_in_project_root() -> bool:
    """Check if we're in the project root directory."""
    markers = ["pyproject.toml", "scripts/run_track_b.sh"]
    return all(Path(m).exists() for m in markers)


# =============================================================================
# vast.ai Provider
# =============================================================================

class VastAIProvider:
    """vast.ai deployment provider."""
    
    def __init__(self, config: VastConfig, training: TrainingConfig, dry_run: bool = False):
        self.config = config
        self.training = training
        self.dry_run = dry_run
    
    def check_cli(self) -> bool:
        """Check if vast.ai CLI is available."""
        code, _ = run_command(["vastai", "show", "user"], capture=True)
        return code == 0
    
    def search(self, limit: int = 10) -> list[dict]:
        """Search for available instances."""
        query_parts = [
            f"gpu_ram >= {self.config.min_gpu_ram}",
            f"cpu_ram >= {self.config.min_ram}",
            f"disk_space >= {self.config.min_disk}",
            f"dph_total <= {self.config.max_price}",
            f"num_gpus >= {self.config.num_gpus}",
            "verified = true",
            "rentable = true",
        ]
        
        if self.config.gpu_name:
            query_parts.append(f'gpu_name = "{self.config.gpu_name}"')
        
        query = " ".join(query_parts)
        cmd = ["vastai", "search", "offers", query, "--order", "dph_total", 
               "--limit", str(limit), "--raw"]
        
        log(f"Searching vast.ai: {' '.join(cmd[:6])}...")
        code, output = run_command(cmd, capture=True)
        
        if code != 0:
            error("Search failed")
            return []
        
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return []
    
    def format_offer(self, offer: dict) -> str:
        """Format an offer for display."""
        return (
            f"ID: {offer.get('id', 'N/A'):8} | "
            f"GPU: {offer.get('gpu_name', 'N/A'):20} | "
            f"RAM: {offer.get('gpu_ram', 0):.0f}GB | "
            f"${offer.get('dph_total', 0):.3f}/hr"
        )
    
    def generate_onstart(self, repo_url: str, branch: str = "main") -> str:
        """Generate onstart script."""
        quick = "--quick" if self.training.quick_mode else ""
        skip_data = "--skip-data" if self.training.skip_data else ""
        
        return f"""#!/bin/bash
set -euo pipefail
export TRACK_B_STEPS={self.training.steps}
export TRACK_B_SEEDS="{self.training.seeds}"
apt-get update -qq && apt-get install -y -qq git curl
cd /workspace
git clone --branch {branch} {repo_url} ecd 2>/dev/null || (cd ecd && git pull)
cd ecd
bash scripts/vastai_deploy.sh {quick} {skip_data}
"""
    
    def launch(self, offer_id: int, repo_url: str, branch: str = "main") -> Optional[int]:
        """Launch an instance."""
        onstart = self.generate_onstart(repo_url, branch)
        
        if self.dry_run:
            info("[DRY RUN] Would create instance:")
            print(f"  Offer ID: {offer_id}")
            print(f"  Image: {self.config.image}")
            print(f"  Disk: {self.config.disk_size}GB")
            print(f"\nOnstart script:\n{onstart[:300]}...")
            return None
        
        cmd = [
            "vastai", "create", "instance", str(offer_id),
            "--image", self.config.image,
            "--disk", str(self.config.disk_size),
            "--onstart-cmd", onstart,
            "--raw"
        ]
        
        log("Creating instance...")
        code, output = run_command(cmd, capture=True)
        
        if code != 0:
            error("Failed to create instance")
            return None
        
        try:
            data = json.loads(output)
            instance_id = data.get("new_contract")
            log(f"Instance created: {instance_id}")
            return instance_id
        except (json.JSONDecodeError, KeyError):
            try:
                return int(output.strip())
            except ValueError:
                error(f"Unexpected response: {output}")
                return None
    
    def get_ssh_command(self, instance_id: int) -> Optional[str]:
        """Get SSH command for instance."""
        cmd = ["vastai", "ssh-url", str(instance_id)]
        code, output = run_command(cmd, capture=True)
        return output.strip() if code == 0 else None
    
    def destroy(self, instance_id: int) -> bool:
        """Destroy an instance."""
        if self.dry_run:
            info(f"[DRY RUN] Would destroy instance {instance_id}")
            return True
        
        cmd = ["vastai", "destroy", "instance", str(instance_id)]
        code, _ = run_command(cmd)
        return code == 0


# =============================================================================
# SaladCloud Provider  
# =============================================================================

class SaladCloudProvider:
    """SaladCloud deployment provider."""
    
    API_BASE = "https://api.salad.com/api/public"
    
    def __init__(self, config: SaladConfig, training: TrainingConfig, dry_run: bool = False):
        self.config = config
        self.training = training
        self.dry_run = dry_run
        
        self.api_key = os.environ.get("SALAD_API_KEY", "")
        self.org_name = os.environ.get("SALAD_ORG_NAME", "")
    
    def check_credentials(self) -> bool:
        """Check if credentials are set."""
        return bool(self.api_key and self.org_name)
    
    def _request(self, method: str, path: str, data: Optional[dict] = None) -> dict:
        """Make API request."""
        if not HAS_REQUESTS:
            error("requests library required: pip install requests")
            return {"error": "requests not installed"}
        
        url = f"{self.API_BASE}/organizations/{self.org_name}/{path}"
        headers = {"Salad-Api-Key": self.api_key, "Content-Type": "application/json"}
        
        response = requests.request(method, url, headers=headers, json=data)
        
        if response.status_code >= 400:
            return {"error": response.text, "status_code": response.status_code}
        
        try:
            return response.json()
        except json.JSONDecodeError:
            return {"raw": response.text}
    
    def list_groups(self) -> list[dict]:
        """List container groups."""
        result = self._request("GET", f"projects/{self.config.project}/containers")
        return result.get("items", [])
    
    def create(self) -> Optional[str]:
        """Create container group."""
        gpu_map = {
            "rtx4090": ["rtx4090_24gb"],
            "rtx3090": ["rtx3090_24gb"],
            "a100": ["a100_40gb", "a100_80gb"],
            "any_24gb": ["rtx4090_24gb", "rtx3090_24gb"],
        }
        
        env_vars = {
            "TRACK_B_STEPS": str(self.training.steps),
            "TRACK_B_SEEDS": self.training.seeds,
            "QUICK_MODE": "true" if self.training.quick_mode else "false",
            "UPLOAD_RESULTS": "true" if self.config.results_bucket else "false",
        }
        
        if self.config.results_bucket:
            env_vars["RESULTS_BUCKET"] = self.config.results_bucket
        
        group_name = f"ecd-track-b-{int(time.time())}"
        
        data = {
            "name": group_name,
            "display_name": f"ECD Track-B ({self.training.steps} steps)",
            "container": {
                "image": self.config.image,
                "resources": {
                    "cpu": self.config.cpu,
                    "memory": self.config.memory,
                    "gpu_classes": gpu_map.get(self.config.gpu_class, [self.config.gpu_class])
                },
                "environment_variables": env_vars,
            },
            "replicas": 1,
            "restart_policy": "on_failure",
        }
        
        if self.dry_run:
            info("[DRY RUN] Would create container group:")
            print(json.dumps(data, indent=2))
            return None
        
        log(f"Creating container group: {group_name}")
        result = self._request("POST", f"projects/{self.config.project}/containers", data)
        
        if "error" in result:
            error(f"Failed: {result}")
            return None
        
        # Start the group
        self._request("POST", f"projects/{self.config.project}/containers/{group_name}/start")
        log(f"Container group created and started: {group_name}")
        return group_name
    
    def status(self, group_name: str) -> dict:
        """Get container group status."""
        return self._request("GET", f"projects/{self.config.project}/containers/{group_name}")
    
    def logs(self, group_name: str) -> str:
        """Get logs from container group."""
        instances = self._request("GET", 
            f"projects/{self.config.project}/containers/{group_name}/instances")
        
        all_logs = []
        for inst in instances.get("instances", []):
            inst_id = inst.get("machine_id")
            if inst_id:
                result = self._request("GET",
                    f"projects/{self.config.project}/containers/{group_name}/instances/{inst_id}/logs")
                all_logs.append(result.get("logs", ""))
        
        return "\n".join(all_logs)
    
    def stop(self, group_name: str) -> bool:
        """Stop container group."""
        if self.dry_run:
            info(f"[DRY RUN] Would stop {group_name}")
            return True
        
        result = self._request("POST", 
            f"projects/{self.config.project}/containers/{group_name}/stop")
        return "error" not in result
    
    def delete(self, group_name: str) -> bool:
        """Delete container group."""
        if self.dry_run:
            info(f"[DRY RUN] Would delete {group_name}")
            return True
        
        result = self._request("DELETE",
            f"projects/{self.config.project}/containers/{group_name}")
        return "error" not in result


# =============================================================================
# Local Runner
# =============================================================================

class LocalRunner:
    """Run training locally."""
    
    def __init__(self, training: TrainingConfig, dry_run: bool = False):
        self.training = training
        self.dry_run = dry_run
    
    def run(self) -> int:
        """Run Track-B locally."""
        if not check_in_project_root():
            error("Not in project root. Run from directory containing pyproject.toml")
            return 1
        
        cmd = ["bash", "scripts/run_track_b.sh"]
        
        if self.training.quick_mode:
            cmd.append("--quick")
        
        env = os.environ.copy()
        env["STEPS"] = str(self.training.steps)
        env["SEEDS"] = self.training.seeds
        
        log(f"Running: {' '.join(cmd)}")
        log(f"  Steps: {self.training.steps}")
        log(f"  Seeds: {self.training.seeds}")
        
        if self.dry_run:
            info("[DRY RUN] Would execute above command")
            return 0
        
        result = subprocess.run(cmd, env=env)
        return result.returncode


# =============================================================================
# CLI
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Cloud Deployment for ECD Track-B Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search vast.ai for GPUs
  python scripts/cloud_deploy.py vast --search

  # Launch on vast.ai
  python scripts/cloud_deploy.py vast --launch --gpu "RTX 4090" --repo YOUR_REPO_URL

  # Deploy to SaladCloud  
  python scripts/cloud_deploy.py salad --create --image myregistry/ecd:latest

  # Run locally
  python scripts/cloud_deploy.py local --quick

  # Dry run (show what would happen)
  python scripts/cloud_deploy.py vast --launch --dry-run
"""
    )
    
    # Global options
    parser.add_argument("--dry-run", "-n", action="store_true", help="Show what would happen")
    parser.add_argument("--steps", type=int, default=20000, help="Training steps")
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Seeds (comma-separated)")
    parser.add_argument("--quick", "-q", action="store_true", help="Quick mode (10 steps)")
    parser.add_argument("--skip-data", action="store_true", help="Skip data preparation")
    
    subparsers = parser.add_subparsers(dest="provider", help="Cloud provider")
    
    # vast.ai subcommand
    vast = subparsers.add_parser("vast", help="vast.ai deployment")
    vast.add_argument("--search", action="store_true", help="Search for instances")
    vast.add_argument("--launch", action="store_true", help="Launch instance")
    vast.add_argument("--destroy", type=int, metavar="ID", help="Destroy instance")
    vast.add_argument("--ssh", type=int, metavar="ID", help="Get SSH command")
    vast.add_argument("--gpu", type=str, help="GPU name filter")
    vast.add_argument("--min-vram", type=float, default=16.0, help="Minimum GPU RAM (GB)")
    vast.add_argument("--max-price", type=float, default=1.0, help="Max price ($/hr)")
    vast.add_argument("--disk", type=float, default=50.0, help="Disk size (GB)")
    vast.add_argument("--repo", type=str, default="https://github.com/YOUR_USER/ecd.git",
                      help="Git repo URL")
    vast.add_argument("--branch", type=str, default="main", help="Git branch")
    vast.add_argument("--offer-id", type=int, help="Specific offer ID to use")
    
    # SaladCloud subcommand
    salad = subparsers.add_parser("salad", help="SaladCloud deployment")
    salad.add_argument("--list", action="store_true", help="List container groups")
    salad.add_argument("--create", action="store_true", help="Create container group")
    salad.add_argument("--status", type=str, metavar="NAME", help="Get status")
    salad.add_argument("--logs", type=str, metavar="NAME", help="Get logs")
    salad.add_argument("--stop", type=str, metavar="NAME", help="Stop container group")
    salad.add_argument("--delete", type=str, metavar="NAME", help="Delete container group")
    salad.add_argument("--image", type=str, help="Container image URL")
    salad.add_argument("--project", type=str, default="ecd-experiments", help="Project name")
    salad.add_argument("--gpu", type=str, default="rtx4090", 
                       help="GPU class (rtx4090, rtx3090, a100, any_24gb)")
    salad.add_argument("--results-bucket", type=str, help="S3/GCS bucket for results")
    
    # Local subcommand
    local = subparsers.add_parser("local", help="Run locally")
    
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.provider:
        parser.print_help()
        print(f"\n{Colors.CYAN}Choose a provider: vast, salad, or local{Colors.NC}")
        return 1
    
    # Build training config
    training = TrainingConfig(
        steps=args.steps,
        seeds=args.seeds,
        quick_mode=args.quick,
        skip_data=getattr(args, 'skip_data', False),
    )
    
    # Handle providers
    if args.provider == "vast":
        return handle_vast(args, training)
    elif args.provider == "salad":
        return handle_salad(args, training)
    elif args.provider == "local":
        return handle_local(args, training)
    
    return 1


def handle_vast(args, training: TrainingConfig) -> int:
    """Handle vast.ai commands."""
    config = VastConfig(
        gpu_name=args.gpu,
        min_gpu_ram=args.min_vram,
        max_price=args.max_price,
        disk_size=args.disk,
    )
    
    provider = VastAIProvider(config, training, args.dry_run)
    
    if not provider.check_cli():
        error("vast.ai CLI not configured")
        print("\nSetup:")
        print("  pip install vastai")
        print("  vastai set api-key YOUR_API_KEY")
        return 1
    
    if args.search:
        offers = provider.search()
        if not offers:
            warn("No instances found matching criteria")
            return 1
        
        print(f"\n{Colors.BOLD}Available Instances:{Colors.NC}")
        print("-" * 70)
        for offer in offers[:15]:
            print(provider.format_offer(offer))
        print("-" * 70)
        print(f"\nTo launch: python scripts/cloud_deploy.py vast --launch --offer-id ID")
        return 0
    
    if args.launch:
        # Search for best offer if no ID specified
        if args.offer_id:
            offer_id = args.offer_id
        else:
            offers = provider.search(limit=5)
            if not offers:
                error("No instances found")
                return 1
            offer_id = offers[0].get("id")
            print(f"Selected: {provider.format_offer(offers[0])}")
        
        instance_id = provider.launch(offer_id, args.repo, args.branch)
        
        if instance_id:
            print(f"\n{Colors.GREEN}Instance {instance_id} created!{Colors.NC}")
            print(f"\nMonitor: https://vast.ai/console/instances/")
            print(f"SSH:     vastai ssh-url {instance_id}")
            print(f"Destroy: python scripts/cloud_deploy.py vast --destroy {instance_id}")
        return 0 if instance_id or args.dry_run else 1
    
    if args.destroy:
        success = provider.destroy(args.destroy)
        if success:
            log(f"Instance {args.destroy} destroyed")
        return 0 if success else 1
    
    if args.ssh:
        ssh_cmd = provider.get_ssh_command(args.ssh)
        if ssh_cmd:
            print(ssh_cmd)
        return 0 if ssh_cmd else 1
    
    print("Specify an action: --search, --launch, --destroy, or --ssh")
    return 1


def handle_salad(args, training: TrainingConfig) -> int:
    """Handle SaladCloud commands."""
    config = SaladConfig(
        image=getattr(args, 'image', '') or '',
        project=args.project,
        gpu_class=args.gpu,
        results_bucket=getattr(args, 'results_bucket', '') or '',
    )
    
    provider = SaladCloudProvider(config, training, args.dry_run)
    
    if not provider.check_credentials():
        error("SaladCloud credentials not set")
        print("\nSetup:")
        print("  export SALAD_API_KEY=your_api_key")
        print("  export SALAD_ORG_NAME=your_organization")
        return 1
    
    if not HAS_REQUESTS:
        error("requests library required: pip install requests")
        return 1
    
    if args.list:
        groups = provider.list_groups()
        print(f"\n{Colors.BOLD}Container Groups in {args.project}:{Colors.NC}")
        for g in groups:
            status = g.get("current_state", {}).get("status", "unknown")
            print(f"  - {g.get('name')}: {status}")
        return 0
    
    if args.create:
        if not config.image:
            error("--image required for --create")
            return 1
        
        group_name = provider.create()
        if group_name:
            print(f"\n{Colors.GREEN}Container group created: {group_name}{Colors.NC}")
            print(f"\nStatus: python scripts/cloud_deploy.py salad --status {group_name}")
            print(f"Logs:   python scripts/cloud_deploy.py salad --logs {group_name}")
            print(f"Delete: python scripts/cloud_deploy.py salad --delete {group_name}")
        return 0 if group_name or args.dry_run else 1
    
    if args.status:
        status = provider.status(args.status)
        print(json.dumps(status, indent=2))
        return 0
    
    if args.logs:
        logs = provider.logs(args.logs)
        print(logs)
        return 0
    
    if args.stop:
        success = provider.stop(args.stop)
        return 0 if success else 1
    
    if args.delete:
        success = provider.delete(args.delete)
        if success:
            log(f"Container group {args.delete} deleted")
        return 0 if success else 1
    
    print("Specify an action: --list, --create, --status, --logs, --stop, or --delete")
    return 1


def handle_local(args, training: TrainingConfig) -> int:
    """Handle local execution."""
    runner = LocalRunner(training, args.dry_run)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
