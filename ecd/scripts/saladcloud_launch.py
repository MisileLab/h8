#!/usr/bin/env python3
"""
SaladCloud Programmatic Launcher for ECD

This script uses the SaladCloud API to:
1. Create a container group with GPU requirements
2. Deploy the ECD training container
3. Monitor job status
4. Retrieve results

Prerequisites:
    pip install requests
    
    Set environment variables:
        SALAD_API_KEY=your_api_key
        SALAD_ORG_NAME=your_organization

Usage:
    python scripts/saladcloud_launch.py --help
    python scripts/saladcloud_launch.py --create --image your-registry/ecd:latest
    python scripts/saladcloud_launch.py --status --group-id abc123
    python scripts/saladcloud_launch.py --delete --group-id abc123
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional
import requests


# SaladCloud API base URL
SALAD_API_BASE = "https://api.salad.com/api/public"


@dataclass
class ContainerGroupConfig:
    """Configuration for SaladCloud container group."""
    name: str = "ecd-track-b"
    display_name: str = "ECD Track-B Training"
    image: str = ""
    replicas: int = 1
    
    # Resource requirements
    cpu: int = 4
    memory: int = 32768  # MB
    gpu_classes: list = field(default_factory=lambda: ["rtx4090", "rtx3090", "a100"])
    
    # Environment variables
    env_vars: dict = field(default_factory=dict)
    
    # Networking
    container_port: int = 8080
    
    # Restart policy
    restart_policy: str = "on_failure"
    max_restarts: int = 3


class SaladCloudClient:
    """Client for SaladCloud API."""
    
    def __init__(self, api_key: str, org_name: str):
        self.api_key = api_key
        self.org_name = org_name
        self.headers = {
            "Salad-Api-Key": api_key,
            "Content-Type": "application/json"
        }
    
    def _url(self, path: str) -> str:
        return f"{SALAD_API_BASE}/organizations/{self.org_name}/{path}"
    
    def _request(self, method: str, path: str, data: Optional[dict] = None) -> dict:
        url = self._url(path)
        response = requests.request(method, url, headers=self.headers, json=data)
        
        if response.status_code >= 400:
            print(f"API Error: {response.status_code}", file=sys.stderr)
            print(response.text, file=sys.stderr)
            return {"error": response.text, "status_code": response.status_code}
        
        try:
            return response.json()
        except json.JSONDecodeError:
            return {"raw": response.text}
    
    def list_projects(self) -> list:
        """List all projects."""
        result = self._request("GET", "projects")
        return result.get("items", [])
    
    def create_project(self, name: str, display_name: str = "") -> dict:
        """Create a new project."""
        data = {
            "name": name,
            "display_name": display_name or name
        }
        return self._request("POST", "projects", data)
    
    def list_container_groups(self, project_name: str) -> list:
        """List container groups in a project."""
        result = self._request("GET", f"projects/{project_name}/containers")
        return result.get("items", [])
    
    def create_container_group(self, project_name: str, config: ContainerGroupConfig) -> dict:
        """Create a new container group."""
        
        # Build container spec
        container = {
            "image": config.image,
            "resources": {
                "cpu": config.cpu,
                "memory": config.memory,
                "gpu_classes": config.gpu_classes
            },
            "environment_variables": config.env_vars,
            "logging": {
                "type": "stdout"
            }
        }
        
        # Build networking if port specified
        networking = None
        if config.container_port:
            networking = {
                "protocol": "http",
                "port": config.container_port,
                "auth": False
            }
        
        data = {
            "name": config.name,
            "display_name": config.display_name,
            "container": container,
            "replicas": config.replicas,
            "restart_policy": config.restart_policy,
        }
        
        if networking:
            data["networking"] = networking
        
        return self._request("POST", f"projects/{project_name}/containers", data)
    
    def get_container_group(self, project_name: str, group_name: str) -> dict:
        """Get container group details."""
        return self._request("GET", f"projects/{project_name}/containers/{group_name}")
    
    def start_container_group(self, project_name: str, group_name: str) -> dict:
        """Start a container group."""
        return self._request("POST", f"projects/{project_name}/containers/{group_name}/start")
    
    def stop_container_group(self, project_name: str, group_name: str) -> dict:
        """Stop a container group."""
        return self._request("POST", f"projects/{project_name}/containers/{group_name}/stop")
    
    def delete_container_group(self, project_name: str, group_name: str) -> dict:
        """Delete a container group."""
        return self._request("DELETE", f"projects/{project_name}/containers/{group_name}")
    
    def get_instances(self, project_name: str, group_name: str) -> list:
        """Get instances in a container group."""
        result = self._request("GET", f"projects/{project_name}/containers/{group_name}/instances")
        return result.get("instances", [])
    
    def get_logs(self, project_name: str, group_name: str, instance_id: str) -> str:
        """Get logs from an instance."""
        result = self._request("GET", 
            f"projects/{project_name}/containers/{group_name}/instances/{instance_id}/logs")
        return result.get("logs", "")


def create_ecd_container_group(
    client: SaladCloudClient,
    project_name: str,
    image: str,
    steps: int = 20000,
    seeds: str = "0,1,2",
    quick_mode: bool = False,
    results_bucket: str = "",
    gpu_class: str = "rtx4090",
    dry_run: bool = False
) -> Optional[dict]:
    """Create ECD training container group."""
    
    # Build environment variables
    env_vars = {
        "TRACK_B_STEPS": str(steps),
        "TRACK_B_SEEDS": seeds,
        "QUICK_MODE": "true" if quick_mode else "false",
        "UPLOAD_RESULTS": "true" if results_bucket else "false",
    }
    
    if results_bucket:
        env_vars["RESULTS_BUCKET"] = results_bucket
    
    # Map GPU class to SaladCloud format
    gpu_classes_map = {
        "rtx4090": ["rtx4090_24gb"],
        "rtx3090": ["rtx3090_24gb"],
        "a100": ["a100_40gb", "a100_80gb"],
        "any_24gb": ["rtx4090_24gb", "rtx3090_24gb"],
        "any": ["rtx4090_24gb", "rtx3090_24gb", "rtx3080_10gb", "a100_40gb"]
    }
    
    gpu_classes = gpu_classes_map.get(gpu_class, [gpu_class])
    
    config = ContainerGroupConfig(
        name=f"ecd-track-b-{int(time.time())}",
        display_name=f"ECD Track-B ({steps} steps)",
        image=image,
        replicas=1,
        cpu=4,
        memory=32768,
        gpu_classes=gpu_classes,
        env_vars=env_vars,
        container_port=8080,
    )
    
    print(f"\nContainer Group Configuration:")
    print(f"  Name: {config.name}")
    print(f"  Image: {config.image}")
    print(f"  GPU Classes: {config.gpu_classes}")
    print(f"  Environment:")
    for k, v in env_vars.items():
        print(f"    {k}={v}")
    
    if dry_run:
        print("\n[DRY RUN] Would create container group")
        return None
    
    # Ensure project exists
    projects = client.list_projects()
    if not any(p.get("name") == project_name for p in projects):
        print(f"Creating project: {project_name}")
        client.create_project(project_name, "ECD Experiments")
    
    # Create container group
    print(f"\nCreating container group...")
    result = client.create_container_group(project_name, config)
    
    if "error" in result:
        print(f"Failed to create container group: {result}", file=sys.stderr)
        return None
    
    print(f"Container group created: {result.get('name', config.name)}")
    
    # Start the container group
    print("Starting container group...")
    client.start_container_group(project_name, config.name)
    
    return result


def monitor_container_group(
    client: SaladCloudClient,
    project_name: str,
    group_name: str,
    timeout: int = 3600,
    poll_interval: int = 30
) -> bool:
    """Monitor container group until completion or timeout."""
    
    print(f"\nMonitoring container group: {group_name}")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        group = client.get_container_group(project_name, group_name)
        
        if "error" in group:
            print(f"Error getting status: {group}")
            return False
        
        status = group.get("current_state", {}).get("status", "unknown")
        running = group.get("current_state", {}).get("running_count", 0)
        
        print(f"  [{time.strftime('%H:%M:%S')}] Status: {status}, Running: {running}")
        
        if status == "running" and running > 0:
            # Get instance logs
            instances = client.get_instances(project_name, group_name)
            for inst in instances[:1]:  # Just first instance
                inst_id = inst.get("machine_id", "")
                if inst_id:
                    logs = client.get_logs(project_name, group_name, inst_id)
                    if logs:
                        # Show last few lines
                        lines = logs.strip().split("\n")[-5:]
                        print("  Recent logs:")
                        for line in lines:
                            print(f"    {line}")
        
        if status in ["stopped", "failed"]:
            print(f"Container group {status}")
            return status == "stopped"
        
        time.sleep(poll_interval)
    
    print("Timeout waiting for completion")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="SaladCloud launcher for ECD experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create and start training
  python scripts/saladcloud_launch.py --create --image myregistry/ecd:latest
  
  # Quick test run
  python scripts/saladcloud_launch.py --create --image myregistry/ecd:latest --quick
  
  # Check status
  python scripts/saladcloud_launch.py --status --group-name ecd-track-b-123
  
  # Delete container group
  python scripts/saladcloud_launch.py --delete --group-name ecd-track-b-123
  
  # Dry run
  python scripts/saladcloud_launch.py --create --image myregistry/ecd:latest --dry-run
"""
    )
    
    # Actions
    parser.add_argument("--create", action="store_true", help="Create container group")
    parser.add_argument("--status", action="store_true", help="Check status")
    parser.add_argument("--logs", action="store_true", help="Get logs")
    parser.add_argument("--stop", action="store_true", help="Stop container group")
    parser.add_argument("--delete", action="store_true", help="Delete container group")
    parser.add_argument("--list", action="store_true", help="List container groups")
    parser.add_argument("--monitor", action="store_true", help="Monitor until completion")
    
    # Container config
    parser.add_argument("--image", type=str, help="Container image URL")
    parser.add_argument("--project", type=str, default="ecd-experiments", help="Project name")
    parser.add_argument("--group-name", type=str, help="Container group name")
    
    # Training config
    parser.add_argument("--steps", type=int, default=20000, help="Training steps")
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Seeds")
    parser.add_argument("--quick", action="store_true", help="Quick mode")
    parser.add_argument("--gpu", type=str, default="rtx4090", 
                        help="GPU class (rtx4090, rtx3090, a100, any_24gb, any)")
    
    # Results
    parser.add_argument("--results-bucket", type=str, help="S3/GCS bucket for results")
    
    # Other
    parser.add_argument("--dry-run", action="store_true", help="Dry run")
    parser.add_argument("--timeout", type=int, default=3600, help="Monitor timeout")
    
    args = parser.parse_args()
    
    # Get credentials from environment
    api_key = os.environ.get("SALAD_API_KEY")
    org_name = os.environ.get("SALAD_ORG_NAME")
    
    if not api_key or not org_name:
        print("Error: SALAD_API_KEY and SALAD_ORG_NAME environment variables required",
              file=sys.stderr)
        print("\nSet them with:", file=sys.stderr)
        print("  export SALAD_API_KEY=your_api_key", file=sys.stderr)
        print("  export SALAD_ORG_NAME=your_organization", file=sys.stderr)
        sys.exit(1)
    
    client = SaladCloudClient(api_key, org_name)
    
    # Handle actions
    if args.list:
        groups = client.list_container_groups(args.project)
        print(f"Container groups in {args.project}:")
        for g in groups:
            status = g.get("current_state", {}).get("status", "unknown")
            print(f"  - {g.get('name')}: {status}")
        sys.exit(0)
    
    if args.create:
        if not args.image:
            print("Error: --image required for --create", file=sys.stderr)
            sys.exit(1)
        
        result = create_ecd_container_group(
            client=client,
            project_name=args.project,
            image=args.image,
            steps=args.steps,
            seeds=args.seeds,
            quick_mode=args.quick,
            results_bucket=args.results_bucket or "",
            gpu_class=args.gpu,
            dry_run=args.dry_run
        )
        
        if result and args.monitor and not args.dry_run:
            group_name = result.get("name")
            if group_name:
                monitor_container_group(client, args.project, group_name, args.timeout)
        
        sys.exit(0)
    
    if args.status:
        if not args.group_name:
            print("Error: --group-name required for --status", file=sys.stderr)
            sys.exit(1)
        
        group = client.get_container_group(args.project, args.group_name)
        print(json.dumps(group, indent=2))
        sys.exit(0)
    
    if args.logs:
        if not args.group_name:
            print("Error: --group-name required for --logs", file=sys.stderr)
            sys.exit(1)
        
        instances = client.get_instances(args.project, args.group_name)
        for inst in instances:
            inst_id = inst.get("machine_id")
            if inst_id:
                print(f"=== Logs from {inst_id} ===")
                logs = client.get_logs(args.project, args.group_name, inst_id)
                print(logs)
        sys.exit(0)
    
    if args.stop:
        if not args.group_name:
            print("Error: --group-name required for --stop", file=sys.stderr)
            sys.exit(1)
        
        result = client.stop_container_group(args.project, args.group_name)
        print(f"Stop result: {result}")
        sys.exit(0)
    
    if args.delete:
        if not args.group_name:
            print("Error: --group-name required for --delete", file=sys.stderr)
            sys.exit(1)
        
        result = client.delete_container_group(args.project, args.group_name)
        print(f"Delete result: {result}")
        sys.exit(0)
    
    if args.monitor:
        if not args.group_name:
            print("Error: --group-name required for --monitor", file=sys.stderr)
            sys.exit(1)
        
        success = monitor_container_group(
            client, args.project, args.group_name, args.timeout
        )
        sys.exit(0 if success else 1)
    
    # No action specified
    parser.print_help()


if __name__ == "__main__":
    main()
