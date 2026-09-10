#!/usr/bin/env python3
"""Cleanup benchmark-created memories from Caura tenant."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=True)


def main():
    parser = argparse.ArgumentParser(description="Purge LongMemEval memories from Caura tenant")
    parser.add_argument("--apply", action="store_true", help="Perform the deletion (default is dry-run)")
    parser.add_argument("--prefix", default=os.environ.get("CAURA_AGENT_PREFIX", "lme"))
    args = parser.parse_args()

    api_key = os.environ.get("CAURA_API_KEY")
    tenant_id = os.environ.get("CAURA_TENANT_ID")
    base_url = os.environ.get("CAURA_BASE_URL", "https://caura.ai/api/v1").rstrip("/")

    if not api_key or not tenant_id:
        print("Error: CAURA_API_KEY and CAURA_TENANT_ID must be configured in .env", file=sys.stderr)
        sys.exit(1)

    client = httpx.Client(base_url=base_url, headers={"X-API-Key": api_key}, timeout=60.0)

    try:
        resp = client.get("/memories/stats")
        resp.raise_for_status()
        stats = resp.json()
    except Exception as exc:
        print(f"Error connecting to Caura: {exc}", file=sys.stderr)
        sys.exit(1)

    by_agent: dict[str, int] = stats.get("by_agent", {})
    targets = {a: n for a, n in by_agent.items() if a.startswith(f"{args.prefix}-")}

    if not targets:
        print(f"No agents found matching prefix '{args.prefix}-*' in tenant {tenant_id}.")
        return

    total = sum(targets.values())
    print(f"Found {len(targets)} benchmark agents with {total} memories in tenant {tenant_id}:")
    for agent, count in sorted(targets.items(), key=lambda x: -x[1])[:20]:
        print(f"  {agent}: {count}")
    if len(targets) > 20:
        print(f"  ... and {len(targets) - 20} more agents")

    if not args.apply:
        print("\nDry run. Run with --apply to permanently delete.")
        return

    print("\nApplying deletions...")
    deleted = 0
    for agent in sorted(targets):
        del_resp = client.request(
            "DELETE",
            "/memories",
            params={"tenant_id": tenant_id, "agent_id": agent, "fleet_id": agent},
        )
        if del_resp.status_code in (200, 204, 404):
            deleted += 1
        else:
            print(f"Failed deleting {agent}: {del_resp.status_code} - {del_resp.text}")

    print(f"Successfully cleaned up {deleted} benchmark agents.")


if __name__ == "__main__":
    main()
