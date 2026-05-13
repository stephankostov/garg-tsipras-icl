"""
check_wandb_ratelimit.py
Hits the W&B API and reads rate limit headers — no active experiment needed.
 
Usage:
    python check_wandb_ratelimit.py                      # uses WANDB_API_KEY env var
    python check_wandb_ratelimit.py --key YOUR_API_KEY   # explicit key
"""
 
import argparse
import os
import requests
 
WANDB_API = "https://api.wandb.ai/graphql"
 
QUERY = """
query Viewer {
  viewer {
    id
    username
    email
  }
}
"""
 
def check_rate_limit(api_key: str):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
 
    print(f"Querying {WANDB_API} ...")
    response = requests.post(WANDB_API, json={"query": QUERY}, headers=headers)
 
    print(f"\nHTTP Status: {response.status_code}")
 
    # Rate limit headers
    rl_headers = {
        k: v for k, v in response.headers.items()
        if "rate" in k.lower() or "retry" in k.lower() or "x-ratelimit" in k.lower()
    }
 
    if rl_headers:
        print("\nRate limit headers:")
        for k, v in rl_headers.items():
            print(f"  {k}: {v}")
    else:
        print("\nNo rate limit headers found in response.")
        print("(W&B may only include these on 429 responses)")
 
    if response.status_code == 429:
        reset = response.headers.get("RateLimit-Reset") or response.headers.get("X-RateLimit-Reset")
        retry = response.headers.get("Retry-After")
        print("\n⚠️  Rate limit exceeded (429).")
        if reset:
            print(f"   Reset in: {reset} seconds")
        if retry:
            print(f"   Retry after: {retry} seconds")
    elif response.status_code == 200:
        data = response.json()
        viewer = data.get("data", {}).get("viewer", {})
        print(f"\n✅ API key valid. Logged in as: {viewer.get('username')} ({viewer.get('email')})")
        print("   No active rate limit — you should be good to go.")
    elif response.status_code == 401:
        print("\n❌ Unauthorized — check your API key.")
    else:
        print(f"\nUnexpected response: {response.text[:300]}")
 
 
def main():
    parser = argparse.ArgumentParser(description="Check W&B API rate limit status.")
    parser.add_argument("--key", help="W&B API key (defaults to WANDB_API_KEY env var)")
    args = parser.parse_args()
 
    api_key = args.key or os.environ.get("WANDB_API_KEY")
 
    if not api_key:
        print("No API key found. Provide one with --key or set WANDB_API_KEY.")
        print("Get your key at: https://wandb.ai/authorize")
        return
 
    check_rate_limit(api_key)
 
 
if __name__ == "__main__":
    main()
