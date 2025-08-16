"""
Client module for running in restricted environments.

This script should be executed on the client machine where the training and
production data reside.  It calculates model statistics and drift metrics
locally using the helper functions from `data_generator` and then uploads
only the aggregated metrics to the central server.  Sensitive data never
leaves the client environment.

Usage:

```
python client_module.py \
    --server-url http://localhost:8000 \
    --token <access_token> \
    --model-name "Credit Card Fraud Detection" \
    --training-data /path/to/train.csv \
    --production-data /path/to/prod.csv
```

For demonstration the training and production data paths are ignored, and
random metrics are generated instead.  Replace the dummy calculation in
`data_generator.calculate_metrics_from_paths` with your own logic when
integrating this script into a real system.
"""

import argparse
import json
import sys

import requests

from data_generator import calculate_metrics_from_paths


def main():
    parser = argparse.ArgumentParser(description="Upload local model metrics to Vision Server")
    parser.add_argument("--server-url", required=True, help="URL of the Vision server, e.g. http://localhost:8000")
    parser.add_argument("--token", required=True, help="Bearer token obtained from the server via /login")
    parser.add_argument("--model-name", required=True, help="Name of the model being registered")
    parser.add_argument("--training-data", required=True, help="Path to the training data")
    parser.add_argument("--production-data", required=True, help="Path to the production data")
    args = parser.parse_args()

    # Calculate metrics locally
    metrics = calculate_metrics_from_paths(args.training_data, args.production_data)

    payload = {
        "model_name": args.model_name,
        "metrics": metrics,
    }

    headers = {"Authorization": f"Bearer {args.token}", "Content-Type": "application/json"}
    try:
        resp = requests.post(f"{args.server_url}/model-metadata/submit", headers=headers, data=json.dumps(payload))
        if resp.status_code != 200:
            print(f"Failed to upload metrics: {resp.status_code} {resp.text}")
            sys.exit(1)
        print("Metrics uploaded successfully:")
        print(resp.json())
    except Exception as exc:
        print(f"Error communicating with server: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()