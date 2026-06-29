#!/usr/bin/env bash
# S-010: runtime smoke test for a built serving image.
#
# Starts the image under the AWS Lambda Runtime Interface Emulator (baked into the
# public.ecr.aws/lambda base image) and asserts a `GET /health` invocation returns
# statusCode 200 — catching a broken handler / entrypoint / import that a build-only
# check misses (the build can pass while the container fails to import the handler).
#
# Usage: .github/serving-smoke.sh <image> [host_port]
set -euo pipefail

IMAGE="${1:?usage: serving-smoke.sh <image> [host_port]}"
PORT="${2:-9000}"
NAME="kitchen-serve-smoke-$$"

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

docker run -d --name "$NAME" -p "${PORT}:8080" "$IMAGE" >/dev/null

# A minimal API Gateway HTTP-API (v2) event for GET /health — what Mangum expects.
EVENT='{"version":"2.0","routeKey":"GET /health","rawPath":"/health","rawQueryString":"",'
EVENT+='"headers":{"host":"localhost"},"requestContext":{"http":{"method":"GET","path":"/health",'
EVENT+='"protocol":"HTTP/1.1","sourceIp":"127.0.0.1"}},"isBase64Encoded":false}'

# --retry-connrefused waits for the RIE to come up; the endpoint is the Lambda invocations URL.
RESP="$(curl -s --fail-with-body --retry 20 --retry-delay 1 --retry-connrefused \
  -XPOST "http://localhost:${PORT}/2015-03-31/functions/function/invocations" -d "$EVENT")"

echo "smoke: /health → ${RESP}"

if ! python3 -c "import sys, json; sys.exit(0 if json.loads(sys.argv[1]).get('statusCode') == 200 else 1)" "$RESP"; then
  echo "SMOKE FAILED: GET /health did not return statusCode 200" >&2
  echo "--- container logs ---" >&2
  docker logs "$NAME" >&2 2>&1 || true
  exit 1
fi

echo "smoke: OK — handler imported, container served /health with statusCode 200"
