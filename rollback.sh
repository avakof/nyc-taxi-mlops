#!/usr/bin/env bash
set -e
TAXI_DEPLOY_IMAGE=nyc-taxi-mlops:serving TAXI_RELEASE_PATH=models/final-serving TAXI_API_PORT=8001 \
  docker compose -p nyc-taxi-mlops -f compose.yaml -f compose.deploy.yaml \
  up --no-build --pull never --force-recreate --detach --wait --wait-timeout 60
