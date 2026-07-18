#!/bin/bash
#*******************************************************************************
# FORGE K3s Interactive Research Shell
# 
# Purpose:  Start an interactive FORGE shell session with persistent results storage.
# Usage:    ./forge-shell.sh
#
# Author:
#   Emily D. Carpenter
#   Anderson College of Business and Computing, Regis University
#   MSDS 696/S71: Data Science Practicum II
#   Dr. Douglas Hart, Dr. Kellen Sorauf
#   Practicum II, February-May 2026
#*******************************************************************************

export FORGE_USER=$(whoami)
export FORGE_UID=$(id -u)
mkdir -p ~/forge-results

echo "Starting FORGE container for ${FORGE_USER}..."
envsubst < manifests/forge-code.yml | kubectl apply -f -

echo "Waiting for pod to start..."
kubectl wait --for=condition=Ready pod/forge-${FORGE_USER} --timeout=120s

echo "Pod created. Connecting..."
kubectl exec -it forge-${FORGE_USER} -- su ${FORGE_USER}

echo ""
echo "Session ended. Your results are saved in ~/forge-results/"
echo "NOTE: Code changes made inside the container are NOT preserved."
echo "      Clone the repo and use a virtual environment for development work."
echo ""
echo "To reconnect: kubectl exec -it forge-${FORGE_USER} -- su ${FORGE_USER}"
echo "To remove:    kubectl delete pod forge-${FORGE_USER}"