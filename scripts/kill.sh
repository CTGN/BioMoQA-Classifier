#!/usr/bin/env bash

# List processes matching 'uv run'
ps -ef | grep '[u]v run'

# Kill processes running 'uv run'
ps -ef | grep '[u]v run' | awk '{print $2}' | xargs -r kill