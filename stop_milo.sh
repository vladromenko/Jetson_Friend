#!/usr/bin/env bash
pkill -INT -f 'python.*milo.main' 2>/dev/null || true
sleep 1
pkill -TERM -f 'python.*milo.main' 2>/dev/null || true
echo "MILO stop signal sent"
