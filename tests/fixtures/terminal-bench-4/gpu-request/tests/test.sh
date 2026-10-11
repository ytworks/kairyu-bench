#!/bin/sh
if [ -f /app/done.txt ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
