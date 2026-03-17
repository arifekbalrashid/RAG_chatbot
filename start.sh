#!/bin/bash

PORT_TO_USE=${PORT:-8501}

streamlit run app.py \
  --server.port=$PORT_TO_USE \
  --server.address=0.0.0.0 \
  --server.headless=true