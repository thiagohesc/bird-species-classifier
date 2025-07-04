#!/bin/bash

echo "Verificando se é necessário treinar o modelo..."
python -m services.classifier

echo "Subindo FastAPI..."
exec uvicorn main:app --host 0.0.0.0 --port 8000