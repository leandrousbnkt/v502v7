#!/bin/bash
# ARQV30 Enhanced v2.0 - Startup Script

echo "🚀 Iniciando ARQV30 Enhanced v2.0..."

# Ativa ambiente virtual
source venv/bin/activate

# Executa aplicação
python run_production.py

echo "✅ ARQV30 Enhanced v2.0 iniciado"
