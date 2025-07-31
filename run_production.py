#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v2.0 - Production Runner
Script para executar em produção com Gunicorn
"""

import os
import sys
import subprocess
import signal
import time
import logging
from pathlib import Path

# Adiciona src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/production.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Verifica se todas as dependências estão instaladas"""
    logger.info("🔍 Verificando dependências...")
    
    try:
        import flask
        import gunicorn
        import requests
        import google.generativeai
        import supabase
        import pandas
        import PyPDF2
        logger.info("✅ Todas as dependências principais encontradas")
        return True
    except ImportError as e:
        logger.error(f"❌ Dependência ausente: {e}")
        logger.error("Execute: pip install -r requirements.txt")
        return False

def check_environment():
    """Verifica variáveis de ambiente críticas"""
    logger.info("🔍 Verificando configuração do ambiente...")
    
    critical_vars = [
        'SUPABASE_URL',
        'SUPABASE_ANON_KEY',
        'GEMINI_API_KEY'
    ]
    
    missing_vars = []
    for var in critical_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"❌ Variáveis de ambiente ausentes: {', '.join(missing_vars)}")
        logger.error("Configure o arquivo .env com as chaves necessárias")
        return False
    
    logger.info("✅ Configuração do ambiente OK")
    return True

def setup_directories():
    """Cria diretórios necessários"""
    logger.info("📁 Configurando diretórios...")
    
    directories = [
        'logs',
        'cache',
        'src/uploads',
        'src/static/images'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("✅ Diretórios configurados")

def run_with_gunicorn():
    """Executa aplicação com Gunicorn"""
    logger.info("🚀 Iniciando ARQV30 Enhanced v2.0 com Gunicorn...")
    
    # Configurações do Gunicorn
    workers = os.getenv('GUNICORN_WORKERS', '4')
    port = os.getenv('PORT', '5000')
    
    cmd = [
        'gunicorn',
        '--config', 'gunicorn.conf.py',
        '--workers', workers,
        '--bind', f'0.0.0.0:{port}',
        '--chdir', 'src',
        'run:create_app()'
    ]
    
    logger.info(f"🔧 Comando: {' '.join(cmd)}")
    
    try:
        # Executa Gunicorn
        process = subprocess.Popen(cmd)
        
        # Aguarda sinal de interrupção
        def signal_handler(signum, frame):
            logger.info(f"🛑 Recebido sinal {signum}, encerrando...")
            process.terminate()
            process.wait()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Aguarda processo
        process.wait()
        
    except FileNotFoundError:
        logger.error("❌ Gunicorn não encontrado. Instale com: pip install gunicorn")
        return False
    except Exception as e:
        logger.error(f"❌ Erro ao executar Gunicorn: {e}")
        return False
    
    return True

def run_with_flask():
    """Executa aplicação com Flask (desenvolvimento)"""
    logger.info("🔧 Iniciando ARQV30 Enhanced v2.0 com Flask (desenvolvimento)...")
    
    # Importa e executa aplicação Flask
    from run import main
    main()

def main():
    """Função principal"""
    logger.info("=" * 60)
    logger.info("🚀 ARQV30 Enhanced v2.0 - Production Runner")
    logger.info("=" * 60)
    
    # Verifica dependências
    if not check_dependencies():
        sys.exit(1)
    
    # Verifica ambiente
    if not check_environment():
        sys.exit(1)
    
    # Configura diretórios
    setup_directories()
    
    # Determina modo de execução
    flask_env = os.getenv('FLASK_ENV', 'production')
    
    if flask_env == 'production':
        logger.info("🏭 Modo: PRODUÇÃO")
        
        # Tenta usar Gunicorn primeiro
        try:
            import gunicorn
            success = run_with_gunicorn()
            if not success:
                logger.warning("⚠️ Gunicorn falhou, usando Flask...")
                run_with_flask()
        except ImportError:
            logger.warning("⚠️ Gunicorn não disponível, usando Flask...")
            run_with_flask()
    else:
        logger.info("🔧 Modo: DESENVOLVIMENTO")
        run_with_flask()

if __name__ == '__main__':
    main()