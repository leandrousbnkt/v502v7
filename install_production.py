#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v2.0 - Production Installation Script
Script de instalação completo para produção
"""

import os
import sys
import subprocess
import logging
import platform
from pathlib import Path

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_python_version():
    """Verifica versão do Python"""
    logger.info("🐍 Verificando versão do Python...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("❌ Python 3.8+ é necessário")
        logger.error(f"Versão atual: {version.major}.{version.minor}.{version.micro}")
        return False
    
    logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro} OK")
    return True

def create_virtual_environment():
    """Cria ambiente virtual"""
    logger.info("🔧 Criando ambiente virtual...")
    
    try:
        subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
        logger.info("✅ Ambiente virtual criado")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Erro ao criar ambiente virtual: {e}")
        return False

def get_pip_command():
    """Retorna comando pip correto para o sistema"""
    system = platform.system().lower()
    
    if system == 'windows':
        return ['venv\\Scripts\\pip.exe']
    else:
        return ['venv/bin/pip']

def install_dependencies():
    """Instala dependências"""
    logger.info("📦 Instalando dependências...")
    
    pip_cmd = get_pip_command()
    
    try:
        # Atualiza pip
        subprocess.run(pip_cmd + ['install', '--upgrade', 'pip'], check=True)
        logger.info("✅ Pip atualizado")
        
        # Instala dependências principais
        subprocess.run(pip_cmd + ['install', '-r', 'requirements.txt'], check=True)
        logger.info("✅ Dependências principais instaladas")
        
        # Instala dependências de produção
        production_deps = [
            'gunicorn==21.2.0',
            'flask-compress==1.13',
            'redis==4.5.4',
            'psutil==5.9.5'
        ]
        
        subprocess.run(pip_cmd + ['install'] + production_deps, check=True)
        logger.info("✅ Dependências de produção instaladas")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Erro ao instalar dependências: {e}")
        return False

def setup_directories():
    """Cria estrutura de diretórios"""
    logger.info("📁 Criando estrutura de diretórios...")
    
    directories = [
        'logs',
        'cache',
        'src/uploads',
        'src/static/images',
        'backups',
        'tmp'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Diretório criado: {directory}")

def setup_environment_file():
    """Configura arquivo .env se não existir"""
    logger.info("⚙️ Verificando arquivo .env...")
    
    if not os.path.exists('.env'):
        logger.warning("⚠️ Arquivo .env não encontrado")
        logger.info("📝 Criando .env de exemplo...")
        
        env_template = """# ARQV30 Enhanced v2.0 - Production Environment
SECRET_KEY=your-secret-key-here
FLASK_ENV=production
CORS_ORIGINS=*
HOST=0.0.0.0
PORT=5000

# Supabase Configuration
SUPABASE_URL=your-supabase-url
SUPABASE_ANON_KEY=your-supabase-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-supabase-service-role-key
DATABASE_URL=your-database-url

# AI APIs
GEMINI_API_KEY=your-gemini-api-key
OPENAI_API_KEY=your-openai-api-key
HUGGINGFACE_API_KEY=your-huggingface-api-key
DEEPSEEK_API_KEY=your-deepseek-api-key

# Search APIs
GOOGLE_SEARCH_KEY=your-google-search-key
GOOGLE_CSE_ID=your-google-cse-id
SERPER_API_KEY=your-serper-api-key
JINA_API_KEY=your-jina-api-key

# Production Configuration
SEARCH_CACHE_ENABLED=true
SEARCH_CACHE_TTL=3600
CACHE_ENABLED=true
LOG_LEVEL=INFO
LOG_FILE_ENABLED=true
RATE_LIMIT_ENABLED=true
SECURE_HEADERS_ENABLED=true
GZIP_ENABLED=true
"""
        
        with open('.env.example', 'w', encoding='utf-8') as f:
            f.write(env_template)
        
        logger.info("✅ Arquivo .env.example criado")
        logger.warning("⚠️ Configure suas chaves de API no arquivo .env")
    else:
        logger.info("✅ Arquivo .env encontrado")

def test_installation():
    """Testa a instalação"""
    logger.info("🧪 Testando instalação...")
    
    try:
        # Testa importações críticas
        sys.path.insert(0, 'src')
        
        import flask
        import requests
        import google.generativeai
        import supabase
        import pandas
        import PyPDF2
        import gunicorn
        
        logger.info("✅ Todas as dependências importadas com sucesso")
        
        # Testa criação da aplicação
        from run import create_app
        app = create_app()
        
        logger.info("✅ Aplicação criada com sucesso")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Erro de importação: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Erro no teste: {e}")
        return False

def create_startup_scripts():
    """Cria scripts de inicialização"""
    logger.info("📜 Criando scripts de inicialização...")
    
    # Script para Linux/Mac
    startup_script_unix = """#!/bin/bash
# ARQV30 Enhanced v2.0 - Startup Script

echo "🚀 Iniciando ARQV30 Enhanced v2.0..."

# Ativa ambiente virtual
source venv/bin/activate

# Executa aplicação
python run_production.py

echo "✅ ARQV30 Enhanced v2.0 iniciado"
"""
    
    # Script para Windows
    startup_script_windows = """@echo off
REM ARQV30 Enhanced v2.0 - Startup Script

echo 🚀 Iniciando ARQV30 Enhanced v2.0...

REM Ativa ambiente virtual
call venv\\Scripts\\activate.bat

REM Executa aplicação
python run_production.py

echo ✅ ARQV30 Enhanced v2.0 iniciado
pause
"""
    
    # Cria scripts
    with open('start.sh', 'w', encoding='utf-8') as f:
        f.write(startup_script_unix)
    
    with open('start.bat', 'w', encoding='utf-8') as f:
        f.write(startup_script_windows)
    
    # Torna executável no Unix
    if platform.system() != 'Windows':
        os.chmod('start.sh', 0o755)
    
    logger.info("✅ Scripts de inicialização criados")

def main():
    """Função principal de instalação"""
    logger.info("=" * 60)
    logger.info("🚀 ARQV30 Enhanced v2.0 - Instalação para Produção")
    logger.info("=" * 60)
    
    # Verifica Python
    if not check_python_version():
        sys.exit(1)
    
    # Cria ambiente virtual
    if not os.path.exists('venv'):
        if not create_virtual_environment():
            sys.exit(1)
    else:
        logger.info("✅ Ambiente virtual já existe")
    
    # Instala dependências
    if not install_dependencies():
        sys.exit(1)
    
    # Configura diretórios
    setup_directories()
    
    # Configura .env
    setup_environment_file()
    
    # Cria scripts
    create_startup_scripts()
    
    # Testa instalação
    if not test_installation():
        logger.error("❌ Teste de instalação falhou")
        sys.exit(1)
    
    # Sucesso
    logger.info("=" * 60)
    logger.info("🎉 INSTALAÇÃO CONCLUÍDA COM SUCESSO!")
    logger.info("=" * 60)
    logger.info("")
    logger.info("📋 PRÓXIMOS PASSOS:")
    logger.info("1. Configure suas chaves de API no arquivo .env")
    logger.info("2. Execute: python run_production.py")
    logger.info("3. Acesse: http://localhost:5000")
    logger.info("")
    logger.info("🔧 Para desenvolvimento:")
    logger.info("   - Mude FLASK_ENV=development no .env")
    logger.info("   - Execute: python src/run.py")
    logger.info("")
    logger.info("🏭 Para produção:")
    logger.info("   - Mantenha FLASK_ENV=production no .env")
    logger.info("   - Execute: python run_production.py")
    logger.info("   - Ou use: ./start.sh (Linux/Mac) ou start.bat (Windows)")
    logger.info("")
    logger.info("✅ Sistema pronto para uso!")

if __name__ == '__main__':
    main()