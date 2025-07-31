
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v2.0 - Instalador de Extratores
Instala todas as dependências necessárias para extração robusta
"""

import subprocess
import sys
import logging

logger = logging.getLogger(__name__)

def install_package(package_name):
    """Instala um pacote Python"""
    try:
        logger.info(f"📦 Instalando {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        logger.info(f"✅ {package_name} instalado com sucesso")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Erro ao instalar {package_name}: {e}")
        return False

def main():
    """Instala todas as dependências de extração"""
    
    # Lista de pacotes necessários para extração robusta
    packages = [
        "trafilatura",
        "readability-lxml", 
        "newspaper3k",
        "beautifulsoup4",
        "lxml",
        "html5lib",
        "requests",
        "python-dateutil"
    ]
    
    print("🚀 ARQV30 - Instalando extratores robustos...")
    print(f"📋 Pacotes a instalar: {len(packages)}")
    
    failed_packages = []
    
    for package in packages:
        if not install_package(package):
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n❌ Falha ao instalar: {', '.join(failed_packages)}")
        print("⚠️ O sistema pode não funcionar corretamente")
        sys.exit(1)
    else:
        print("\n✅ Todos os extratores instalados com sucesso!")
        print("🎉 ARQV30 está pronto para extração robusta!")

if __name__ == "__main__":
    main()
