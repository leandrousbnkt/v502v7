@echo off
REM ARQV30 Enhanced v2.0 - Startup Script

echo 🚀 Iniciando ARQV30 Enhanced v2.0...

REM Ativa ambiente virtual
call venv\Scripts\activate.bat

REM Executa aplicação
python run_production.py

echo ✅ ARQV30 Enhanced v2.0 iniciado
pause
