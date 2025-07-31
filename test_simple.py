#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v2.0 - Teste Simples
Teste básico das funcionalidades principais
"""

import sys
import os
import time
import requests
import json

# Adiciona o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Testa se todas as importações funcionam"""
    print("🔍 Testando importações...")
    
    try:
        import flask
        print("✓ Flask OK")
        
        import requests
        print("✓ Requests OK")
        
        from services.gemini_client import gemini_client
        print("✓ Gemini Client OK")
        
        from services.deep_search_service import deep_search_service
        print("✓ Deep Search Service OK")
        
        from services.attachment_service import attachment_service
        print("✓ Attachment Service OK")
        
        from database import get_db_connection
        print("✓ Database Connection OK")
        
        print("✅ Todas as importações funcionaram!")
        return True
        
    except Exception as e:
        print(f"❌ Erro na importação: {str(e)}")
        return False

def test_gemini_connection():
    """Testa conexão com Gemini"""
    print("\n🔍 Testando conexão com Gemini...")
    
    try:
        from services.gemini_client import gemini_client
        
        # Teste simples
        response = gemini_client.generate_content("Diga apenas 'OK' se você está funcionando.")
        
        if response and 'OK' in response.upper():
            print("✅ Gemini está funcionando!")
            return True
        else:
            print(f"⚠️ Gemini respondeu: {response}")
            return False
            
    except Exception as e:
        print(f"❌ Erro no Gemini: {str(e)}")
        return False

def test_database_connection():
    """Testa conexão com banco de dados"""
    print("\n🔍 Testando conexão com banco de dados...")
    
    try:
        from database import get_db_connection
        
        conn = get_db_connection()
        if conn:
            print("✅ Conexão com banco de dados OK!")
            conn.close()
            return True
        else:
            print("❌ Falha na conexão com banco de dados")
            return False
            
    except Exception as e:
        print(f"❌ Erro no banco de dados: {str(e)}")
        return False

def test_analysis_service():
    """Testa serviço de análise"""
    print("\n🔍 Testando serviço de análise...")
    
    try:
        from services.gemini_client import gemini_client
        
        # Dados de teste
        test_data = {
            'segmento': 'Produtos Digitais',
            'produto': 'Curso Online',
            'preco': 997.0,
            'publico': 'Empreendedores digitais'
        }
        
        # Testa geração de avatar
        avatar_prompt = f"""
        Crie um avatar detalhado para o segmento: {test_data['segmento']}
        Produto: {test_data['produto']}
        Preço: R$ {test_data['preco']}
        Público: {test_data['publico']}
        
        Responda apenas com: AVATAR_OK
        """
        
        response = gemini_client.generate_content(avatar_prompt)
        
        if response and 'AVATAR_OK' in response:
            print("✅ Serviço de análise funcionando!")
            return True
        else:
            print(f"⚠️ Resposta do serviço: {response}")
            return False
            
    except Exception as e:
        print(f"❌ Erro no serviço de análise: {str(e)}")
        return False

def test_file_processing():
    """Testa processamento de arquivos"""
    print("\n🔍 Testando processamento de arquivos...")
    
    try:
        from services.attachment_service import attachment_service
        
        # Cria arquivo de teste
        test_content = "Este é um teste de processamento de arquivo."
        test_file_path = "/tmp/test_file.txt"
        
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        # Testa processamento
        result = attachment_service.process_text_file(test_file_path)
        
        if result and test_content in result:
            print("✅ Processamento de arquivos OK!")
            os.remove(test_file_path)
            return True
        else:
            print(f"❌ Falha no processamento: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Erro no processamento de arquivos: {str(e)}")
        return False

def run_all_tests():
    """Executa todos os testes"""
    print("=" * 50)
    print("🚀 ARQV30 Enhanced v2.0 - Teste Completo")
    print("=" * 50)
    
    tests = [
        ("Importações", test_imports),
        ("Conexão Gemini", test_gemini_connection),
        ("Banco de Dados", test_database_connection),
        ("Serviço de Análise", test_analysis_service),
        ("Processamento de Arquivos", test_file_processing)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Erro crítico em {test_name}: {str(e)}")
            results.append((test_name, False))
    
    # Relatório final
    print("\n" + "=" * 50)
    print("📊 RELATÓRIO FINAL DOS TESTES")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print("-" * 50)
    print(f"Total: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("✅ Sistema está funcionando corretamente!")
    elif passed >= total * 0.7:
        print("⚠️ MAIORIA DOS TESTES PASSOU")
        print("🔧 Alguns ajustes podem ser necessários")
    else:
        print("❌ MUITOS TESTES FALHARAM")
        print("🚨 Sistema precisa de correções")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

