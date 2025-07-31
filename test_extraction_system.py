#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v2.0 - Teste do Sistema de Extração
Teste abrangente do novo sistema robusto de extração de conteúdo
"""

import sys
import os
import time
import logging
from typing import List, Dict, Any

# Adiciona src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_robust_extractor():
    """Testa o novo sistema robusto de extração"""
    
    print("=" * 80)
    print("🚀 TESTE DO SISTEMA ROBUSTO DE EXTRAÇÃO DE CONTEÚDO")
    print("=" * 80)
    
    try:
        from services.robust_content_extractor import robust_content_extractor
        from services.content_quality_validator import content_quality_validator
        from services.url_resolver import url_resolver
        
        print("✅ Módulos importados com sucesso")
        
        # URLs de teste variadas
        test_urls = [
            'https://g1.globo.com/tecnologia/',
            'https://www.estadao.com.br/economia/',
            'https://exame.com/negocios/',
            'https://valor.globo.com/empresas/',
            'https://canaltech.com.br/mercado/',
            'https://www.tecmundo.com.br/mercado/',
            'https://olhardigital.com.br/2024/',
            'https://www.infomoney.com.br/mercados/',
            'https://www.startse.com/noticia/',
            'https://revistapegn.globo.com/tecnologia/'
        ]
        
        print(f"\n🔍 Testando com {len(test_urls)} URLs variadas...")
        
        # Teste individual de cada URL
        results = []
        total_start_time = time.time()
        
        for i, url in enumerate(test_urls, 1):
            print(f"\n--- TESTE {i}/{len(test_urls)}: {url} ---")
            
            try:
                # Resolve URL primeiro
                resolved_url = url_resolver.resolve_url(url)
                if resolved_url != url:
                    print(f"🔗 URL resolvida: {resolved_url}")
                
                # Extrai conteúdo
                start_time = time.time()
                content = robust_content_extractor.extract_content(url)
                extraction_time = time.time() - start_time
                
                if content:
                    # Valida qualidade
                    validation = content_quality_validator.validate_content(content, url)
                    
                    result = {
                        'url': url,
                        'resolved_url': resolved_url,
                        'success': True,
                        'content_length': len(content),
                        'extraction_time': extraction_time,
                        'quality_score': validation['score'],
                        'quality_valid': validation['valid'],
                        'quality_reason': validation['reason']
                    }
                    
                    print(f"✅ SUCESSO: {len(content)} chars, qualidade {validation['score']:.1f}%, tempo {extraction_time:.2f}s")
                    if not validation['valid']:
                        print(f"⚠️ Qualidade baixa: {validation['reason']}")
                    
                else:
                    result = {
                        'url': url,
                        'resolved_url': resolved_url,
                        'success': False,
                        'error': 'Nenhum conteúdo extraído',
                        'extraction_time': extraction_time
                    }
                    print(f"❌ FALHA: Nenhum conteúdo extraído em {extraction_time:.2f}s")
                
                results.append(result)
                
            except Exception as e:
                result = {
                    'url': url,
                    'success': False,
                    'error': str(e),
                    'extraction_time': time.time() - start_time if 'start_time' in locals() else 0
                }
                results.append(result)
                print(f"❌ ERRO: {str(e)}")
            
            # Delay entre testes
            time.sleep(1)
        
        total_time = time.time() - total_start_time
        
        # Relatório final
        print("\n" + "=" * 80)
        print("📊 RELATÓRIO FINAL DO TESTE")
        print("=" * 80)
        
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        valid_quality = [r for r in successful if r.get('quality_valid', False)]
        
        print(f"Total de URLs testadas: {len(results)}")
        print(f"Extrações bem-sucedidas: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
        print(f"Extrações falharam: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
        print(f"Conteúdo de qualidade válida: {len(valid_quality)} ({len(valid_quality)/len(results)*100:.1f}%)")
        print(f"Tempo total: {total_time:.2f} segundos")
        print(f"Tempo médio por extração: {total_time/len(results):.2f} segundos")
        
        if successful:
            avg_content_length = sum(r['content_length'] for r in successful) / len(successful)
            avg_quality_score = sum(r['quality_score'] for r in successful) / len(successful)
            print(f"Tamanho médio do conteúdo: {avg_content_length:.0f} caracteres")
            print(f"Score médio de qualidade: {avg_quality_score:.1f}%")
        
        # Estatísticas dos extratores
        print(f"\n📈 ESTATÍSTICAS DOS EXTRATORES:")
        extractor_stats = robust_content_extractor.get_extractor_stats()
        
        for extractor_name, stats in extractor_stats.items():
            if extractor_name == 'global':
                continue
                
            if stats.get('available'):
                print(f"  {extractor_name}:")
                print(f"    - Usos: {stats['usage_count']}")
                print(f"    - Sucessos: {stats['success_count']}")
                print(f"    - Taxa de sucesso: {stats['success_rate']}%")
                print(f"    - Tempo médio: {stats['avg_response_time']:.2f}s")
            else:
                print(f"  {extractor_name}: INDISPONÍVEL ({stats.get('reason', 'Motivo desconhecido')})")
        
        # Detalhes das falhas
        if failed:
            print(f"\n❌ DETALHES DAS FALHAS:")
            for result in failed:
                print(f"  {result['url']}: {result.get('error', 'Erro desconhecido')}")
        
        # Avaliação final
        success_rate = len(successful) / len(results) * 100
        quality_rate = len(valid_quality) / len(results) * 100
        
        print(f"\n🎯 AVALIAÇÃO FINAL:")
        
        if success_rate >= 80 and quality_rate >= 60:
            print("🎉 EXCELENTE: Sistema funcionando perfeitamente!")
            print("✅ Taxa de sucesso alta e boa qualidade de conteúdo")
            return True
        elif success_rate >= 60 and quality_rate >= 40:
            print("👍 BOM: Sistema funcionando adequadamente")
            print("⚠️ Algumas melhorias podem ser feitas")
            return True
        elif success_rate >= 40:
            print("⚠️ REGULAR: Sistema precisa de ajustes")
            print("🔧 Considere configurar mais extratores")
            return False
        else:
            print("❌ RUIM: Sistema precisa de correções urgentes")
            print("🚨 Verifique configurações e dependências")
            return False
        
    except ImportError as e:
        print(f"❌ ERRO DE IMPORTAÇÃO: {e}")
        print("🔧 Instale as dependências: pip install readability-lxml trafilatura newspaper3k")
        return False
    except Exception as e:
        print(f"❌ ERRO CRÍTICO: {e}")
        return False

def test_url_resolver():
    """Testa o resolvedor de URLs"""
    
    print("\n" + "=" * 60)
    print("🔗 TESTE DO RESOLVEDOR DE URLS")
    print("=" * 60)
    
    try:
        from services.url_resolver import url_resolver
        
        # URLs de teste com redirecionamentos
        test_urls = [
            'https://www.bing.com/ck/a?!&&p=test&u=a1aHR0cHM6Ly9nMS5nbG9iby5jb20%3d',  # Bing redirect
            'https://www.google.com/url?q=https://exame.com&sa=U',  # Google redirect
            'https://bit.ly/3example',  # URL encurtada (exemplo)
        ]
        
        for url in test_urls:
            try:
                resolved = url_resolver.resolve_url(url)
                print(f"✅ {url[:50]}... → {resolved[:50]}...")
            except Exception as e:
                print(f"❌ {url[:50]}... → ERRO: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste de URL resolver: {e}")
        return False

def test_content_validator():
    """Testa o validador de qualidade"""
    
    print("\n" + "=" * 60)
    print("✅ TESTE DO VALIDADOR DE QUALIDADE")
    print("=" * 60)
    
    try:
        from services.content_quality_validator import content_quality_validator
        
        # Conteúdos de teste
        test_contents = [
            {
                'name': 'Conteúdo de qualidade',
                'content': 'Este é um artigo sobre análise de mercado no Brasil. O mercado brasileiro de tecnologia está crescendo rapidamente, com investimentos de R$ 2,5 bilhões em 2024. As principais tendências incluem inteligência artificial, automação e transformação digital. Empresas como Magazine Luiza, Nubank e Stone estão liderando a inovação. Os consumidores brasileiros estão cada vez mais digitais, com 87% usando smartphones para compras online. O e-commerce cresceu 27% no último ano, atingindo R$ 185 bilhões em faturamento. As oportunidades incluem fintech, healthtech e edtech.' * 3
            },
            {
                'name': 'Página de erro',
                'content': 'Página não encontrada. Erro 404. O recurso solicitado não existe.'
            },
            {
                'name': 'Conteúdo muito curto',
                'content': 'Texto muito curto.'
            },
            {
                'name': 'Só navegação',
                'content': 'Home Sobre Contato Produtos Serviços Blog Login Cadastro Menu Navegação'
            }
        ]
        
        for test in test_contents:
            validation = content_quality_validator.validate_content(test['content'])
            status = "✅ VÁLIDO" if validation['valid'] else "❌ INVÁLIDO"
            print(f"{status} {test['name']}: {validation['score']:.1f}% - {validation['reason']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste de validador: {e}")
        return False

def run_comprehensive_test():
    """Executa teste abrangente do sistema"""
    
    print("🚀 INICIANDO TESTE ABRANGENTE DO SISTEMA DE EXTRAÇÃO")
    print("=" * 80)
    
    tests = [
        ("Resolvedor de URLs", test_url_resolver),
        ("Validador de Qualidade", test_content_validator),
        ("Sistema Robusto de Extração", test_robust_extractor)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Executando: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Erro crítico em {test_name}: {e}")
            results.append((test_name, False))
    
    # Relatório final
    print("\n" + "=" * 80)
    print("🏁 RELATÓRIO FINAL DOS TESTES")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{test_name:.<40} {status}")
    
    print(f"\nTotal: {passed}/{total} testes passaram ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 TODOS OS TESTES PASSARAM!")
        print("✅ Sistema de extração está funcionando perfeitamente!")
        print("🚀 Pronto para uso em produção!")
    elif passed >= total * 0.7:
        print("\n👍 MAIORIA DOS TESTES PASSOU!")
        print("⚠️ Sistema funcional com algumas limitações")
        print("🔧 Considere instalar dependências opcionais")
    else:
        print("\n❌ MUITOS TESTES FALHARAM!")
        print("🚨 Sistema precisa de correções antes do uso")
        print("🔧 Verifique instalação das dependências")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_test()
    
    if success:
        print("\n🎯 PRÓXIMOS PASSOS:")
        print("1. ✅ Sistema testado e funcionando")
        print("2. 🚀 Execute uma análise completa para testar integração")
        print("3. 📊 Monitore estatísticas dos extratores em produção")
        print("4. 🔧 Ajuste configurações conforme necessário")
    else:
        print("\n🔧 AÇÕES NECESSÁRIAS:")
        print("1. ❌ Instale dependências faltantes")
        print("2. 🔧 Verifique configurações de rede")
        print("3. 🧪 Execute testes individuais para debug")
        print("4. 📞 Consulte logs para detalhes dos erros")
    
    sys.exit(0 if success else 1)