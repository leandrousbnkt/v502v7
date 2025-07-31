#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v2.0 - Rotas de Análise
Endpoints para análise de mercado ultra-detalhada
"""

import os
import logging
import time
import json
from datetime import datetime
from flask import Blueprint, request, jsonify, session
from services.enhanced_analysis_engine import enhanced_analysis_engine
from services.ultra_detailed_analysis_engine import ultra_detailed_analysis_engine
from services.ai_manager import ai_manager
from services.production_search_manager import production_search_manager
from services.robust_content_extractor import robust_content_extractor
from services.content_quality_validator import content_quality_validator
from services.attachment_service import attachment_service
from database import db_manager
from routes.progress import get_progress_tracker, update_analysis_progress

logger = logging.getLogger(__name__)

# Cria blueprint
analysis_bp = Blueprint('analysis', __name__)

@analysis_bp.route('/analyze', methods=['POST'])
def analyze_market():
    """Endpoint principal para análise de mercado"""
    
    try:
        start_time = time.time()
        logger.info("🚀 Iniciando análise de mercado ultra-detalhada")
        
        # Coleta dados da requisição
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'Dados não fornecidos',
                'message': 'Envie os dados da análise no corpo da requisição'
            }), 400
        
        # Validação básica
        if not data.get('segmento'):
            return jsonify({
                'error': 'Segmento obrigatório',
                'message': 'O campo "segmento" é obrigatório para análise'
            }), 400
        
        # Adiciona session_id se não fornecido
        if not data.get('session_id'):
            data['session_id'] = f"session_{int(time.time())}_{os.urandom(4).hex()}"
        
        # Inicia rastreamento de progresso
        session_id = data['session_id']
        progress_tracker = get_progress_tracker(session_id)
        
        # Função de callback para progresso
        def progress_callback(step: int, message: str, details: str = None):
            update_analysis_progress(session_id, step, message, details)
        
        # Log dos dados recebidos
        logger.info(f"📊 Dados recebidos: Segmento={data.get('segmento')}, Produto={data.get('produto')}")
        
        # Prepara query de pesquisa se não fornecida
        if not data.get('query'):
            segmento = data.get('segmento', '')
            produto = data.get('produto', '')
            if produto:
                data['query'] = f"mercado {segmento} {produto} Brasil tendências oportunidades 2024"
            else:
                data['query'] = f"análise mercado {segmento} Brasil dados estatísticas crescimento"
        
        logger.info(f"🔍 Query de pesquisa: {data['query']}")
        
        # Executa análise GIGANTE ultra-detalhada
        logger.info("🚀 Executando análise GIGANTE ultra-detalhada...")
        try:
            analysis_result = ultra_detailed_analysis_engine.generate_gigantic_analysis(
                data,
                session_id=session_id,
                progress_callback=progress_callback
            )
        except Exception as e:
            logger.error(f"❌ Análise GIGANTE falhou: {str(e)}")
            
            # Verifica se é erro de IA e sugere soluções
            if "IA FALHOU" in str(e):
                error_message = "Todos os provedores de IA estão temporariamente indisponíveis"
                recommendation = "Aguarde alguns minutos e tente novamente. Os serviços de IA podem estar sobrecarregados."
                
                # Verifica quais provedores estão configurados
                ai_status = ai_manager.get_provider_status()
                available_providers = [name for name, status in ai_status.items() if status.get('available', False)]
                
                if not available_providers:
                    recommendation = "Configure pelo menos uma API de IA (Gemini, Groq, OpenAI ou HuggingFace)"
                
                return jsonify({
                    'error': error_message,
                    'message': str(e),
                    'timestamp': datetime.now().isoformat(),
                    'recommendation': recommendation,
                    'available_providers': available_providers,
                    'retry_suggested': True,
                    'fallback_available': False
                }), 503
            
            return jsonify({
                'error': 'Análise falhou por dados insuficientes',
                'message': str(e),
                'timestamp': datetime.now().isoformat(),
                'recommendation': 'Configure todas as APIs necessárias e verifique conectividade',
                'required_apis': [
                    'GEMINI_API_KEY ou OPENAI_API_KEY (obrigatório)',
                    'GROQ_API_KEY (recomendado para backup)',
                    'GOOGLE_SEARCH_KEY + GOOGLE_CSE_ID (recomendado)',
                    'JINA_API_KEY (recomendado)',
                    'SERPER_API_KEY (opcional)'
                ],
                'fallback_available': False
            }), 500
        
        # Verifica se a análise foi bem-sucedida
        if not analysis_result or not isinstance(analysis_result, dict):
            return jsonify({
                'error': 'Análise retornou resultado inválido',
                'message': 'Sistema não conseguiu gerar análise válida',
                'timestamp': datetime.now().isoformat(),
                'recommendation': 'Verifique configuração das APIs e tente novamente'
            }), 500
        
        # Marca progresso como completo
        progress_tracker.complete()
        
        # Salva no banco de dados
        try:
            logger.info("💾 Salvando análise no banco de dados...")
            db_record = db_manager.create_analysis({
                'segmento': data.get('segmento'),
                'produto': data.get('produto'),
                'descricao': data.get('dados_adicionais'),
                'preco': data.get('preco'),
                'publico': data.get('publico'),
                'concorrentes': data.get('concorrentes'),
                'dados_adicionais': data.get('dados_adicionais'),
                'objetivo_receita': data.get('objetivo_receita'),
                'orcamento_marketing': data.get('orcamento_marketing'),
                'prazo_lancamento': data.get('prazo_lancamento'),
                'status': 'completed',
                'avatar_data': analysis_result.get('avatar_ultra_detalhado'),
                'positioning_data': analysis_result.get('escopo'),
                'competition_data': analysis_result.get('analise_concorrencia_detalhada'),
                'marketing_data': analysis_result.get('estrategia_palavras_chave'),
                'metrics_data': analysis_result.get('metricas_performance_detalhadas'),
                'funnel_data': analysis_result.get('funil_vendas_detalhado'),
                'market_intelligence': analysis_result.get('pesquisa_web_massiva'),
                'action_plan': analysis_result.get('plano_acao_detalhado'),
                'comprehensive_analysis': analysis_result
            })
            
            if db_record:
                analysis_result['database_id'] = db_record['id']
                logger.info(f"✅ Análise salva com ID: {db_record['id']}")
            else:
                logger.warning("⚠️ Falha ao salvar no banco, mas análise continua")
                
        except Exception as e:
            logger.error(f"❌ Erro ao salvar no banco: {str(e)}")
            # Não falha a análise por erro no banco
            analysis_result['database_warning'] = f"Falha ao salvar no banco: {str(e)}"
        
        # Calcula tempo de processamento
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Adiciona metadados finais
        if 'metadata' not in analysis_result:
            analysis_result['metadata'] = {}
        
        analysis_result['metadata'].update({
            'processing_time_seconds': processing_time,
            'processing_time_formatted': f"{int(processing_time // 60)}m {int(processing_time % 60)}s",
            'request_timestamp': datetime.now().isoformat(),
            'session_id': data.get('session_id'),
            'input_data': {
                'segmento': data.get('segmento'),
                'produto': data.get('produto'),
                'query': data.get('query')
            }
        })
        
        logger.info(f"✅ Análise concluída em {processing_time:.2f} segundos")
        
        return jsonify(analysis_result)
        
    except Exception as e:
        logger.error(f"❌ Erro crítico na análise: {str(e)}", exc_info=True)
        
        # Remove progresso em caso de erro
        if session_id in get_progress_tracker.__globals__.get('progress_sessions', {}):
            del get_progress_tracker.__globals__['progress_sessions'][session_id]
        
        return jsonify({
            'error': 'Erro na análise',
            'message': str(e),
            'timestamp': datetime.now().isoformat(),
            'fallback_available': False,
            'recommendation': 'Configure todas as APIs necessárias antes de tentar novamente',
            'debug_info': {
                'session_id': session_id,
                'input_data': {
                    'segmento': data.get('segmento'),
                    'produto': data.get('produto'),
                    'query': data.get('query')
                }
            }
        }), 500

@analysis_bp.route('/status', methods=['GET'])
def get_analysis_status():
    """Retorna status dos sistemas de análise"""
    
    try:
        # Status dos provedores de IA
        ai_status = ai_manager.get_provider_status()
        
        # Status dos provedores de busca
        search_status = production_search_manager.get_provider_status()
        
        # Status do banco de dados
        db_status = db_manager.test_connection()
        
        # Status geral
        total_ai_available = len([p for p in ai_status.values() if p['available']])
        total_search_available = len([p for p in search_status.values() if p['available']])
        
        overall_status = "healthy" if (total_ai_available > 0 and total_search_available > 0 and db_status) else "degraded"
        
        return jsonify({
            'status': overall_status,
            'timestamp': datetime.now().isoformat(),
            'systems': {
                'ai_providers': {
                    'status': 'healthy' if total_ai_available > 0 else 'error',
                    'available_count': total_ai_available,
                    'total_count': len(ai_status),
                    'providers': ai_status
                },
                'search_providers': {
                    'status': 'healthy' if total_search_available > 0 else 'error',
                    'available_count': total_search_available,
                    'total_count': len(search_status),
                    'providers': search_status
                },
                'database': {
                    'status': 'healthy' if db_status else 'error',
                    'connected': db_status
                },
                'content_extraction': {
                    'status': 'healthy',
                    'available': True
                }
            },
            'capabilities': {
                'multi_ai_fallback': total_ai_available > 1,
                'multi_search_fallback': total_search_available > 1,
                'real_time_search': total_search_available > 0,
                'content_extraction': True,
                'database_storage': db_status
            }
        })
        
    except Exception as e:
        logger.error(f"Erro ao obter status: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@analysis_bp.route('/reset_providers', methods=['POST'])
def reset_providers():
    """Reset contadores de erro dos provedores"""
    
    try:
        data = request.get_json() or {}
        provider_type = data.get('type')  # 'ai' ou 'search'
        provider_name = data.get('provider')  # nome específico do provedor
        
        if provider_type == 'ai':
            ai_manager.reset_provider_errors(provider_name)
            message = f"Reset erros do provedor de IA: {provider_name}" if provider_name else "Reset erros de todos os provedores de IA"
        elif provider_type == 'search':
            production_search_manager.reset_provider_errors(provider_name)
            message = f"Reset erros do provedor de busca: {provider_name}" if provider_name else "Reset erros de todos os provedores de busca"
        else:
            # Reset todos
            ai_manager.reset_provider_errors()
            production_search_manager.reset_provider_errors()
            message = "Reset erros de todos os provedores"
        
        logger.info(f"🔄 {message}")
        
        return jsonify({
            'success': True,
            'message': message,
            'ai_status': ai_manager.get_provider_status(),
            'search_status': production_search_manager.get_provider_status(),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erro ao resetar provedores: {str(e)}")
        return jsonify({
            'error': 'Erro ao resetar provedores',
            'message': str(e)
        }), 500

@analysis_bp.route('/test_search', methods=['POST'])
def test_search():
    """Testa sistema de busca"""
    
    try:
        data = request.get_json()
        query = data.get('query', 'teste mercado digital Brasil')
        max_results = min(int(data.get('max_results', 5)), 10)
        
        logger.info(f"🧪 Testando busca: {query}")
        
        # Testa busca
        results = production_search_manager.search_with_fallback(query, max_results)
        
        return jsonify({
            'success': True,
            'query': query,
            'results_count': len(results),
            'results': results,
            'provider_status': production_search_manager.get_provider_status(),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erro no teste de busca: {str(e)}")
        return jsonify({
            'error': 'Erro no teste de busca',
            'message': str(e)
        }), 500

@analysis_bp.route('/test_extraction', methods=['POST'])
def test_extraction():
    """Testa sistema de extração de conteúdo"""
    
    try:
        data = request.get_json()
        test_url = data.get('url', 'https://g1.globo.com/tecnologia/')
        
        logger.info(f"🧪 Testando extração: {test_url}")
        
        # Testa extração
        content = robust_content_extractor.extract_content(test_url)
        
        if content:
            # Valida qualidade
            validation = content_quality_validator.validate_content(content, test_url)
            
            return jsonify({
                'success': True,
                'url': test_url,
                'content_length': len(content),
                'content_preview': content[:500] + '...' if len(content) > 500 else content,
                'validation': validation,
                'extractor_stats': robust_content_extractor.get_extractor_stats(),
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'url': test_url,
                'error': 'Falha na extração de conteúdo',
                'extractor_stats': robust_content_extractor.get_extractor_stats(),
                'timestamp': datetime.now().isoformat()
            })
        
    except Exception as e:
        logger.error(f"Erro no teste de extração: {str(e)}")
        return jsonify({
            'error': 'Erro no teste de extração',
            'message': str(e)
        }), 500

@analysis_bp.route('/extractor_stats', methods=['GET'])
def get_extractor_stats():
    """Obtém estatísticas dos extratores"""
    
    try:
        stats = robust_content_extractor.get_extractor_stats()
        
        return jsonify({
            'success': True,
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erro ao obter estatísticas: {str(e)}")
        return jsonify({
            'error': 'Erro ao obter estatísticas dos extratores',
            'message': str(e)
        }), 500

@analysis_bp.route('/reset_extractors', methods=['POST'])
def reset_extractors():
    """Reset estatísticas dos extratores"""
    
    try:
        data = request.get_json() or {}
        extractor_name = data.get('extractor')
        
        robust_content_extractor.reset_extractor_stats(extractor_name)
        
        message = f"Reset estatísticas do extrator: {extractor_name}" if extractor_name else "Reset estatísticas de todos os extratores"
        
        return jsonify({
            'success': True,
            'message': message,
            'stats': robust_content_extractor.get_extractor_stats(),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erro ao resetar extratores: {str(e)}")
        return jsonify({
            'error': 'Erro ao resetar extratores',
            'message': str(e)
        }), 500

@analysis_bp.route('/analyze', methods=['POST'])
def analyze():
    """Endpoint principal de análise"""
    
    try:
        data = request.get_json()
        
        # Validar dados obrigatórios
        required_fields = ['segmento', 'produto', 'publico']
        missing_fields = [field for field in required_fields if not data.get(field)]
        
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Campos obrigatórios: {", ".join(missing_fields)}'
            }), 400
        
        logger.info(f"🚀 Iniciando análise: {data.get('segmento')} - {data.get('produto')}")
        
        # Usar o engine de análise existente
        from services.enhanced_analysis_engine import enhanced_analysis_engine
        
        # Gerar análise completa
        analysis_result = enhanced_analysis_engine.generate_complete_analysis(data)
        
        if not analysis_result:
            return jsonify({
                'success': False,
                'error': 'Falha na geração da análise'
            }), 500
        
        return jsonify({
            'success': True,
            'analysis': analysis_result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erro na análise: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Erro interno: {str(e)}'
        }), 500

@analysis_bp.route('/test_ai', methods=['POST'])
def test_ai():
    """Testa sistema de IA"""
    
    try:
        data = request.get_json()
        prompt = data.get('prompt', 'Gere um breve resumo sobre o mercado digital brasileiro em 2024.')
        
        logger.info("🧪 Testando sistema de IA...")
        
        # Testa IA
        response = ai_manager.generate_analysis(prompt, max_tokens=500)
        
        return jsonify({
            'success': bool(response),
            'prompt': prompt,
            'response': response,
            'response_length': len(response) if response else 0,
            'provider_status': ai_manager.get_provider_status(),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erro no teste de IA: {str(e)}")
        return jsonify({
            'error': 'Erro no teste de IA',
            'message': str(e)
        }), 500

@analysis_bp.route('/upload_attachment', methods=['POST'])
def upload_attachment():
    """Upload e processamento de anexos"""
    
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Nenhum arquivo enviado'
            }), 400
        
        file = request.files['file']
        session_id = request.form.get('session_id', f"session_{int(time.time())}")
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Nome de arquivo vazio'
            }), 400
        
        # Processa anexo
        result = attachment_service.process_attachment(file, session_id)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Erro no upload de anexo: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Erro interno no processamento do anexo',
            'message': str(e)
        }), 500

@analysis_bp.route('/list_analyses', methods=['GET'])
def list_analyses():
    """Lista análises salvas"""
    
    try:
        limit = min(int(request.args.get('limit', 20)), 100)
        offset = int(request.args.get('offset', 0))
        
        analyses = db_manager.list_analyses(limit, offset)
        
        return jsonify({
            'success': True,
            'analyses': analyses,
            'count': len(analyses),
            'limit': limit,
            'offset': offset,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erro ao listar análises: {str(e)}")
        return jsonify({
            'error': 'Erro ao listar análises',
            'message': str(e)
        }), 500

@analysis_bp.route('/get_analysis/<int:analysis_id>', methods=['GET'])
def get_analysis(analysis_id):
    """Obtém análise específica"""
    
    try:
        analysis = db_manager.get_analysis(analysis_id)
        
        if analysis:
            return jsonify({
                'success': True,
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Análise não encontrada'
            }), 404
            
    except Exception as e:
        logger.error(f"Erro ao obter análise {analysis_id}: {str(e)}")
        return jsonify({
            'error': 'Erro ao obter análise',
            'message': str(e)
        }), 500

@analysis_bp.route('/stats', methods=['GET'])
def get_stats():
    """Obtém estatísticas do sistema"""
    
    try:
        db_stats = db_manager.get_stats()
        ai_status = ai_manager.get_provider_status()
        search_status = production_search_manager.get_provider_status()
        
        return jsonify({
            'database_stats': db_stats,
            'ai_providers': ai_status,
            'search_providers': search_status,
            'system_health': {
                'ai_available': len([p for p in ai_status.values() if p['available']]),
                'search_available': len([p for p in search_status.values() if p['available']]),
                'database_connected': db_manager.test_connection()
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erro ao obter estatísticas: {str(e)}")
        return jsonify({
            'error': 'Erro ao obter estatísticas',
            'message': str(e)
        }), 500