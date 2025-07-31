
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v2.0 - Monitoring Routes
Endpoints para monitoramento do sistema de extração
"""
from flask import Blueprint, jsonify, request
from services.robust_content_extractor import robust_content_extractor
import logging

logger = logging.getLogger(__name__)

monitoring_bp = Blueprint('monitoring', __name__)


@monitoring_bp.route('/api/extractor_stats', methods=['GET'])
def get_extractor_stats():
    """Retorna estatísticas dos extratores"""
    try:
        stats = robust_content_extractor.get_extractor_stats()
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        logger.error(f"❌ Erro ao obter estatísticas: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@monitoring_bp.route('/api/test_extraction', methods=['GET'])
def test_extraction():
    """Testa extração para uma URL específica"""
    url = request.args.get('url')
    
    if not url:
        return jsonify({
            'success': False,
            'error': 'URL é obrigatória'
        }), 400
    
    try:
        result = robust_content_extractor.test_extraction(url)
        return jsonify({
            'success': True,
            'result': result
        })
    except Exception as e:
        logger.error(f"❌ Erro ao testar extração: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@monitoring_bp.route('/api/health', methods=['GET'])
def health_check():
    """Verifica saúde do sistema"""
    try:
        # Testa extração com URL simples
        test_url = "https://httpbin.org/html"
        result = robust_content_extractor.test_extraction(test_url)
        
        stats = robust_content_extractor.get_extractor_stats()
        available_extractors = sum(1 for name, data in stats.items() 
                                 if name != 'global' and data.get('available', False))
        
        return jsonify({
            'success': True,
            'status': 'healthy' if available_extractors > 0 else 'degraded',
            'available_extractors': available_extractors,
            'test_extraction': result['success'],
            'stats': stats
        })
    except Exception as e:
        logger.error(f"❌ Erro no health check: {str(e)}")
        return jsonify({
            'success': False,
            'status': 'unhealthy',
            'error': str(e)
        }), 500
