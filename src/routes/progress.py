#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v2.0 - Progress Routes
Endpoints para progresso em tempo real da análise
"""

import os
import logging
import time
import json
from datetime import datetime
from flask import Blueprint, request, jsonify, session
from flask_socketio import SocketIO, emit, join_room, leave_room
import threading
from queue import Queue

logger = logging.getLogger(__name__)

# Cria blueprint
progress_bp = Blueprint('progress', __name__)

# Sistema de progresso global
progress_sessions = {}
progress_queues = {}

class ProgressTracker:
    """Rastreador de progresso em tempo real"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.current_step = 0
        self.total_steps = 13
        self.start_time = time.time()
        self.steps = [
            "🔍 Coletando dados do formulário",
            "📊 Processando anexos inteligentes", 
            "🌐 Realizando pesquisa profunda massiva",
            "🧠 Analisando com múltiplas IAs",
            "👤 Criando avatar arqueológico completo",
            "🧠 Gerando drivers mentais customizados",
            "🎭 Desenvolvendo provas visuais instantâneas",
            "🛡️ Construindo sistema anti-objeção",
            "🎯 Arquitetando pré-pitch invisível",
            "⚔️ Mapeando concorrência profunda",
            "📈 Calculando métricas e projeções",
            "🔮 Predizendo futuro do mercado",
            "✨ Consolidando insights exclusivos"
        ]
        self.detailed_logs = []
        
        # Registra sessão global
        progress_sessions[session_id] = self
        progress_queues[session_id] = Queue()
    
    def update_progress(self, step: int, message: str, details: str = None):
        """Atualiza progresso da análise"""
        self.current_step = step
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Calcula tempo estimado
        if step > 0:
            estimated_total = (elapsed / step) * self.total_steps
            remaining = max(0, estimated_total - elapsed)
        else:
            remaining = 0
        
        progress_data = {
            "session_id": self.session_id,
            "current_step": step,
            "total_steps": self.total_steps,
            "percentage": (step / self.total_steps) * 100,
            "current_message": message,
            "detailed_message": details or message,
            "elapsed_time": elapsed,
            "estimated_remaining": remaining,
            "estimated_total": elapsed + remaining,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log detalhado
        log_entry = {
            "step": step,
            "message": message,
            "details": details,
            "timestamp": datetime.now().isoformat(),
            "elapsed": elapsed
        }
        self.detailed_logs.append(log_entry)
        
        # Adiciona à queue para polling
        if self.session_id in progress_queues:
            try:
                progress_queues[self.session_id].put(progress_data)
            except:
                pass
        
        logger.info(f"Progress {self.session_id}: Step {step}/{self.total_steps} - {message}")
        
        return progress_data
    
    def complete(self):
        """Marca análise como completa"""
        self.update_progress(self.total_steps, "🎉 Análise concluída! Preparando resultados...")
        
        # Remove da sessão após 5 minutos
        def cleanup():
            time.sleep(300)  # 5 minutos
            if self.session_id in progress_sessions:
                del progress_sessions[self.session_id]
            if self.session_id in progress_queues:
                del progress_queues[self.session_id]
        
        threading.Thread(target=cleanup, daemon=True).start()
    
    def get_current_status(self):
        """Retorna status atual"""
        elapsed = time.time() - self.start_time
        
        if self.current_step > 0:
            estimated_total = (elapsed / self.current_step) * self.total_steps
            remaining = max(0, estimated_total - elapsed)
        else:
            remaining = 0
        
        return {
            "session_id": self.session_id,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "percentage": (self.current_step / self.total_steps) * 100,
            "current_message": self.steps[min(self.current_step, len(self.steps) - 1)],
            "elapsed_time": elapsed,
            "estimated_remaining": remaining,
            "detailed_logs": self.detailed_logs[-5:],  # Últimos 5 logs
            "is_complete": self.current_step >= self.total_steps
        }

@progress_bp.route('/start_tracking', methods=['POST'])
def start_tracking():
    """Inicia rastreamento de progresso"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({
                'error': 'Session ID obrigatório'
            }), 400
        
        # Cria novo tracker
        tracker = ProgressTracker(session_id)
        tracker.update_progress(0, "🚀 Iniciando análise ultra-detalhada...")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Rastreamento iniciado',
            'status': tracker.get_current_status()
        })
        
    except Exception as e:
        logger.error(f"Erro ao iniciar rastreamento: {str(e)}")
        return jsonify({
            'error': 'Erro interno',
            'message': str(e)
        }), 500

@progress_bp.route('/get_progress/<session_id>', methods=['GET'])
def get_progress(session_id):
    """Obtém progresso atual da análise"""
    try:
        if session_id not in progress_sessions:
            return jsonify({
                'error': 'Sessão não encontrada',
                'session_id': session_id
            }), 404
        
        tracker = progress_sessions[session_id]
        status = tracker.get_current_status()
        
        return jsonify({
            'success': True,
            'progress': status
        })
        
    except Exception as e:
        logger.error(f"Erro ao obter progresso: {str(e)}")
        return jsonify({
            'error': 'Erro interno',
            'message': str(e)
        }), 500

@progress_bp.route('/poll_updates/<session_id>', methods=['GET'])
def poll_updates(session_id):
    """Polling para atualizações de progresso"""
    try:
        if session_id not in progress_queues:
            return jsonify({
                'error': 'Sessão não encontrada'
            }), 404
        
        queue = progress_queues[session_id]
        updates = []
        
        # Coleta todas as atualizações disponíveis
        while not queue.empty():
            try:
                update = queue.get_nowait()
                updates.append(update)
            except:
                break
        
        return jsonify({
            'success': True,
            'updates': updates,
            'has_updates': len(updates) > 0
        })
        
    except Exception as e:
        logger.error(f"Erro no polling: {str(e)}")
        return jsonify({
            'error': 'Erro interno',
            'message': str(e)
        }), 500

@progress_bp.route('/update_progress', methods=['POST'])
def update_progress():
    """Atualiza progresso (usado internamente)"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        step = data.get('step')
        message = data.get('message')
        details = data.get('details')
        
        if session_id not in progress_sessions:
            return jsonify({
                'error': 'Sessão não encontrada'
            }), 404
        
        tracker = progress_sessions[session_id]
        progress_data = tracker.update_progress(step, message, details)
        
        return jsonify({
            'success': True,
            'progress': progress_data
        })
        
    except Exception as e:
        logger.error(f"Erro ao atualizar progresso: {str(e)}")
        return jsonify({
            'error': 'Erro interno',
            'message': str(e)
        }), 500

@progress_bp.route('/complete_analysis', methods=['POST'])
def complete_analysis():
    """Marca análise como completa"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if session_id not in progress_sessions:
            return jsonify({
                'error': 'Sessão não encontrada'
            }), 404
        
        tracker = progress_sessions[session_id]
        tracker.complete()
        
        return jsonify({
            'success': True,
            'message': 'Análise marcada como completa',
            'final_status': tracker.get_current_status()
        })
        
    except Exception as e:
        logger.error(f"Erro ao completar análise: {str(e)}")
        return jsonify({
            'error': 'Erro interno',
            'message': str(e)
        }), 500

@progress_bp.route('/get_detailed_logs/<session_id>', methods=['GET'])
def get_detailed_logs(session_id):
    """Obtém logs detalhados da análise"""
    try:
        if session_id not in progress_sessions:
            return jsonify({
                'error': 'Sessão não encontrada'
            }), 404
        
        tracker = progress_sessions[session_id]
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'logs': tracker.detailed_logs,
            'total_logs': len(tracker.detailed_logs),
            'analysis_duration': time.time() - tracker.start_time
        })
        
    except Exception as e:
        logger.error(f"Erro ao obter logs: {str(e)}")
        return jsonify({
            'error': 'Erro interno',
            'message': str(e)
        }), 500

@progress_bp.route('/active_sessions', methods=['GET'])
def get_active_sessions():
    """Lista sessões ativas de progresso"""
    try:
        active = []
        current_time = time.time()
        
        for session_id, tracker in progress_sessions.items():
            active.append({
                'session_id': session_id,
                'current_step': tracker.current_step,
                'total_steps': tracker.total_steps,
                'elapsed_time': current_time - tracker.start_time,
                'is_complete': tracker.current_step >= tracker.total_steps,
                'last_message': tracker.steps[min(tracker.current_step, len(tracker.steps) - 1)]
            })
        
        return jsonify({
            'success': True,
            'active_sessions': active,
            'total_active': len(active)
        })
        
    except Exception as e:
        logger.error(f"Erro ao listar sessões: {str(e)}")
        return jsonify({
            'error': 'Erro interno',
            'message': str(e)
        }), 500

# Função helper para usar em outros módulos
def get_progress_tracker(session_id: str) -> ProgressTracker:
    """Obtém tracker de progresso para uma sessão"""
    if session_id not in progress_sessions:
        return ProgressTracker(session_id)
    return progress_sessions[session_id]

def update_analysis_progress(session_id: str, step: int, message: str, details: str = None):
    """Função helper para atualizar progresso de qualquer lugar"""
    if session_id in progress_sessions:
        tracker = progress_sessions[session_id]
        return tracker.update_progress(step, message, details)
    return None
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v2.0 - Rotas de Progresso
Tracking de progresso das análises
"""

import logging
from flask import Blueprint, request, jsonify, session
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

progress_bp = Blueprint('progress', __name__)

# Armazenamento em memória para progresso (em produção usaria Redis/Database)
progress_data = {}

@progress_bp.route('/progress/start_tracking', methods=['POST'])
def start_tracking():
    """Inicia tracking de progresso"""
    try:
        data = request.get_json() or {}
        
        # Gerar ID de sessão se não existir
        session_id = session.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
            session['session_id'] = session_id
        
        # Inicializar progresso
        progress_data[session_id] = {
            'status': 'iniciado',
            'progress': 0,
            'message': 'Preparando análise...',
            'started_at': datetime.now().isoformat(),
            'steps': [
                {'name': 'Validação de dados', 'status': 'pending'},
                {'name': 'Análise de mercado', 'status': 'pending'},
                {'name': 'Geração de insights', 'status': 'pending'},
                {'name': 'Criação de estratégias', 'status': 'pending'},
                {'name': 'Finalização', 'status': 'pending'}
            ]
        }
        
        logger.info(f"Tracking iniciado para sessão: {session_id}")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Tracking iniciado'
        })
        
    except Exception as e:
        logger.error(f"Erro ao iniciar tracking: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@progress_bp.route('/progress/update', methods=['POST'])
def update_progress():
    """Atualiza progresso"""
    try:
        data = request.get_json()
        session_id = data.get('session_id') or session.get('session_id')
        
        if not session_id or session_id not in progress_data:
            return jsonify({
                'success': False,
                'error': 'Sessão não encontrada'
            }), 404
        
        # Atualizar progresso
        progress_data[session_id].update({
            'progress': data.get('progress', 0),
            'message': data.get('message', ''),
            'updated_at': datetime.now().isoformat()
        })
        
        # Atualizar step se fornecido
        step_index = data.get('step_index')
        if step_index is not None and 0 <= step_index < len(progress_data[session_id]['steps']):
            progress_data[session_id]['steps'][step_index]['status'] = data.get('step_status', 'completed')
        
        return jsonify({
            'success': True,
            'message': 'Progresso atualizado'
        })
        
    except Exception as e:
        logger.error(f"Erro ao atualizar progresso: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@progress_bp.route('/progress/status/<session_id>', methods=['GET'])
def get_progress(session_id):
    """Obtém status do progresso"""
    try:
        if session_id not in progress_data:
            return jsonify({
                'success': False,
                'error': 'Sessão não encontrada'
            }), 404
        
        return jsonify({
            'success': True,
            'data': progress_data[session_id]
        })
        
    except Exception as e:
        logger.error(f"Erro ao obter progresso: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@progress_bp.route('/progress/complete', methods=['POST'])
def complete_progress():
    """Completa progresso"""
    try:
        data = request.get_json()
        session_id = data.get('session_id') or session.get('session_id')
        
        if not session_id or session_id not in progress_data:
            return jsonify({
                'success': False,
                'error': 'Sessão não encontrada'
            }), 404
        
        # Completar progresso
        progress_data[session_id].update({
            'status': 'concluido',
            'progress': 100,
            'message': 'Análise concluída com sucesso!',
            'completed_at': datetime.now().isoformat()
        })
        
        # Marcar todos os steps como concluídos
        for step in progress_data[session_id]['steps']:
            step['status'] = 'completed'
        
        return jsonify({
            'success': True,
            'message': 'Progresso concluído'
        })
        
    except Exception as e:
        logger.error(f"Erro ao completar progresso: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
