#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v2.0 - AI Manager com Sistema de Fallback
Gerenciador inteligente de múltiplas IAs com fallback automático
"""

import os
import logging
import time
import json
from typing import Dict, List, Optional, Any
import requests

# Imports condicionais para os clientes de IA
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from services.groq_client import groq_client
    HAS_GROQ_CLIENT = True
except ImportError:
    HAS_GROQ_CLIENT = False

logger = logging.getLogger(__name__)

class AIManager:
    """Gerenciador de IAs com sistema de fallback automático"""

    def __init__(self):
        """Inicializa o gerenciador de IAs"""
        self.providers = {
            'gemini': {
                'client': None,
                'available': False,
                'priority': 1,
                'error_count': 0,
                'model': 'gemini-1.5-flash',
                'max_errors': 3
            },
            'groq': {
                'client': None,
                'available': False,
                'priority': 2,
                'error_count': 0,
                'model': 'llama3-70b-8192',
                'max_errors': 3
            },
            'openai': {
                'client': None,
                'available': False,
                'priority': 3,
                'error_count': 0,
                'model': 'gpt-3.5-turbo',
                'max_errors': 3
            },
            'huggingface': {
                'client': None,
                'available': False,
                'priority': 4,
                'error_count': 0,
                'models': ["HuggingFaceH4/zephyr-7b-beta", "google/flan-t5-base"],
                'current_model_index': 0,
                'max_errors': 5
            }
        }

        self.initialize_providers()
        available_count = len([p for p in self.providers.values() if p['available']])
        logger.info(f"🤖 AI Manager inicializado com {available_count} provedores disponíveis.")

    def initialize_providers(self):
        """Inicializa todos os provedores de IA com base nas chaves de API disponíveis."""

        # Inicializa Gemini
        if HAS_GEMINI:
            try:
                gemini_key = os.getenv('GEMINI_API_KEY')
                if gemini_key:
                    genai.configure(api_key=gemini_key)
                    self.providers['gemini']['client'] = genai.GenerativeModel("gemini-1.5-flash")
                    self.providers['gemini']['available'] = True
                    logger.info("✅ Gemini (gemini-1.5-flash) inicializado com sucesso")
            except Exception as e:
                logger.warning(f"⚠️ Falha ao inicializar Gemini: {str(e)}")
        else:
            logger.warning("⚠️ Biblioteca 'google-generativeai' não instalada. Gemini desabilitado.")

        # Inicializa OpenAI
        if HAS_OPENAI:
            try:
                openai_key = os.getenv('OPENAI_API_KEY')
                if openai_key:
                    self.providers["openai"]["client"] = openai.OpenAI(api_key=openai_key)
                    self.providers["openai"]["available"] = True
                    logger.info("✅ OpenAI (gpt-3.5-turbo) inicializado com sucesso")
            except Exception as e:
                logger.warning(f"⚠️ Falha ao inicializar OpenAI: {str(e)}")
        else:
            logger.warning("⚠️ Biblioteca 'openai' não instalada. OpenAI desabilitado.")

        # Inicializa Groq
        try:
            if HAS_GROQ_CLIENT and groq_client and groq_client.is_enabled():
                self.providers['groq']['client'] = groq_client
                self.providers['groq']['available'] = True
                logger.info("✅ Groq (llama3-70b-8192) inicializado com sucesso")
            else:
                logger.warning("⚠️ Groq client não está habilitado")
        except Exception as e:
            logger.warning(f"⚠️ Falha ao inicializar Groq: {str(e)}")

        # Inicializa HuggingFace
        try:
            hf_key = os.getenv('HUGGINGFACE_API_KEY')
            if hf_key:
                self.providers['huggingface']['client'] = {
                    'api_key': hf_key,
                    'base_url': 'https://api-inference.huggingface.co/models/'
                }
                self.providers['huggingface']['available'] = True
                logger.info("✅ HuggingFace inicializado com sucesso")
        except Exception as e:
            logger.warning(f"⚠️ Falha ao inicializar HuggingFace: {str(e)}")

    def get_best_provider(self) -> Optional[str]:
        """Retorna o melhor provedor disponível com base na prioridade e contagem de erros."""
        available_providers = [
            (name, provider) for name, provider in self.providers.items() 
            if provider['available'] and provider['error_count'] < provider.get('max_errors', 3)
        ]

        if not available_providers:
            logger.warning("Nenhum provedor saudável disponível. Resetando contagem de erros.")
            for provider in self.providers.values():
                provider['error_count'] = 0
            available_providers = [(name, p) for name, p in self.providers.items() if p['available']]

        if available_providers:
            available_providers.sort(key=lambda x: (x[1]['priority'], x[1]['error_count']))
            return available_providers[0][0]

        return None

    def generate_analysis(self, prompt: str, max_tokens: int = 8192, provider: Optional[str] = None) -> Optional[str]:
        """Gera análise usando um provedor específico ou o melhor disponível com fallback."""
        
        # Se um provedor específico for solicitado
        if provider:
            if self.providers.get(provider) and self.providers[provider]['available']:
                logger.info(f"🤖 Usando provedor solicitado: {provider.upper()}")
                try:
                    result = self._call_provider(provider, prompt, max_tokens)
                    if result:
                        return result
                    else:
                        raise Exception("Resposta vazia")
                except Exception as e:
                    logger.error(f"❌ Provedor solicitado {provider.upper()} falhou: {e}")
                    self.providers[provider]['error_count'] += 1
                    return None # Não tenta fallback se um provedor específico foi pedido e falhou
            else:
                logger.error(f"❌ Provedor solicitado '{provider}' não está disponível.")
                return None

        # Lógica de fallback padrão
        provider_name = self.get_best_provider()
        if not provider_name:
            raise Exception("NENHUM PROVEDOR DE IA DISPONÍVEL: Configure pelo menos uma API de IA.")

        try:
            return self._call_provider(provider_name, prompt, max_tokens)
        except Exception as e:
            logger.error(f"❌ Erro no provedor {provider_name}: {e}")
            self.providers[provider_name]['error_count'] += 1
            return self._try_fallback(prompt, max_tokens, exclude=[provider_name])

    def _call_provider(self, provider_name: str, prompt: str, max_tokens: int) -> Optional[str]:
        """Chama a função de geração do provedor especificado."""
        if provider_name == 'gemini':
            return self._generate_with_gemini(prompt, max_tokens)
        elif provider_name == 'groq':
            return self._generate_with_groq(prompt, max_tokens)
        elif provider_name == 'openai':
            return self._generate_with_openai(prompt, max_tokens)
        elif provider_name == 'huggingface':
            return self._generate_with_huggingface(prompt, max_tokens)
        return None

    def _generate_with_gemini(self, prompt: str, max_tokens: int) -> Optional[str]:
        """Gera conteúdo usando Gemini."""
        client = self.providers['gemini']['client']
        config = {"temperature": 0.7, "max_output_tokens": min(max_tokens, 8192)}
        safety = [
            {"category": c, "threshold": "BLOCK_NONE"} 
            for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
        ]
        response = client.generate_content(prompt, generation_config=config, safety_settings=safety)
        if response.text:
            logger.info(f"✅ Gemini gerou {len(response.text)} caracteres")
            return response.text
        raise Exception("Resposta vazia do Gemini")

    def _generate_with_groq(self, prompt: str, max_tokens: int) -> Optional[str]:
        """Gera conteúdo usando Groq."""
        client = self.providers['groq']['client']
        content = client.generate(prompt, max_tokens=min(max_tokens, 8192))
        if content:
            logger.info(f"✅ Groq gerou {len(content)} caracteres")
            return content
        raise Exception("Resposta vazia do Groq")

    def _generate_with_openai(self, prompt: str, max_tokens: int) -> Optional[str]:
        """Gera conteúdo usando OpenAI."""
        client = self.providers['openai']['client']
        response = client.chat.completions.create(
            model=self.providers['openai']['model'],
            messages=[
                {"role": "system", "content": "Você é um especialista em análise de mercado ultra-detalhada."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=min(max_tokens, 4096),
            temperature=0.7
        )
        content = response.choices[0].message.content
        if content:
            logger.info(f"✅ OpenAI gerou {len(content)} caracteres")
            return content
        raise Exception("Resposta vazia do OpenAI")

    def _generate_with_huggingface(self, prompt: str, max_tokens: int) -> Optional[str]:
        """Gera conteúdo usando HuggingFace com rotação de modelos."""
        config = self.providers['huggingface']
        for _ in range(len(config['models'])):
            model_index = config['current_model_index']
            model = config['models'][model_index]
            config['current_model_index'] = (model_index + 1) % len(config['models']) # Rotaciona para a próxima vez
            
            try:
                url = f"{config['client']['base_url']}{model}"
                headers = {"Authorization": f"Bearer {config['client']['api_key']}"}
                payload = {"inputs": prompt, "parameters": {"max_new_tokens": min(max_tokens, 1024)}}
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                
                if response.status_code == 200:
                    res_json = response.json()
                    content = res_json[0].get("generated_text", "")
                    if content:
                        logger.info(f"✅ HuggingFace ({model}) gerou {len(content)} caracteres")
                        return content
                elif response.status_code == 503:
                    logger.warning(f"⚠️ Modelo HuggingFace {model} está carregando (503), tentando próximo...")
                    continue
                else:
                    logger.warning(f"⚠️ Erro {response.status_code} no modelo {model}")
                    continue
            except Exception as e:
                logger.warning(f"⚠️ Erro no modelo {model}: {e}")
                continue
        raise Exception("Todos os modelos HuggingFace falharam")

    def reset_provider_errors(self, provider_name: str = None):
        """Reset contadores de erro dos provedores"""
        if provider_name:
            if provider_name in self.providers:
                self.providers[provider_name]['error_count'] = 0
                logger.info(f"🔄 Reset erros do provedor: {provider_name}")
        else:
            for provider in self.providers.values():
                provider['error_count'] = 0
            logger.info("🔄 Reset erros de todos os provedores")

    def _try_fallback(self, prompt: str, max_tokens: int, exclude: List[str]) -> Optional[str]:
        """Tenta usar o próximo provedor disponível como fallback."""
        logger.info(f"🔄 Acionando fallback, excluindo: {', '.join(exclude)}")
        
        # Ordena provedores por prioridade, excluindo os que já falharam
        available_providers = [
            (name, provider) for name, provider in self.providers.items()
            if (provider['available'] and 
                name not in exclude and 
                provider['error_count'] < provider.get('max_errors', 3))
        ]
        
        if not available_providers:
            logger.critical("❌ Todos os provedores de fallback falharam.")
            return None
        
        # Ordena por prioridade
        available_providers.sort(key=lambda x: (x[1]['priority'], x[1]['error_count']))
        next_provider = available_providers[0][0]
        
        logger.info(f"🔄 Tentando fallback para: {next_provider.upper()}")
        
        try:
            return self._call_provider(next_provider, prompt, max_tokens)
        except Exception as e:
            logger.error(f"❌ Fallback para {next_provider} também falhou: {e}")
            self.providers[next_provider]['error_count'] += 1
            return self._try_fallback(prompt, max_tokens, exclude + [next_provider])

# Instância global
ai_manager = AIManager()
