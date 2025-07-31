#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v2.0 - Production Search Manager
Sistema de busca robusto para produção com cache, fallback e otimizações
"""

import os
import logging
import time
import requests
import json
import hashlib
import random
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import quote_plus, urljoin
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import sqlite3
from dataclasses import dataclass
from services.robust_content_extractor import robust_content_extractor
from services.url_resolver import resolve_url
from services.content_quality_validator import content_quality_validator

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Estrutura de dados para resultados de busca"""
    title: str
    url: str
    snippet: str
    source: str
    relevance_score: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class ProductionSearchCache:
    """Sistema de cache robusto para produção"""

    def __init__(self, cache_dir: str = "cache", ttl: int = 3600):
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.db_path = os.path.join(cache_dir, "search_cache.db")
        os.makedirs(cache_dir, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Inicializa banco de dados SQLite para cache"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS search_cache (
                        query_hash TEXT PRIMARY KEY,
                        query TEXT NOT NULL,
                        results BLOB NOT NULL,
                        timestamp REAL NOT NULL,
                        ttl INTEGER NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp ON search_cache(timestamp)
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Erro ao inicializar cache: {e}")

    def _get_query_hash(self, query: str, provider: str = "") -> str:
        """Gera hash único para query"""
        combined = f"{query}:{provider}".encode('utf-8')
        return hashlib.sha256(combined).hexdigest()

    def get(self, query: str, provider: str = "") -> Optional[List[SearchResult]]:
        """Recupera resultados do cache"""
        if not os.getenv('SEARCH_CACHE_ENABLED', 'true').lower() == 'true':
            return None

        try:
            query_hash = self._get_query_hash(query, provider)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT results, timestamp, ttl FROM search_cache WHERE query_hash = ?",
                    (query_hash,)
                )
                row = cursor.fetchone()

                if row:
                    results_blob, timestamp, ttl = row

                    # Verifica se não expirou
                    if time.time() - timestamp < ttl:
                        results = pickle.loads(results_blob)
                        logger.info(f"✅ Cache hit para query: {query[:50]}...")
                        return results
                    else:
                        # Remove entrada expirada
                        conn.execute("DELETE FROM search_cache WHERE query_hash = ?", (query_hash,))
                        conn.commit()
                        logger.info(f"🗑️ Cache expirado removido para: {query[:50]}...")

                return None

        except Exception as e:
            logger.error(f"Erro ao recuperar cache: {e}")
            return None

    def set(self, query: str, results: List[SearchResult], provider: str = ""):
        """Armazena resultados no cache"""
        if not os.getenv('SEARCH_CACHE_ENABLED', 'true').lower() == 'true':
            return

        try:
            query_hash = self._get_query_hash(query, provider)
            results_blob = pickle.dumps(results)
            timestamp = time.time()
            ttl = int(os.getenv('SEARCH_CACHE_TTL', self.ttl))

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO search_cache 
                    (query_hash, query, results, timestamp, ttl) 
                    VALUES (?, ?, ?, ?, ?)
                """, (query_hash, query, results_blob, timestamp, ttl))
                conn.commit()

            logger.info(f"💾 Cache salvo para query: {query[:50]}...")

        except Exception as e:
            logger.error(f"Erro ao salvar cache: {e}")

    def cleanup_expired(self):
        """Remove entradas expiradas do cache"""
        try:
            current_time = time.time()

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM search_cache WHERE ? - timestamp > ttl",
                    (current_time,)
                )
                expired_count = cursor.fetchone()[0]

                if expired_count > 0:
                    conn.execute("DELETE FROM search_cache WHERE ? - timestamp > ttl", (current_time,))
                    conn.commit()
                    logger.info(f"🗑️ {expired_count} entradas expiradas removidas do cache")

        except Exception as e:
            logger.error(f"Erro na limpeza do cache: {e}")

class ProductionSearchManager:
    """Gerenciador de busca robusto para produção"""

    def __init__(self):
        """Inicializa o gerenciador de busca para produção"""
        self.cache = ProductionSearchCache()
        self.rate_limiter = {}
        self.error_counts = {}
        self.last_cleanup = time.time()
        self.content_extractor = robust_content_extractor

        # Configurações de produção
        self.max_retries = int(os.getenv('SEARCH_MAX_RETRIES', 3))
        self.retry_delay = float(os.getenv('SEARCH_RETRY_DELAY', 2.0))
        self.rate_limit_delay = float(os.getenv('SEARCH_RATE_LIMIT_DELAY', 1.5))
        self.request_timeout = int(os.getenv('REQUEST_TIMEOUT', 30))

        # User agents rotativos para evitar detecção
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0'
        ]

        # Configuração de provedores
        self.providers = {
            'google': {
                'enabled': bool(os.getenv('GOOGLE_SEARCH_KEY') and os.getenv('GOOGLE_CSE_ID')),
                'priority': 1,
                'rate_limit': 100,  # requests per day
                'error_count': 0,
                'last_error': None,
                'quota_reset': None
            },
            'serper': {
                'enabled': bool(os.getenv('SERPER_API_KEY')),
                'priority': 2,
                'rate_limit': 2500,  # requests per month
                'error_count': 0,
                'last_error': None,
                'quota_reset': None
            },
            'bing': {
                'enabled': True,  # Sempre disponível via scraping
                'priority': 3,
                'rate_limit': 1000,  # requests per hour
                'error_count': 0,
                'last_error': None,
                'quota_reset': None
            },
            'duckduckgo': {
                'enabled': True,  # Sempre disponível via scraping
                'priority': 4,
                'rate_limit': 500,  # requests per hour
                'error_count': 0,
                'last_error': None,
                'quota_reset': None
            }
        }

        logger.info("🚀 Production Search Manager inicializado")
        self._log_provider_status()

        # Configurações de serviços de extração (Jina e Mercury removidos - são pagos/inativos)
        # self.jina_api_key = os.getenv('JINA_API_KEY')  # Removido - serviço pago
        # self.mercury_api_key = os.getenv('MERCURY_API_KEY')  # Removido - serviço descontinuado

    def _log_provider_status(self):
        """Log do status dos provedores"""
        enabled_providers = [name for name, config in self.providers.items() if config['enabled']]
        logger.info(f"📊 Provedores habilitados: {', '.join(enabled_providers)}")

        for name, config in self.providers.items():
            if not config['enabled']:
                logger.warning(f"⚠️ Provedor {name} desabilitado")

    def _get_headers(self, provider: str = "") -> Dict[str, str]:
        """Gera headers otimizados para cada provedor"""
        base_headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'pt-BR,pt;q=0.9,en;q=0.8,en-US;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }

        # User agent rotativo
        if os.getenv('SEARCH_USER_AGENT_ROTATION', 'true').lower() == 'true':
            base_headers['User-Agent'] = random.choice(self.user_agents)
        else:
            base_headers['User-Agent'] = self.user_agents[0]

        # Headers específicos por provedor
        if provider == 'google':
            base_headers.update({
                'Referer': 'https://www.google.com/',
                'Origin': 'https://www.google.com'
            })
        elif provider == 'bing':
            base_headers.update({
                'Referer': 'https://www.bing.com/',
                'Origin': 'https://www.bing.com'
            })
        elif provider == 'duckduckgo':
            base_headers.update({
                'Referer': 'https://duckduckgo.com/',
                'Origin': 'https://duckduckgo.com'
            })

        return base_headers

    def _check_rate_limit(self, provider: str) -> bool:
        """Verifica rate limiting"""
        current_time = time.time()

        if provider not in self.rate_limiter:
            self.rate_limiter[provider] = []

        # Remove requisições antigas (última hora)
        self.rate_limiter[provider] = [
            req_time for req_time in self.rate_limiter[provider]
            if current_time - req_time < 3600
        ]

        # Verifica limite
        limit = self.providers[provider]['rate_limit']
        if len(self.rate_limiter[provider]) >= limit:
            logger.warning(f"⚠️ Rate limit atingido para {provider}")
            return False

        return True

    def _record_request(self, provider: str):
        """Registra requisição para rate limiting"""
        if provider not in self.rate_limiter:
            self.rate_limiter[provider] = []

        self.rate_limiter[provider].append(time.time())

    def _handle_provider_error(self, provider: str, error: Exception):
        """Gerencia erros de provedores"""
        self.error_counts[provider] = self.error_counts.get(provider, 0) + 1
        self.providers[provider]['error_count'] = self.error_counts[provider]
        self.providers[provider]['last_error'] = str(error)

        # Desabilita temporariamente se muitos erros
        if self.error_counts[provider] >= 5:
            logger.error(f"❌ Provedor {provider} desabilitado temporariamente (muitos erros)")
            self.providers[provider]['enabled'] = False
            # Reabilita após 1 hora
            self.providers[provider]['quota_reset'] = time.time() + 3600

    def _reset_provider_if_needed(self, provider: str):
        """Reabilita provedor se tempo de reset passou"""
        if (not self.providers[provider]['enabled'] and 
            self.providers[provider]['quota_reset'] and
            time.time() > self.providers[provider]['quota_reset']):

            logger.info(f"🔄 Reabilitando provedor {provider}")
            self.providers[provider]['enabled'] = True
            self.providers[provider]['quota_reset'] = None
            self.error_counts[provider] = 0

    def search_google_custom(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Busca usando Google Custom Search API com validação robusta"""
        provider = 'google'

        if not self.providers[provider]['enabled']:
            self._reset_provider_if_needed(provider)
            if not self.providers[provider]['enabled']:
                return []

        if not self._check_rate_limit(provider):
            return []

        try:
            api_key = os.getenv('GOOGLE_SEARCH_KEY')
            cse_id = os.getenv('GOOGLE_CSE_ID')

            if not api_key or not cse_id:
                logger.error("❌ Google Search API não configurada corretamente")
                self.providers[provider]['enabled'] = False
                return []

            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': api_key,
                'cx': cse_id,
                'q': query,
                'num': min(max_results, 10),
                'lr': 'lang_pt',
                'gl': 'br',
                'safe': 'off',
                'dateRestrict': 'y1',
                'sort': 'date',
                'fields': 'items(title,link,snippet,displayLink)'
            }

            headers = self._get_headers('google')

            self._record_request(provider)

            response = requests.get(
                url, 
                params=params, 
                headers=headers, 
                timeout=self.request_timeout
            )

            logger.info(f"🔍 Google API Response: {response.status_code}")

            if response.status_code == 200:
                data = response.json()

                # Verifica se há erro na resposta
                if 'error' in data:
                    error_msg = data['error'].get('message', 'Erro desconhecido')
                    logger.error(f"❌ Google API Error: {error_msg}")

                    # Verifica se é erro de quota
                    if 'quota' in error_msg.lower() or 'limit' in error_msg.lower():
                        self.providers[provider]['quota_reset'] = time.time() + 86400  # 24h
                        self.providers[provider]['enabled'] = False

                    # Retorna vazio em vez de tentar novamente
                    return []

                items = data.get('items', [])
                results = []

                for item in items:
                    result = SearchResult(
                        title=item.get('title', ''),
                        url=item.get('link', ''),
                        snippet=item.get('snippet', ''),
                        source='google_custom'
                    )

                    if result.url and result.title:
                        results.append(result)

                logger.info(f"✅ Google Custom Search: {len(results)} resultados válidos")
                return results

            elif response.status_code == 403:
                logger.error("❌ Google API: Acesso negado (403) - Verifique chaves e quotas")
                self.providers[provider]['enabled'] = False
                return []

            elif response.status_code == 429:
                logger.warning("⚠️ Google API: Rate limit (429) - Aguardando reset")
                self.providers[provider]['quota_reset'] = time.time() + 3600
                return []

            else:
                logger.error(f"❌ Google API: Status {response.status_code}")
                return []

        except requests.exceptions.Timeout:
            logger.error(f"⏰ Timeout na requisição Google Search")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erro de rede Google Search: {e}")
            self._handle_provider_error(provider, e)
            return []
        except Exception as e:
            logger.error(f"❌ Erro inesperado Google Search: {e}")
            self._handle_provider_error(provider, e)
            return []

    def search_serper(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Busca usando Serper API com validação robusta"""
        provider = 'serper'

        if not self.providers[provider]['enabled']:
            self._reset_provider_if_needed(provider)
            if not self.providers[provider]['enabled']:
                return []

        if not self._check_rate_limit(provider):
            return []

        try:
            api_key = os.getenv('SERPER_API_KEY')

            if not api_key or len(api_key) < 30:
                logger.error("❌ SERPER_API_KEY não configurada ou inválida")
                self.providers[provider]['enabled'] = False
                return []

            url = "https://google.serper.dev/search"
            headers = {
                **self._get_headers('serper'),
                'X-API-KEY': api_key,
                'Content-Type': 'application/json'
            }

            payload = {
                'q': query,
                'gl': 'br',
                'hl': 'pt',
                'num': max_results,
                'autocorrect': True,
                'page': 1
            }

            self._record_request(provider)

            response = requests.post(
                url, 
                json=payload, 
                headers=headers, 
                timeout=self.request_timeout
            )

            if response.status_code == 200:
                data = response.json()
                results = []

                for item in data.get('organic', []):
                    result = SearchResult(
                        title=item.get('title', ''),
                        url=item.get('link', ''),
                        snippet=item.get('snippet', ''),
                        source='serper'
                    )

                    if result.url and result.title:
                        results.append(result)

                logger.info(f"✅ Serper Search: {len(results)} resultados")
                return results

            elif response.status_code == 429:
                logger.warning("⚠️ Serper API: Rate limit atingido")
                self.providers[provider]['quota_reset'] = time.time() + 3600
                return []

            else:
                logger.error(f"❌ Serper API: Status {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"❌ Erro Serper Search: {e}")
            self._handle_provider_error(provider, e)
            return []

    def search_bing_scraping(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Busca Bing via scraping robusto com anti-detecção"""
        provider = 'bing'

        if not self.providers[provider]['enabled']:
            self._reset_provider_if_needed(provider)
            if not self.providers[provider]['enabled']:
                return []

        try:
            # URL com parâmetros otimizados
            search_url = f"https://www.bing.com/search"
            params = {
                'q': query,
                'cc': 'br',
                'setlang': 'pt-br',
                'count': max_results,
                'first': 1,
                'FORM': 'PERE'
            }

            headers = self._get_headers('bing')

            # Adiciona delay para evitar detecção
            time.sleep(random.uniform(1.0, 2.0))

            self._record_request(provider)

            response = requests.get(
                search_url,
                params=params,
                headers=headers,
                timeout=self.request_timeout,
                allow_redirects=True
            )

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                results = []

                # Múltiplos seletores para robustez
                result_selectors = [
                    'li.b_algo',
                    '.b_algo',
                    'li[class*="algo"]',
                    '.b_webResult'
                ]

                result_items = []
                for selector in result_selectors:
                    items = soup.select(selector)
                    if items:
                        result_items = items
                        break

                logger.info(f"🔍 Bing encontrou {len(result_items)} elementos")

                for item in result_items[:max_results]:
                    try:
                        # Múltiplos seletores para título
                        title_elem = (item.select_one('h2 a') or 
                                    item.select_one('h2') or 
                                    item.select_one('.b_title a') or
                                    item.select_one('a[href]'))

                        if not title_elem:
                            continue

                        title = title_elem.get_text(strip=True)
                        url = title_elem.get('href', '')

                        # Múltiplos seletores para snippet
                        snippet_elem = (item.select_one('.b_caption p') or
                                      item.select_one('p') or
                                      item.select_one('.b_snippet') or
                                      item.select_one('[class*="caption"]'))

                        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                        # Valida URL
                        if url and title and url.startswith('http'):
                            result = SearchResult(
                                title=title,
                                url=url,
                                snippet=snippet,
                                source='bing_scraping'
                            )
                            results.append(result)

                    except Exception as e:
                        logger.debug(f"Erro ao processar item Bing: {e}")
                        continue

                logger.info(f"✅ Bing Scraping: {len(results)} resultados válidos")
                return results

            elif response.status_code == 429:
                logger.warning("⚠️ Bing: Rate limit detectado")
                time.sleep(5)
                return []

            else:
                logger.warning(f"⚠️ Bing retornou status {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"❌ Erro Bing Scraping: {e}")
            self._handle_provider_error(provider, e)
            return []

    def search_duckduckgo_scraping(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Busca DuckDuckGo via scraping robusto com anti-detecção"""
        provider = 'duckduckgo'

        # DESABILITA DUCKDUCKGO TEMPORARIAMENTE - RETORNA STATUS 202
        logger.warning("⚠️ DuckDuckGo temporariamente desabilitado devido a problemas de API")
        return []

        try:
            # DuckDuckGo requer abordagem em duas etapas
            # 1. Primeira requisição para obter token
            session = requests.Session()
            session.headers.update(self._get_headers('duckduckgo'))

            # Delay anti-detecção
            time.sleep(random.uniform(1.5, 3.0))

            # Primeira requisição
            initial_url = "https://duckduckgo.com/"
            session.get(initial_url, timeout=self.request_timeout)

            # Segunda requisição com busca
            search_url = "https://html.duckduckgo.com/html/"
            params = {
                'q': query,
                'b': '',
                'kl': 'br-pt',
                'df': 'm'
            }

            self._record_request(provider)

            response = session.get(
                search_url,
                params=params,
                timeout=self.request_timeout
            )

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                results = []

                # Múltiplos seletores para robustez
                result_selectors = [
                    '.result',
                    'div[class*="result"]',
                    '.web-result',
                    '.results_links'
                ]

                result_items = []
                for selector in result_selectors:
                    items = soup.select(selector)
                    if items:
                        result_items = items
                        break

                logger.info(f"🔍 DuckDuckGo encontrou {len(result_items)} elementos")

                for item in result_items[:max_results]:
                    try:
                        # Múltiplos seletores para título e URL
                        title_elem = (item.select_one('.result__a') or
                                    item.select_one('a.result__title') or
                                    item.select_one('h2 a') or
                                    item.select_one('a[href*="uddg"]'))

                        if not title_elem:
                            continue

                        title = title_elem.get_text(strip=True)
                        url = title_elem.get('href', '')

                        # DuckDuckGo usa URLs redirecionadas
                        if url.startswith('/l/?uddg='):
                            # Extrai URL real do parâmetro
                            import urllib.parse
                            parsed = urllib.parse.parse_qs(url.split('?')[1])
                            if 'uddg' in parsed:
                                url = urllib.parse.unquote(parsed['uddg'][0])

                        # Snippet
                        snippet_elem = (item.select_one('.result__snippet') or
                                      item.select_one('.snippet') or
                                      item.select_one('p'))

                        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                        # Valida URL
                        if url and title and url.startswith('http'):
                            result = SearchResult(
                                title=title,
                                url=url,
                                snippet=snippet,
                                source='duckduckgo_scraping'
                            )
                            results.append(result)

                    except Exception as e:
                        logger.debug(f"Erro ao processar item DuckDuckGo: {e}")
                        continue

                logger.info(f"✅ DuckDuckGo Scraping: {len(results)} resultados válidos")
                return results

            elif response.status_code == 202:
                logger.warning("⚠️ DuckDuckGo: Busca em processamento (202)")
                return []

            else:
                logger.warning(f"⚠️ DuckDuckGo retornou status {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"❌ Erro DuckDuckGo Scraping: {e}")
            self._handle_provider_error(provider, e)
            return []

    def search_with_fallback(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Busca com sistema de fallback robusto"""

        # Verifica cache primeiro
        cached_results = self.cache.get(query, "combined")
        if cached_results:
            logger.info(f"📦 Usando resultados do cache para: {query[:50]}...")
            return cached_results

        all_results = []
        successful_providers = []

        # Ordena provedores por prioridade e disponibilidade
        available_providers = [
            (name, config) for name, config in self.providers.items()
            if config['enabled'] and config['error_count'] < 5 and name != 'duckduckgo'  # Exclui DuckDuckGo
        ]
        available_providers.sort(key=lambda x: x[1]['priority'])

        # Executa busca em paralelo para otimização
        with ThreadPoolExecutor(max_workers=3) as executor:  # Reduz workers
            future_to_provider = {}

            for provider_name, config in available_providers:
                if provider_name == 'google':
                    future = executor.submit(self.search_google_custom, query, max_results // 2)
                elif provider_name == 'serper':
                    future = executor.submit(self.search_serper, query, max_results // 2)
                elif provider_name == 'bing':
                    future = executor.submit(self.search_bing_scraping, query, max_results // 2)
                # DuckDuckGo removido temporariamente
                else:
                    continue

                future_to_provider[future] = provider_name

            # Coleta resultados conforme completam
            for future in as_completed(future_to_provider, timeout=60):
                provider_name = future_to_provider[future]
                try:
                    results = future.result()
                    if results:
                        all_results.extend(results)
                        successful_providers.append(provider_name)
                        logger.info(f"✅ {provider_name}: {len(results)} resultados")
                    else:
                        logger.warning(f"⚠️ {provider_name}: 0 resultados")

                except Exception as e:
                    logger.error(f"❌ Erro em {provider_name}: {e}")
                    self._handle_provider_error(provider_name, e)

        # Remove duplicatas baseado na URL
        unique_results = []
        seen_urls = set()

        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)

        # Ordena por relevância (pode ser implementado)
        unique_results.sort(key=lambda x: x.timestamp, reverse=True)

        # Limita resultados
        final_results = unique_results[:max_results]

        # Converte SearchResult para dict se necessário
        dict_results = []
        for result in final_results:
            if hasattr(result, '__dict__'):
                dict_results.append({
                    'title': result.title,
                    'url': result.url,
                    'snippet': result.snippet,
                    'source': result.source,
                    'relevance_score': getattr(result, 'relevance_score', 0.0),
                    'timestamp': result.timestamp.isoformat() if hasattr(result, 'timestamp') and result.timestamp else datetime.now().isoformat()
                })
            else:
                dict_results.append(result)

        # Salva no cache se obteve resultados
        if dict_results:
            # Converte para SearchResult para cache
            cache_results = []
            for result_dict in dict_results:
                cache_result = SearchResult(
                    title=result_dict['title'],
                    url=result_dict['url'],
                    snippet=result_dict['snippet'],
                    source=result_dict['source']
                )
                cache_results.append(cache_result)
            self.cache.set(query, cache_results, "combined")

        logger.info(f"🎯 Busca final: {len(dict_results)} resultados únicos de {len(successful_providers)} provedores")

        # Limpeza periódica do cache
        if time.time() - self.last_cleanup > 3600:  # 1 hora
            self.cache.cleanup_expired()
            self.last_cleanup = time.time()

        return dict_results

    def get_provider_status(self) -> Dict[str, Any]:
        """Retorna status detalhado dos provedores"""
        status = {}

        for name, config in self.providers.items():
            status[name] = {
                'enabled': config['enabled'],
                'priority': config['priority'],
                'error_count': config['error_count'],
                'last_error': config.get('last_error'),
                'rate_limited': (config.get('quota_reset') or 0) > time.time(),
                'requests_today': len(self.rate_limiter.get(name, [])),
                'rate_limit': config['rate_limit']
            }

        return status

    def reset_provider_errors(self, provider_name: str = None):
        """Reset contadores de erro        """
        if provider_name:
            if provider_name in self.providers:
                self.providers[provider_name]['error_count'] = 0
                self.providers[provider_name]['enabled'] = True
                self.providers[provider_name]['quota_reset'] = None
                self.error_counts[provider_name] = 0
                logger.info(f"🔄 Reset erros do provedor: {provider_name}")
        else:
            for name in self.providers:
                self.providers[name]['error_count'] = 0
                self.providers[name]['enabled'] = True
                self.providers[name]['quota_reset'] = None
                self.error_counts[name] = 0
            logger.info("🔄 Reset erros de todos os provedores")

    def clear_cache(self):
        """Limpa todo o cache"""
        try:
            if os.path.exists(self.cache.db_path):
                with sqlite3.connect(self.cache.db_path) as conn:
                    conn.execute("DELETE FROM search_cache")
                    conn.commit()
                logger.info("🗑️ Cache limpo completamente")
        except Exception as e:
            logger.error(f"Erro ao limpar cache: {e}")

# Instância global para produção
production_search_manager = ProductionSearchManager()