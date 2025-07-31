
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v2.0 - Robust Content Extractor
Extrator multicamadas sem dependências pagas
"""

import os
import logging
import time
import requests
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urljoin, urlparse
import re

# Imports condicionais para não quebrar se não estiver instalado
try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

try:
    from readability import Document
    HAS_READABILITY = True
except ImportError:
    HAS_READABILITY = False

try:
    import newspaper
    from newspaper import Article
    HAS_NEWSPAPER = True
except ImportError:
    HAS_NEWSPAPER = False

try:
    from bs4 import BeautifulSoup
    HAS_BEAUTIFULSOUP = True
except ImportError:
    HAS_BEAUTIFULSOUP = False

from services.url_resolver import url_resolver

logger = logging.getLogger(__name__)

class RobustContentExtractor:
    """Extrator de conteúdo multicamadas e robusto"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'pt-BR,pt;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        self.timeout = 30
        self.min_content_length = 500
        self.max_content_length = 50000  # 50K chars max
        
        # Estatísticas dos extratores
        self.stats = {
            'trafilatura': {'success': 0, 'failed': 0, 'total_time': 0},
            'readability': {'success': 0, 'failed': 0, 'total_time': 0},
            'newspaper': {'success': 0, 'failed': 0, 'total_time': 0},
            'beautifulsoup': {'success': 0, 'failed': 0, 'total_time': 0},
            'total_extractions': 0,
            'successful_extractions': 0
        }
        
        logger.info("🔧 Robust Content Extractor inicializado")
        logger.info(f"📚 Extratores disponíveis: {self._get_available_extractors()}")
    
    def extract_content(self, url: str) -> Optional[str]:
        """
        Extrai conteúdo usando múltiplos extratores em ordem de prioridade
        
        Returns:
            str: Conteúdo extraído (mínimo 500 chars) ou None se falhar
        """
        try:
            start_time = time.time()
            self.stats['total_extractions'] += 1
            
            # 1. Resolve URL de redirecionamento
            resolved_url = url_resolver.resolve_redirect_url(url)
            if resolved_url != url:
                logger.info(f"🔄 URL resolvida: {url} -> {resolved_url}")
                url = resolved_url
            
            # 2. Baixa conteúdo HTML
            html_content = self._fetch_html(url)
            if not html_content:
                logger.error(f"❌ Falha ao baixar HTML para {url}")
                return None
            
            logger.info(f"📥 HTML baixado: {len(html_content)} caracteres")
            
            # 3. Tenta extratores em ordem de prioridade
            extractors = [
                ('trafilatura', self._extract_with_trafilatura),
                ('readability', self._extract_with_readability),
                ('newspaper', self._extract_with_newspaper),
                ('beautifulsoup', self._extract_with_beautifulsoup)
            ]
            
            for extractor_name, extractor_func in extractors:
                if not self._is_extractor_available(extractor_name):
                    continue
                
                try:
                    logger.info(f"🔍 Tentando extração com {extractor_name}...")
                    content = extractor_func(html_content, url)
                    
                    if self._validate_content(content, url):
                        extraction_time = time.time() - start_time
                        self.stats[extractor_name]['success'] += 1
                        self.stats[extractor_name]['total_time'] += extraction_time
                        self.stats['successful_extractions'] += 1
                        
                        logger.info(f"✅ Extração bem-sucedida com {extractor_name}: {len(content)} caracteres em {extraction_time:.2f}s")
                        return content
                    else:
                        self.stats[extractor_name]['failed'] += 1
                        logger.warning(f"⚠️ Conteúdo insuficiente com {extractor_name}: {len(content) if content else 0} caracteres")
                        
                except Exception as e:
                    self.stats[extractor_name]['failed'] += 1
                    logger.error(f"❌ Erro com {extractor_name}: {str(e)}")
                    continue
            
            # Todos os extratores falharam
            logger.error(f"❌ FALHA CRÍTICA: Todos os extratores falharam para {url}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Erro crítico na extração de {url}: {str(e)}")
            return None
    
    def _fetch_html(self, url: str) -> Optional[str]:
        """Baixa conteúdo HTML da URL"""
        try:
            response = self.session.get(
                url,
                timeout=self.timeout,
                verify=False,  # Para evitar problemas de SSL
                allow_redirects=True
            )
            
            response.raise_for_status()
            
            # Detecta encoding
            if response.encoding is None:
                response.encoding = 'utf-8'
            
            html = response.text
            
            if len(html) < 1000:
                logger.warning(f"⚠️ HTML muito pequeno: {len(html)} caracteres")
                return None
            
            return html
            
        except Exception as e:
            logger.error(f"❌ Erro ao baixar {url}: {str(e)}")
            return None
    
    def _extract_with_trafilatura(self, html: str, url: str) -> Optional[str]:
        """Extrai com Trafilatura (prioridade 1)"""
        if not HAS_TRAFILATURA:
            return None
        
        try:
            content = trafilatura.extract(
                html, 
                include_comments=False,
                include_tables=True,
                include_formatting=False,
                favor_precision=True,
                url=url
            )
            
            if content:
                # Limpa conteúdo
                content = self._clean_content(content)
                return content
            
            return None
            
        except Exception as e:
            logger.error(f"Erro Trafilatura: {e}")
            return None
    
    def _extract_with_readability(self, html: str, url: str) -> Optional[str]:
        """Extrai com Readability (prioridade 2)"""
        if not HAS_READABILITY:
            return None
        
        try:
            doc = Document(html)
            content = doc.summary()
            
            if content:
                # Remove tags HTML
                if HAS_BEAUTIFULSOUP:
                    soup = BeautifulSoup(content, 'html.parser')
                    content = soup.get_text()
                else:
                    # Remove tags manualmente
                    content = re.sub(r'<[^>]+>', '', content)
                
                content = self._clean_content(content)
                return content
            
            return None
            
        except Exception as e:
            logger.error(f"Erro Readability: {e}")
            return None
    
    def _extract_with_newspaper(self, html: str, url: str) -> Optional[str]:
        """Extrai com Newspaper3k (prioridade 3)"""
        if not HAS_NEWSPAPER:
            return None
        
        try:
            article = Article(url)
            article.set_html(html)
            article.parse()
            
            content = article.text
            if content:
                content = self._clean_content(content)
                return content
            
            return None
            
        except Exception as e:
            logger.error(f"Erro Newspaper: {e}")
            return None
    
    def _extract_with_beautifulsoup(self, html: str, url: str) -> Optional[str]:
        """Extrai com BeautifulSoup (fallback final)"""
        if not HAS_BEAUTIFULSOUP:
            return None
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove scripts e styles
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            # Procura pelo conteúdo principal
            main_content = None
            
            # Tenta encontrar containers de conteúdo
            content_selectors = [
                'article', 'main', '.content', '#content', '.post', '.article',
                '.entry', '.text', '.body', '.container', 'div[role="main"]'
            ]
            
            for selector in content_selectors:
                try:
                    element = soup.select_one(selector)
                    if element:
                        main_content = element.get_text()
                        break
                except:
                    continue
            
            # Se não encontrou, pega o body
            if not main_content:
                body = soup.find('body')
                if body:
                    main_content = body.get_text()
                else:
                    main_content = soup.get_text()
            
            if main_content:
                content = self._clean_content(main_content)
                return content
            
            return None
            
        except Exception as e:
            logger.error(f"Erro BeautifulSoup: {e}")
            return None
    
    def _clean_content(self, content: str) -> str:
        """Limpa e normaliza o conteúdo extraído"""
        if not content:
            return ""
        
        # Remove quebras de linha excessivas
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        # Remove espaços excessivos
        content = re.sub(r'[ \t]+', ' ', content)
        
        # Remove caracteres de controle
        content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
        
        # Normaliza
        content = content.strip()
        
        # Limita tamanho
        if len(content) > self.max_content_length:
            content = content[:self.max_content_length] + "..."
        
        return content
    
    def _validate_content(self, content: str, url: str) -> bool:
        """Valida se o conteúdo extraído é válido"""
        if not content:
            return False
        
        # Verifica tamanho mínimo
        if len(content) < self.min_content_length:
            logger.warning(f"⚠️ Conteúdo muito pequeno para {url}: {len(content)} < {self.min_content_length}")
            return False
        
        # Verifica se não é só lixo
        words = content.split()
        if len(words) < 50:  # Mínimo 50 palavras
            logger.warning(f"⚠️ Muito poucas palavras para {url}: {len(words)}")
            return False
        
        # Verifica se tem conteúdo real (não só navegação)
        common_words = ['o', 'a', 'de', 'da', 'do', 'e', 'em', 'um', 'uma', 'com', 'não', 'para', 'que', 'se']
        real_words = sum(1 for word in words if any(common in word.lower() for common in common_words))
        
        if real_words / len(words) < 0.1:  # Pelo menos 10% de palavras comuns
            logger.warning(f"⚠️ Conteúdo suspeito para {url}: poucos conectivos")
            return False
        
        logger.info(f"✅ Conteúdo válido para {url}: {len(content)} caracteres, {len(words)} palavras")
        return True
    
    def _is_extractor_available(self, extractor_name: str) -> bool:
        """Verifica se o extrator está disponível"""
        availability = {
            'trafilatura': HAS_TRAFILATURA,
            'readability': HAS_READABILITY,
            'newspaper': HAS_NEWSPAPER,
            'beautifulsoup': HAS_BEAUTIFULSOUP
        }
        return availability.get(extractor_name, False)
    
    def _get_available_extractors(self) -> List[str]:
        """Retorna lista de extratores disponíveis"""
        available = []
        if HAS_TRAFILATURA:
            available.append('trafilatura')
        if HAS_READABILITY:
            available.append('readability')
        if HAS_NEWSPAPER:
            available.append('newspaper')
        if HAS_BEAUTIFULSOUP:
            available.append('beautifulsoup')
        return available
    
    def get_extractor_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas dos extratores"""
        stats_copy = self.stats.copy()
        
        # Calcula percentuais de sucesso
        for extractor in ['trafilatura', 'readability', 'newspaper', 'beautifulsoup']:
            total = stats_copy[extractor]['success'] + stats_copy[extractor]['failed']
            if total > 0:
                stats_copy[extractor]['success_rate'] = (stats_copy[extractor]['success'] / total) * 100
                if stats_copy[extractor]['success'] > 0:
                    stats_copy[extractor]['avg_time'] = stats_copy[extractor]['total_time'] / stats_copy[extractor]['success']
                else:
                    stats_copy[extractor]['avg_time'] = 0
            else:
                stats_copy[extractor]['success_rate'] = 0
                stats_copy[extractor]['avg_time'] = 0
        
        # Taxa geral de sucesso
        if stats_copy['total_extractions'] > 0:
            stats_copy['overall_success_rate'] = (stats_copy['successful_extractions'] / stats_copy['total_extractions']) * 100
        else:
            stats_copy['overall_success_rate'] = 0
        
        return stats_copy
    
    def reset_extractor_stats(self, extractor_name: Optional[str] = None):
        """Reset estatísticas dos extratores"""
        if extractor_name and extractor_name in self.stats:
            self.stats[extractor_name] = {'success': 0, 'failed': 0, 'total_time': 0}
            logger.info(f"🔄 Reset estatísticas do extrator: {extractor_name}")
        else:
            # Reset todas
            for extractor in ['trafilatura', 'readability', 'newspaper', 'beautifulsoup']:
                self.stats[extractor] = {'success': 0, 'failed': 0, 'total_time': 0}
            self.stats['total_extractions'] = 0
            self.stats['successful_extractions'] = 0
            logger.info("🔄 Reset estatísticas de todos os extratores")
    
    def clear_cache(self):
        """Limpa cache de sessão"""
        self.session.close()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        logger.info("🧹 Cache de extração limpo")

# Instância global
robust_content_extractor = RobustContentExtractor()
