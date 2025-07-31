#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v2.0 - Content Extractor
Extrator de conteúdo robusto com múltiplas estratégias
"""

import os
import logging
import time
import requests
from typing import Optional, Dict, Any
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import re

logger = logging.getLogger(__name__)

class ContentExtractor:
    """Extrator de conteúdo com múltiplas estratégias"""
    
    def __init__(self):
        """Inicializa o extrator de conteúdo"""
        self.jina_api_key = os.getenv('JINA_API_KEY')
        self.jina_reader_url = "https://r.jina.ai/"
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'pt-BR,pt;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        self.extraction_strategies = [
            'jina_reader',
            'direct_extraction',
            'readability_extraction',
            'fallback_extraction'
        ]
        
        logger.info("Content Extractor inicializado com múltiplas estratégias")
    
    def extract_content(self, url: str) -> Optional[str]:
        """Extrai conteúdo usando múltiplas estratégias"""
        
        if not url or not url.startswith('http'):
            return None
        
        logger.info(f"🔍 Extraindo conteúdo de: {url}")
        
        # Tenta cada estratégia em ordem de prioridade
        for strategy in self.extraction_strategies:
            try:
                if strategy == 'jina_reader' and self.jina_api_key:
                    content = self._extract_with_jina(url)
                elif strategy == 'direct_extraction':
                    content = self._extract_direct(url)
                elif strategy == 'readability_extraction':
                    content = self._extract_with_readability(url)
                elif strategy == 'fallback_extraction':
                    content = self._extract_fallback(url)
                else:
                    continue
                
                if content and len(content) > 100:  # Conteúdo substancial
                    logger.info(f"✅ Conteúdo extraído com {strategy}: {len(content)} caracteres")
                    return content
                    
            except Exception as e:
                logger.warning(f"⚠️ Estratégia {strategy} falhou para {url}: {str(e)}")
                continue
        
        logger.error(f"❌ Todas as estratégias falharam para {url}")
        return None
    
    def _extract_with_jina(self, url: str) -> Optional[str]:
        """Extrai conteúdo usando Jina Reader API"""
        try:
            headers = {
                **self.headers,
                "Authorization": f"Bearer {self.jina_api_key}"
            }
            
            jina_url = f"{self.jina_reader_url}{url}"
            
            response = requests.get(
                jina_url,
                headers=headers,
                timeout=60
            )
            
            if response.status_code == 200:
                content = response.text
                
                # Limita tamanho para otimização
                if len(content) > 15000:
                    content = content[:15000] + "... [conteúdo truncado para otimização]"
                
                return content
            else:
                raise Exception(f"Jina Reader retornou status {response.status_code}")
                
        except Exception as e:
            raise e
    
    def _extract_direct(self, url: str) -> Optional[str]:
        """Extração direta usando BeautifulSoup"""
        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=20,
                allow_redirects=True
            )
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                
                # Remove elementos desnecessários
                for element in soup(["script", "style", "nav", "footer", "header", 
                                   "form", "aside", "iframe", "noscript", "advertisement",
                                   "ads", "sidebar", "menu", "breadcrumb"]):
                    element.decompose()
                
                # Busca conteúdo principal
                main_content = (
                    soup.find('main') or 
                    soup.find('article') or 
                    soup.find('div', class_=re.compile(r'content|main|article|post|entry|body')) or
                    soup.find('div', id=re.compile(r'content|main|article|post|entry|body')) or
                    soup.find('section', class_=re.compile(r'content|main|article|post|entry'))
                )
                
                if main_content:
                    text = main_content.get_text()
                else:
                    # Fallback para body completo
                    body = soup.find('body')
                    text = body.get_text() if body else soup.get_text()
                
                # Limpa o texto
                text = self._clean_text(text)
                
                return text
            else:
                raise Exception(f"Resposta HTTP {response.status_code}")
                
        except Exception as e:
            raise e
    
    def _extract_with_readability(self, url: str) -> Optional[str]:
        """Extração usando algoritmo de readability"""
        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=20,
                allow_redirects=True
            )
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                
                # Remove elementos desnecessários
                for element in soup(["script", "style", "nav", "footer", "header", 
                                   "form", "aside", "iframe", "noscript"]):
                    element.decompose()
                
                # Algoritmo simples de readability
                # Busca por elementos com mais texto
                candidates = []
                
                for element in soup.find_all(['div', 'article', 'section', 'main']):
                    text = element.get_text()
                    if len(text) > 200:  # Elementos com conteúdo substancial
                        # Score baseado em tamanho e densidade de parágrafos
                        paragraphs = element.find_all('p')
                        score = len(text) + (len(paragraphs) * 50)
                        candidates.append((score, text))
                
                if candidates:
                    # Pega o elemento com maior score
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    text = candidates[0][1]
                    
                    # Limpa o texto
                    text = self._clean_text(text)
                    
                    return text
                else:
                    raise Exception("Nenhum conteúdo substancial encontrado")
            else:
                raise Exception(f"Resposta HTTP {response.status_code}")
                
        except Exception as e:
            raise e
    
    def _extract_fallback(self, url: str) -> Optional[str]:
        """Extração de fallback mais agressiva"""
        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=15,
                allow_redirects=True
            )
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                
                # Remove apenas elementos críticos
                for element in soup(["script", "style", "noscript"]):
                    element.decompose()
                
                # Pega todo o texto disponível
                text = soup.get_text()
                
                # Limpa o texto
                text = self._clean_text(text)
                
                # Se ainda tem conteúdo substancial, retorna
                if len(text) > 100:
                    return text
                else:
                    raise Exception("Conteúdo insuficiente após limpeza")
            else:
                raise Exception(f"Resposta HTTP {response.status_code}")
                
        except Exception as e:
            raise e
    
    def _clean_text(self, text: str) -> str:
        """Limpa e normaliza o texto extraído"""
        if not text:
            return ""
        
        # Remove quebras de linha excessivas
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove espaços excessivos
        text = re.sub(r' +', ' ', text)
        
        # Remove caracteres especiais problemáticos
        text = re.sub(r'[^\w\s\.,;:!?\-\(\)%$€£¥\n]', '', text)
        
        # Quebra em linhas
        lines = (line.strip() for line in text.splitlines())
        
        # Remove linhas muito curtas (provavelmente menu/navegação)
        meaningful_lines = []
        for line in lines:
            if len(line) > 10:  # Linhas com pelo menos 10 caracteres
                meaningful_lines.append(line)
        
        # Junta linhas significativas
        cleaned_text = '\n'.join(meaningful_lines)
        
        # Limita tamanho final
        if len(cleaned_text) > 12000:
            cleaned_text = cleaned_text[:12000] + "... [conteúdo truncado para otimização]"
        
        return cleaned_text.strip()
    
    def extract_metadata(self, url: str) -> Dict[str, Any]:
        """Extrai metadados da página"""
        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=15,
                allow_redirects=True
            )
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                
                metadata = {
                    'title': '',
                    'description': '',
                    'keywords': '',
                    'author': '',
                    'published_date': '',
                    'language': '',
                    'canonical_url': url
                }
                
                # Título
                title_tag = soup.find('title')
                if title_tag:
                    metadata['title'] = title_tag.get_text().strip()
                
                # Meta tags
                meta_tags = soup.find_all('meta')
                for tag in meta_tags:
                    name = tag.get('name', '').lower()
                    property_attr = tag.get('property', '').lower()
                    content = tag.get('content', '')
                    
                    if name == 'description' or property_attr == 'og:description':
                        metadata['description'] = content
                    elif name == 'keywords':
                        metadata['keywords'] = content
                    elif name == 'author':
                        metadata['author'] = content
                    elif name == 'language' or name == 'lang':
                        metadata['language'] = content
                    elif property_attr == 'article:published_time':
                        metadata['published_date'] = content
                
                # URL canônica
                canonical = soup.find('link', rel='canonical')
                if canonical:
                    metadata['canonical_url'] = canonical.get('href', url)
                
                return metadata
            else:
                return {'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            return {'error': str(e)}
    
    def is_content_relevant(self, content: str, keywords: list) -> bool:
        """Verifica se o conteúdo é relevante baseado em palavras-chave"""
        if not content or not keywords:
            return False
        
        content_lower = content.lower()
        
        # Conta quantas palavras-chave aparecem
        matches = 0
        for keyword in keywords:
            if keyword.lower() in content_lower:
                matches += 1
        
        # Considera relevante se pelo menos 30% das palavras-chave aparecem
        relevance_threshold = len(keywords) * 0.3
        return matches >= relevance_threshold
    
    def extract_links(self, url: str, internal_only: bool = True) -> list:
        """Extrai links da página"""
        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=15,
                allow_redirects=True
            )
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                base_domain = urlparse(url).netloc
                
                links = []
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    full_url = urljoin(url, href)
                    
                    # Filtra apenas links internos se solicitado
                    if internal_only:
                        link_domain = urlparse(full_url).netloc
                        if link_domain != base_domain:
                            continue
                    
                    # Filtra links válidos
                    if (full_url.startswith('http') and 
                        '#' not in full_url and 
                        not any(ext in full_url.lower() for ext in ['.pdf', '.jpg', '.png', '.gif', '.zip'])):
                        links.append({
                            'url': full_url,
                            'text': a_tag.get_text().strip()[:100],
                            'title': a_tag.get('title', '')
                        })
                
                return links[:20]  # Máximo 20 links
            else:
                return []
                
        except Exception as e:
            logger.error(f"Erro ao extrair links de {url}: {str(e)}")
            return []

# Instância global
content_extractor = ContentExtractor()
