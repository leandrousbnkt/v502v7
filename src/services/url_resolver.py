#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v2.0 - URL Resolver
Resolve URLs de redirecionamento do Bing, Google e outros serviços
"""
import base64
import logging
from urllib.parse import parse_qs, urlparse, unquote
from typing import Optional

logger = logging.getLogger(__name__)


class URLResolver:
    """Classe para resolução de URLs de redirecionamento"""

    @staticmethod
    def resolve_url(url: str) -> str:
        """Resolve URLs de redirecionamento"""
        try:
            # Bing redirect URLs
            if "bing.com/ck/a?!" in url:
                return URLResolver._resolve_bing_url(url)

            # Google redirect URLs
            elif "/url?q=" in url:
                return URLResolver._resolve_google_url(url)

            # Yahoo redirect URLs
            elif "r.search.yahoo.com" in url:
                return URLResolver._resolve_yahoo_url(url)

            # DuckDuckGo redirect URLs
            elif "duckduckgo.com/l/" in url:
                return URLResolver._resolve_duckduckgo_url(url)

            # URL já é direta
            return url

        except Exception as e:
            logger.warning(f"⚠️ Erro ao resolver URL: {str(e)}")
            return url

    @staticmethod
    def _resolve_bing_url(url: str) -> str:
        """Resolve URLs de redirecionamento do Bing"""
        try:
            parsed = urlparse(url)
            query_params = parse_qs(parsed.query)

            if 'u' in query_params:
                encoded = query_params['u'][0]

                # Remove prefixo 'a1' se presente
                if encoded.startswith("a1"):
                    encoded = encoded[2:]

                # Adiciona padding se necessário
                missing_padding = len(encoded) % 4
                if missing_padding:
                    encoded += '=' * (4 - missing_padding)

                try:
                    # Primeira decodificação Base64
                    first_decode = base64.b64decode(encoded)

                    # Segunda decodificação Base64
                    missing_padding = len(first_decode) % 4
                    if missing_padding:
                        first_decode += b'=' * (4 - missing_padding)

                    decoded_url = base64.b64decode(first_decode).decode('utf-8')

                    logger.info(f"🔗 URL Bing resolvida: {url[:50]}... → {decoded_url[:50]}...")
                    return decoded_url

                except Exception as decode_error:
                    logger.warning(f"⚠️ Falha na decodificação Bing: {str(decode_error)}")
                    # Tenta decodificação simples
                    try:
                        simple_decode = base64.b64decode(encoded).decode('utf-8')
                        return simple_decode
                    except:
                        pass

        except Exception as e:
            logger.warning(f"⚠️ Erro ao resolver URL Bing: {str(e)}")

        return url

    @staticmethod
    def _resolve_google_url(url: str) -> str:
        """Resolve URLs de redirecionamento do Google"""
        try:
            parsed = urlparse(url)
            query_params = parse_qs(parsed.query)

            if 'q' in query_params:
                resolved = unquote(query_params['q'][0])
                logger.info(f"🔗 URL Google resolvida: {url[:50]}... → {resolved[:50]}...")
                return resolved

        except Exception as e:
            logger.warning(f"⚠️ Erro ao resolver URL Google: {str(e)}")

        return url

    @staticmethod
    def _resolve_yahoo_url(url: str) -> str:
        """Resolve URLs de redirecionamento do Yahoo"""
        try:
            parsed = urlparse(url)
            query_params = parse_qs(parsed.query)

            if 'RU' in query_params:
                encoded_url = query_params['RU'][0]
                resolved = unquote(encoded_url)
                logger.info(f"🔗 URL Yahoo resolvida: {url[:50]}... → {resolved[:50]}...")
                return resolved

        except Exception as e:
            logger.warning(f"⚠️ Erro ao resolver URL Yahoo: {str(e)}")

        return url

    @staticmethod
    def _resolve_duckduckgo_url(url: str) -> str:
        """Resolve URLs de redirecionamento do DuckDuckGo"""
        try:
            # DuckDuckGo usa formato: https://duckduckgo.com/l/?uddg=...
            parsed = urlparse(url)
            query_params = parse_qs(parsed.query)

            if 'uddg' in query_params:
                encoded_url = query_params['uddg'][0]
                resolved = unquote(encoded_url)
                logger.info(f"🔗 URL DuckDuckGo resolvida: {url[:50]}... → {resolved[:50]}...")
                return resolved

        except Exception as e:
            logger.warning(f"⚠️ Erro ao resolver URL DuckDuckGo: {str(e)}")

        return url


# Função de conveniência
def resolve_url(url: str) -> str:
    """Função de conveniência para resolver URLs"""
    return URLResolver.resolve_url(url)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v2.0 - URL Resolver
Resolve URLs de redirecionamento e encurtadores
"""

import os
import logging
import base64
import requests
from urllib.parse import parse_qs, urlparse, unquote
from typing import Optional

logger = logging.getLogger(__name__)

class URLResolver:
    """Resolvedor robusto de URLs de redirecionamento"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        self.timeout = 10
        
    def resolve_redirect_url(self, url: str) -> str:
        """
        Resolve URLs de redirecionamento do Bing, Google e encurtadores.
        """
        try:
            original_url = url
            
            # Bing: URLs com u=a1aHR0c...
            if "bing.com/ck/a" in url and "u=a1" in url:
                logger.info(f"🔄 Resolvendo URL do Bing: {url[:100]}...")
                resolved = self._resolve_bing_url(url)
                if resolved and resolved != url:
                    logger.info(f"✅ URL Bing resolvida: {resolved}")
                    return resolved
            
            # Google: URLs com /url?q=
            elif "/url?q=" in url or "google." in url and "url?q=" in url:
                logger.info(f"🔄 Resolvendo URL do Google: {url[:100]}...")
                resolved = self._resolve_google_url(url)
                if resolved and resolved != url:
                    logger.info(f"✅ URL Google resolvida: {resolved}")
                    return resolved
            
            # Encurtadores conhecidos
            elif self._is_short_url(url):
                logger.info(f"🔄 Resolvendo URL encurtada: {url}")
                resolved = self._resolve_short_url(url)
                if resolved and resolved != url:
                    logger.info(f"✅ URL encurtada resolvida: {resolved}")
                    return resolved
            
            # URL já está limpa
            return url
            
        except Exception as e:
            logger.error(f"❌ Erro ao resolver URL {url}: {str(e)}")
            return url  # Retorna a original se falhar
    
    def _resolve_bing_url(self, url: str) -> str:
        """Resolve URLs específicas do Bing"""
        try:
            # Extrai parâmetro u=
            if "u=a1" in url:
                # Formato: u=a1aHR0c...
                u_param = url.split("u=a1")[1].split("&")[0]
                
                # Decodifica base64 duplo
                try:
                    # Primeiro decode
                    decoded_once = base64.b64decode(u_param + "==").decode('utf-8', errors='ignore')
                    # Segundo decode se necessário
                    if decoded_once.startswith('aHR0'):
                        decoded_twice = base64.b64decode(decoded_once + "==").decode('utf-8', errors='ignore')
                        if decoded_twice.startswith('http'):
                            return decoded_twice
                    elif decoded_once.startswith('http'):
                        return decoded_once
                except:
                    pass
            
            # Método alternativo: follow redirects
            return self._follow_redirects(url)
            
        except Exception as e:
            logger.error(f"Erro ao resolver Bing URL: {e}")
            return url
    
    def _resolve_google_url(self, url: str) -> str:
        """Resolve URLs do Google"""
        try:
            parsed = urlparse(url)
            
            if "url?q=" in url:
                # Extrai parâmetro q
                query_params = parse_qs(parsed.query)
                q_param = query_params.get('q')
                if q_param:
                    decoded_url = unquote(q_param[0])
                    if decoded_url.startswith('http'):
                        return decoded_url
            
            # Follow redirects se não conseguir extrair
            return self._follow_redirects(url)
            
        except Exception as e:
            logger.error(f"Erro ao resolver Google URL: {e}")
            return url
    
    def _is_short_url(self, url: str) -> bool:
        """Verifica se é URL encurtada"""
        short_domains = [
            'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'short.link',
            'ow.ly', 'buff.ly', 'tiny.cc', 'is.gd', 'v.gd'
        ]
        return any(domain in url for domain in short_domains)
    
    def _resolve_short_url(self, url: str) -> str:
        """Resolve URLs encurtadas seguindo redirects"""
        return self._follow_redirects(url)
    
    def _follow_redirects(self, url: str, max_redirects: int = 5) -> str:
        """Segue redirects até a URL final"""
        try:
            response = self.session.head(
                url, 
                allow_redirects=True, 
                timeout=self.timeout,
                verify=False  # Para evitar problemas de SSL
            )
            
            final_url = response.url
            if final_url and final_url != url:
                logger.info(f"🔄 Redirect seguido: {url} -> {final_url}")
                return final_url
            
            return url
            
        except Exception as e:
            logger.warning(f"⚠️ Erro ao seguir redirects para {url}: {e}")
            return url

# Instância global
url_resolver = URLResolver()
