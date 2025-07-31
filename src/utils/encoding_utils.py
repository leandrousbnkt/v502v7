#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v2.0 - Encoding Utilities
Utilitários para garantir encoding UTF-8 correto em produção
"""

import sys
import locale
import logging
import chardet
from typing import Union, Optional

logger = logging.getLogger(__name__)

def setup_utf8_environment():
    """Configura ambiente para UTF-8"""
    try:
        # Força encoding UTF-8 no Python
        if hasattr(sys, 'set_int_max_str_digits'):
            sys.set_int_max_str_digits(0)
        
        # Configura locale
        try:
            locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
            logger.info("✅ Locale configurado para pt_BR.UTF-8")
        except locale.Error:
            try:
                locale.setlocale(locale.LC_ALL, 'C.UTF-8')
                logger.info("✅ Locale configurado para C.UTF-8")
            except locale.Error:
                logger.warning("⚠️ Não foi possível configurar locale UTF-8")
        
        # Força encoding de saída
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao configurar UTF-8: {e}")
        return False

def ensure_utf8_string(text: Union[str, bytes], fallback_encoding: str = 'latin-1') -> str:
    """Garante que o texto seja uma string UTF-8 válida"""
    if isinstance(text, str):
        # Já é string, verifica se é UTF-8 válido
        try:
            text.encode('utf-8')
            return text
        except UnicodeEncodeError:
            # String com caracteres inválidos, tenta limpar
            return text.encode('utf-8', errors='ignore').decode('utf-8')
    
    elif isinstance(text, bytes):
        # É bytes, precisa decodificar
        try:
            # Tenta UTF-8 primeiro
            return text.decode('utf-8')
        except UnicodeDecodeError:
            # Detecta encoding automaticamente
            detected = chardet.detect(text)
            encoding = detected.get('encoding', fallback_encoding)
            
            try:
                return text.decode(encoding, errors='ignore')
            except (UnicodeDecodeError, LookupError):
                # Fallback final
                return text.decode(fallback_encoding, errors='ignore')
    
    else:
        # Não é string nem bytes, converte para string
        return str(text)

def clean_text_encoding(text: str) -> str:
    """Limpa problemas de encoding em texto"""
    if not text:
        return ""
    
    # Correções comuns de encoding
    corrections = {
        'Ã¡': 'á',
        'Ã ': 'à',
        'Ã£': 'ã',
        'Ã©': 'é',
        'Ãª': 'ê',
        'Ã­': 'í',
        'Ã³': 'ó',
        'Ãµ': 'õ',
        'Ãº': 'ú',
        'Ã§': 'ç',
        'Ã‡': 'Ç',
        'Ã': 'Á',
        'Ã‰': 'É',
        'Ã"': 'Ó',
        'Ãš': 'Ú',
        'â€™': "'",
        'â€œ': '"',
        'â€': '"',
        'â€"': '–',
        'â€"': '—',
        'â€¢': '•',
        'Â': '',
        'â€¦': '...',
        'Ã¢': 'â',
        'Ã´': 'ô',
        'Ã»': 'û',
        'Ã¼': 'ü',
        'Ã±': 'ñ',
        'Ã¿': 'ÿ'
    }
    
    # Aplica correções
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    
    # Remove caracteres de controle
    import re
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    return text

def safe_json_dumps(data, ensure_ascii=False, **kwargs):
    """JSON dumps seguro com UTF-8"""
    import json
    
    try:
        return json.dumps(data, ensure_ascii=ensure_ascii, **kwargs)
    except UnicodeEncodeError:
        # Limpa dados recursivamente
        cleaned_data = clean_data_encoding(data)
        return json.dumps(cleaned_data, ensure_ascii=ensure_ascii, **kwargs)

def clean_data_encoding(data):
    """Limpa encoding em estruturas de dados recursivamente"""
    if isinstance(data, dict):
        return {key: clean_data_encoding(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [clean_data_encoding(item) for item in data]
    elif isinstance(data, str):
        return clean_text_encoding(data)
    elif isinstance(data, bytes):
        return ensure_utf8_string(data)
    else:
        return data

def validate_utf8_file(file_path: str) -> bool:
    """Valida se um arquivo está em UTF-8"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read()
        return True
    except UnicodeDecodeError:
        return False

def convert_file_to_utf8(input_path: str, output_path: str = None) -> bool:
    """Converte arquivo para UTF-8"""
    if output_path is None:
        output_path = input_path
    
    try:
        # Detecta encoding atual
        with open(input_path, 'rb') as f:
            raw_data = f.read()
        
        detected = chardet.detect(raw_data)
        current_encoding = detected.get('encoding', 'latin-1')
        
        # Lê com encoding detectado
        with open(input_path, 'r', encoding=current_encoding, errors='ignore') as f:
            content = f.read()
        
        # Escreve em UTF-8
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"✅ Arquivo convertido para UTF-8: {input_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao converter arquivo para UTF-8: {e}")
        return False

class UTF8Logger(logging.Handler):
    """Handler de logging que garante UTF-8"""
    
    def __init__(self, filename, encoding='utf-8', **kwargs):
        super().__init__()
        self.filename = filename
        self.encoding = encoding
        self.kwargs = kwargs
    
    def emit(self, record):
        try:
            msg = self.format(record)
            msg = ensure_utf8_string(msg)
            
            with open(self.filename, 'a', encoding=self.encoding, **self.kwargs) as f:
                f.write(msg + '\n')
                
        except Exception as e:
            self.handleError(record)

def setup_utf8_logging(log_file: str = 'logs/arqv30.log'):
    """Configura logging com UTF-8"""
    import os
    
    # Cria diretório se não existir
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configura handler UTF-8
    utf8_handler = UTF8Logger(log_file)
    utf8_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    # Adiciona ao logger root
    root_logger = logging.getLogger()
    root_logger.addHandler(utf8_handler)
    
    logger.info("✅ Logging UTF-8 configurado")

# Configuração automática ao importar
setup_utf8_environment()