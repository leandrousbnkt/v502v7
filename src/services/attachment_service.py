#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v2.0 - Serviço de Processamento de Anexos
Análise inteligente de documentos e arquivos
"""

import os
import logging
import mimetypes
import re
from typing import Dict, List, Optional, Any, Tuple
from werkzeug.datastructures import FileStorage
import PyPDF2
import pandas as pd
from docx import Document
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class AttachmentService:
    """Serviço para processamento inteligente de anexos"""

    def __init__(self):
        """Inicializa serviço de anexos"""
        self.upload_folder = os.path.join(os.path.dirname(__file__), '..', 'uploads')
        os.makedirs(self.upload_folder, exist_ok=True)

        # Tipos de arquivo suportados
        self.supported_types = {
            'application/pdf': 'pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
            'application/msword': 'doc',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
            'application/vnd.ms-excel': 'xls',
            'text/csv': 'csv',
            'text/plain': 'txt',
            'application/json': 'json'
        }

        # Classificadores de conteúdo
        self.content_classifiers = {
            'drivers_mentais': [
                'urgência', 'escassez', 'autoridade', 'prova social', 'reciprocidade',
                'compromisso', 'aversão à perda', 'ancoragem', 'gatilho', 'persuasão'
            ],
            'provas_visuais': [
                'depoimento', 'testemunho', 'case', 'resultado', 'antes e depois',
                'screenshot', 'gráfico', 'estatística', 'número', 'percentual'
            ],
            'perfis_psicologicos': [
                'persona', 'perfil', 'comportamento', 'personalidade', 'psicológico',
                'motivação', 'desejo', 'dor', 'necessidade', 'aspiração'
            ],
            'dados_pesquisa': [
                'pesquisa', 'survey', 'questionário', 'dados', 'estatística',
                'amostra', 'respondente', 'análise', 'insight', 'tendência'
            ]
        }

    def process_attachment(
        self, 
        file: FileStorage, 
        session_id: str
    ) -> Dict[str, Any]:
        """Processa anexo enviado pelo usuário"""

        try:
            logger.info(f"Processando anexo: {file.filename}")

            # Valida arquivo
            if not file or not file.filename:
                return {
                    'success': False,
                    'error': 'Arquivo inválido'
                }

            # Verifica tipo de arquivo
            mime_type = file.content_type or mimetypes.guess_type(file.filename)[0]
            if mime_type not in self.supported_types:
                return {
                    'success': False,
                    'error': f'Tipo de arquivo não suportado: {mime_type}'
                }

            # Salva arquivo temporariamente
            file_path = self._save_temp_file(file, session_id)
            if not file_path:
                return {
                    'success': False,
                    'error': 'Erro ao salvar arquivo'
                }

            # Extrai conteúdo
            content = self._extract_content(file_path, mime_type)
            if not content:
                return {
                    'success': False,
                    'error': 'Erro ao extrair conteúdo'
                }

            # Classifica conteúdo
            content_type = self._classify_content(content)

            # Processa conteúdo específico
            processed_content = self._process_specific_content(content, content_type)

            # Remove arquivo temporário
            self._cleanup_temp_file(file_path)

            return {
                'success': True,
                'message': 'Anexo processado com sucesso',
                'session_id': session_id,
                'filename': file.filename,
                'content_type': content_type,
                'content_preview': processed_content[:500] + '...' if len(processed_content) > 500 else processed_content,
                'full_content': processed_content,
                'metadata': {
                    'file_size': len(content),
                    'mime_type': mime_type,
                    'processed_at': datetime.now().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"Erro ao processar anexo: {str(e)}")
            return {
                'success': False,
                'error': f'Erro interno: {str(e)}'
            }

    def _save_temp_file(self, file: FileStorage, session_id: str) -> Optional[str]:
        """Salva arquivo temporariamente"""
        try:
            # Gera nome único
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{session_id}_{timestamp}_{file.filename}"
            file_path = os.path.join(self.upload_folder, filename)

            # Salva arquivo
            file.save(file_path)

            return file_path

        except Exception as e:
            logger.error(f"Erro ao salvar arquivo: {str(e)}")
            return None

    def _extract_content(self, file_path: str, mime_type: str) -> Optional[str]:
        """Extrai conteúdo do arquivo baseado no tipo"""
        try:
            file_type = self.supported_types.get(mime_type)

            if file_type == 'pdf':
                return self._extract_pdf_content(file_path)
            elif file_type in ['docx', 'doc']:
                return self._extract_docx_content(file_path)
            elif file_type in ['xlsx', 'xls']:
                return self._extract_excel_content(file_path)
            elif file_type == 'csv':
                return self._extract_csv_content(file_path)
            elif file_type == 'txt':
                return self._extract_text_content(file_path)
            elif file_type == 'json':
                return self._extract_json_content(file_path)
            else:
                return None

        except Exception as e:
            logger.error(f"Erro ao extrair conteúdo: {str(e)}")
            return None

    def _extract_pdf_content(self, file_path: str) -> Optional[str]:
        """Extrai texto de arquivo PDF"""
        try:
            content = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    content += page.extract_text() + "\n"

            return content.strip()

        except Exception as e:
            logger.error(f"Erro ao extrair PDF: {str(e)}")
            return None

    def _extract_docx_content(self, file_path: str) -> Optional[str]:
        """Extrai texto de arquivo DOCX"""
        try:
            doc = Document(file_path)
            content = ""

            for paragraph in doc.paragraphs:
                content += paragraph.text + "\n"

            return content.strip()

        except Exception as e:
            logger.error(f"Erro ao extrair DOCX: {str(e)}")
            return None

    def _extract_excel_content(self, file_path: str) -> Optional[str]:
        """Extrai dados de arquivo Excel"""
        try:
            import pandas as pd

            # Lê todas as planilhas
            excel_file = pd.ExcelFile(file_path)
            content = ""

            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                content += f"PLANILHA: {sheet_name}\n"
                content += df.to_string(index=False) + "\n\n"

            # Valida qualidade do conteúdo extraído
            if len(content.strip()) < 100:
                logger.warning(f"⚠️ Conteúdo Excel muito curto: {len(content)} caracteres")
                return None

            return content.strip()

        except Exception as e:
            logger.error(f"Erro ao extrair Excel: {str(e)}")
            return None

    def _validate_content_quality(self, content: str, filename: str) -> bool:
        """Valida qualidade do conteúdo extraído"""
        if not content or len(content.strip()) < 50:
            logger.error(f"❌ Conteúdo muito curto para {filename}: {len(content) if content else 0} caracteres")
            return False

        # Verifica se não é apenas espaços ou caracteres especiais
        clean_content = re.sub(r'[\s\n\r\t]+', ' ', content.strip())
        if len(clean_content) < 30:
            logger.error(f"❌ Conteúdo inválido para {filename}")
            return False

        # Verifica se há palavras reais (não apenas números/símbolos)
        word_count = len(re.findall(r'\b[a-zA-ZÀ-ÿ]{3,}\b', content))
        if word_count < 5:
            logger.error(f"❌ Muito poucas palavras válidas em {filename}: {word_count}")
            return False

        logger.info(f"✅ Conteúdo validado para {filename}: {len(content)} caracteres, {word_count} palavras")
        return True

    def _extract_csv_content(self, file_path: str) -> Optional[str]:
        """Extrai dados de arquivo CSV"""
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            return df.to_string(index=False)

        except UnicodeDecodeError:
            # Tenta com encoding latin-1
            try:
                df = pd.read_csv(file_path, encoding='latin-1')
                return df.to_string(index=False)
            except Exception as e:
                logger.error(f"Erro ao extrair CSV: {str(e)}")
                return None
        except Exception as e:
            logger.error(f"Erro ao extrair CSV: {str(e)}")
            return None

    def _extract_text_content(self, file_path: str) -> Optional[str]:
        """Extrai conteúdo de arquivo texto"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()

        except UnicodeDecodeError:
            # Tenta com encoding latin-1
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                logger.error(f"Erro ao extrair texto: {str(e)}")
                return None
        except Exception as e:
            logger.error(f"Erro ao extrair texto: {str(e)}")
            return None

    def _extract_json_content(self, file_path: str) -> Optional[str]:
        """Extrai conteúdo de arquivo JSON"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                return json.dumps(data, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Erro ao extrair JSON: {str(e)}")
            return None

    def _classify_content(self, content: str) -> str:
        """Classifica o tipo de conteúdo baseado em palavras-chave"""
        content_lower = content.lower()
        scores = {}

        # Calcula score para cada categoria
        for category, keywords in self.content_classifiers.items():
            score = 0
            for keyword in keywords:
                score += content_lower.count(keyword.lower())
            scores[category] = score

        # Retorna categoria com maior score
        if scores:
            best_category = max(scores, key=scores.get)
            if scores[best_category] > 0:
                return best_category

        return 'geral'

    def _process_specific_content(self, content: str, content_type: str) -> str:
        """Processa conteúdo específico baseado no tipo"""

        if content_type == 'drivers_mentais':
            return self._process_mental_drivers(content)
        elif content_type == 'provas_visuais':
            return self._process_visual_proofs(content)
        elif content_type == 'perfis_psicologicos':
            return self._process_psychological_profiles(content)
        elif content_type == 'dados_pesquisa':
            return self._process_research_data(content)
        else:
            return self._process_general_content(content)

    def _process_mental_drivers(self, content: str) -> str:
        """Processa conteúdo relacionado a gatilhos mentais"""
        processed = "DRIVERS MENTAIS IDENTIFICADOS:\n\n"

        drivers_found = []
        for driver in self.content_classifiers['drivers_mentais']:
            if driver.lower() in content.lower():
                drivers_found.append(driver)

        if drivers_found:
            processed += f"Gatilhos encontrados: {', '.join(drivers_found)}\n\n"

        processed += "CONTEÚDO ORIGINAL:\n"
        processed += content

        return processed

    def _process_visual_proofs(self, content: str) -> str:
        """Processa provas visuais e depoimentos"""
        processed = "PROVAS VISUAIS E DEPOIMENTOS:\n\n"

        # Identifica números e percentuais
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?%?', content)
        if numbers:
            processed += f"Números identificados: {', '.join(numbers[:10])}\n\n"

        processed += "CONTEÚDO ORIGINAL:\n"
        processed += content

        return processed

    def _process_psychological_profiles(self, content: str) -> str:
        """Processa perfis psicológicos e personas"""
        processed = "PERFIS PSICOLÓGICOS IDENTIFICADOS:\n\n"

        # Busca por características de persona
        characteristics = []
        persona_keywords = ['idade', 'gênero', 'renda', 'comportamento', 'interesse']

        for keyword in persona_keywords:
            if keyword in content.lower():
                characteristics.append(keyword)

        if characteristics:
            processed += f"Características encontradas: {', '.join(characteristics)}\n\n"

        processed += "CONTEÚDO ORIGINAL:\n"
        processed += content

        return processed

    def _process_research_data(self, content: str) -> str:
        """Processa dados de pesquisa e estatísticas"""
        processed = "DADOS DE PESQUISA ANALISADOS:\n\n"

        # Identifica dados estatísticos
        import re
        stats = re.findall(r'\d+(?:\.\d+)?%', content)
        if stats:
            processed += f"Estatísticas encontradas: {', '.join(stats[:10])}\n\n"

        processed += "CONTEÚDO ORIGINAL:\n"
        processed += content

        return processed

    def _process_general_content(self, content: str) -> str:
        """Processamento geral de conteúdo"""
        processed = "CONTEÚDO GERAL PROCESSADO:\n\n"

        # Estatísticas básicas
        word_count = len(content.split())
        char_count = len(content)

        processed += f"Estatísticas: {word_count} palavras, {char_count} caracteres\n\n"
        processed += "CONTEÚDO ORIGINAL:\n"
        processed += content

        return processed

    def _cleanup_temp_file(self, file_path: str) -> None:
        """Remove arquivo temporário"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Erro ao remover arquivo temporário: {str(e)}")

    def get_session_attachments(self, session_id: str) -> List[Dict[str, Any]]:
        """Retorna anexos de uma sessão específica"""
        # Esta função seria implementada com um sistema de cache/banco
        # Por enquanto retorna lista vazia
        return []

    def process_text_file(self, file_path: str) -> Optional[str]:
        """Processa arquivo de texto simples"""
        return self._extract_text_content(file_path)

    def clear_session_attachments(self, session_id: str) -> bool:
        """Remove anexos de uma sessão"""
        try:
            # Remove arquivos temporários da sessão
            for filename in os.listdir(self.upload_folder):
                if filename.startswith(session_id):
                    file_path = os.path.join(self.upload_folder, filename)
                    os.remove(file_path)

            return True

        except Exception as e:
            logger.error(f"Erro ao limpar anexos da sessão: {str(e)}")
            return False

# Instância global do serviço
attachment_service = AttachmentService()