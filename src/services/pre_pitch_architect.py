#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v2.0 - Pre-Pitch Architect
Arquiteto do Pré-Pitch Invisível - Orquestração Psicológica
"""

import logging
from typing import Dict, List, Any, Optional
from services.ai_manager import ai_manager

logger = logging.getLogger(__name__)

class PrePitchArchitect:
    """Arquiteto do Pré-Pitch Invisível - Orquestração Psicológica"""
    
    def __init__(self):
        """Inicializa o arquiteto de pré-pitch"""
        self.psychological_phases = self._load_psychological_phases()
        self.transition_templates = self._load_transition_templates()
        
        logger.info("Pre-Pitch Architect inicializado")
    
    def _load_psychological_phases(self) -> Dict[str, Dict[str, Any]]:
        """Carrega fases psicológicas da orquestração"""
        return {
            'quebra': {
                'objetivo': 'Destruir a ilusão confortável',
                'duracao': '3-5 minutos',
                'intensidade': 'Alta',
                'drivers_ideais': ['Diagnóstico Brutal', 'Ferida Exposta'],
                'resultado_esperado': 'Desconforto produtivo'
            },
            'exposicao': {
                'objetivo': 'Revelar a ferida real',
                'duracao': '4-6 minutos',
                'intensidade': 'Crescente',
                'drivers_ideais': ['Custo Invisível', 'Ambiente Vampiro'],
                'resultado_esperado': 'Consciência da dor'
            },
            'indignacao': {
                'objetivo': 'Criar revolta produtiva',
                'duracao': '3-4 minutos',
                'intensidade': 'Máxima',
                'drivers_ideais': ['Relógio Psicológico', 'Inveja Produtiva'],
                'resultado_esperado': 'Urgência de mudança'
            },
            'vislumbre': {
                'objetivo': 'Mostrar o possível',
                'duracao': '5-7 minutos',
                'intensidade': 'Esperançosa',
                'drivers_ideais': ['Ambição Expandida', 'Troféu Secreto'],
                'resultado_esperado': 'Desejo amplificado'
            },
            'tensao': {
                'objetivo': 'Amplificar o gap',
                'duracao': '2-3 minutos',
                'intensidade': 'Crescente',
                'drivers_ideais': ['Identidade Aprisionada', 'Oportunidade Oculta'],
                'resultado_esperado': 'Tensão máxima'
            },
            'necessidade': {
                'objetivo': 'Tornar a mudança inevitável',
                'duracao': '3-4 minutos',
                'intensidade': 'Definitiva',
                'drivers_ideais': ['Método vs Sorte', 'Mentor Salvador'],
                'resultado_esperado': 'Necessidade de solução'
            }
        }
    
    def _load_transition_templates(self) -> Dict[str, str]:
        """Carrega templates de transição"""
        return {
            'quebra_para_exposicao': "Eu sei que isso dói ouvir... Mas sabe o que dói mais?",
            'exposicao_para_indignacao': "E o pior de tudo é que isso não precisa ser assim...",
            'indignacao_para_vislumbre': "Mas calma, não vim aqui só para abrir feridas...",
            'vislumbre_para_tensao': "Agora você vê a diferença entre onde está e onde poderia estar...",
            'tensao_para_necessidade': "A pergunta não é SE você vai mudar, é COMO...",
            'necessidade_para_logica': "Eu sei que você está sentindo isso agora... Mas seu cérebro racional está gritando: 'Será que funciona mesmo?' Então deixa eu te mostrar os números..."
        }
    
    def generate_complete_pre_pitch_system(
        self, 
        drivers_list: List[Dict[str, Any]], 
        avatar_analysis: Dict[str, Any], 
        context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Gera sistema completo de pré-pitch invisível"""
        
        try:
            logger.info(f"🎯 Gerando pré-pitch invisível com {len(drivers_list)} drivers")
            
            # Seleciona drivers ótimos para pré-pitch
            selected_drivers = self._select_optimal_drivers(drivers_list)
            
            # Cria orquestração emocional
            emotional_orchestration = self._create_emotional_orchestration(selected_drivers, avatar_analysis)
            
            # Gera roteiro completo
            complete_script = self._generate_complete_script(emotional_orchestration, context_data)
            
            # Cria variações por formato
            format_variations = self._create_format_variations(complete_script, context_data)
            
            # Gera métricas de sucesso
            success_metrics = self._create_success_metrics()
            
            return {
                'orquestracao_emocional': emotional_orchestration,
                'roteiro_completo': complete_script,
                'variacoes_formato': format_variations,
                'metricas_sucesso': success_metrics,
                'drivers_utilizados': [driver['nome'] for driver in selected_drivers],
                'duracao_total': self._calculate_total_duration(emotional_orchestration),
                'intensidade_maxima': self._calculate_max_intensity(emotional_orchestration)
            }
            
        except Exception as e:
            logger.error(f"❌ Erro ao gerar pré-pitch: {str(e)}")
            raise Exception(f"PRÉ-PITCH FALHOU: {str(e)}")
    
    def _select_optimal_drivers(self, drivers_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Seleciona drivers ótimos para pré-pitch"""
        
        # Drivers essenciais para pré-pitch
        essential_drivers = [
            'Diagnóstico Brutal', 'Ambição Expandida', 'Relógio Psicológico',
            'Método vs Sorte', 'Decisão Binária', 'Custo Invisível'
        ]
        
        selected = []
        
        # Seleciona drivers essenciais disponíveis
        for driver in drivers_list:
            driver_name = driver.get('nome', '')
            if any(essential in driver_name for essential in essential_drivers):
                selected.append(driver)
        
        # Se não tem drivers suficientes, pega os primeiros disponíveis
        if len(selected) < 4:
            selected.extend(drivers_list[:6])
        
        # Remove duplicatas
        seen_names = set()
        unique_selected = []
        for driver in selected:
            name = driver.get('nome', '')
            if name not in seen_names:
                seen_names.add(name)
                unique_selected.append(driver)
        
        return unique_selected[:7]  # Máximo 7 drivers
    
    def _create_emotional_orchestration(
        self, 
        selected_drivers: List[Dict[str, Any]], 
        avatar_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Cria orquestração emocional"""
        
        # Mapeia drivers para fases psicológicas
        phase_mapping = self._map_drivers_to_phases(selected_drivers)
        
        # Cria sequência psicológica
        psychological_sequence = []
        
        for phase_name, phase_data in self.psychological_phases.items():
            if phase_name in phase_mapping:
                phase_drivers = phase_mapping[phase_name]
                
                psychological_sequence.append({
                    'fase': phase_name,
                    'objetivo': phase_data['objetivo'],
                    'duracao': phase_data['duracao'],
                    'intensidade': phase_data['intensidade'],
                    'drivers_utilizados': [driver['nome'] for driver in phase_drivers],
                    'resultado_esperado': phase_data['resultado_esperado'],
                    'tecnicas': self._get_phase_techniques(phase_name, phase_drivers)
                })
        
        return {
            'sequencia_psicologica': psychological_sequence,
            'escalada_emocional': self._create_emotional_escalation(psychological_sequence),
            'pontos_criticos': self._identify_critical_points(psychological_sequence),
            'transicoes': self._create_phase_transitions(psychological_sequence)
        }
    
    def _map_drivers_to_phases(self, drivers: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Mapeia drivers para fases psicológicas"""
        
        mapping = {}
        
        for driver in drivers:
            driver_name = driver.get('nome', '')
            
            # Mapeia baseado no tipo de driver
            if any(word in driver_name.lower() for word in ['diagnóstico', 'brutal', 'ferida']):
                mapping.setdefault('quebra', []).append(driver)
            elif any(word in driver_name.lower() for word in ['custo', 'ambiente', 'vampiro']):
                mapping.setdefault('exposicao', []).append(driver)
            elif any(word in driver_name.lower() for word in ['relógio', 'urgência', 'inveja']):
                mapping.setdefault('indignacao', []).append(driver)
            elif any(word in driver_name.lower() for word in ['ambição', 'troféu', 'expandida']):
                mapping.setdefault('vislumbre', []).append(driver)
            elif any(word in driver_name.lower() for word in ['identidade', 'oportunidade']):
                mapping.setdefault('tensao', []).append(driver)
            elif any(word in driver_name.lower() for word in ['método', 'mentor', 'salvador']):
                mapping.setdefault('necessidade', []).append(driver)
        
        return mapping
    
    def _get_phase_techniques(self, phase_name: str, phase_drivers: List[Dict[str, Any]]) -> List[str]:
        """Obtém técnicas específicas para cada fase"""
        
        techniques = {
            'quebra': ['Confronto direto', 'Pergunta desconfortável', 'Estatística chocante'],
            'exposicao': ['Cálculo de perdas', 'Visualização da dor', 'Comparação cruel'],
            'indignacao': ['Urgência temporal', 'Comparação social', 'Consequências futuras'],
            'vislumbre': ['Visualização do sucesso', 'Casos de transformação', 'Possibilidades expandidas'],
            'tensao': ['Gap atual vs ideal', 'Identidade limitante', 'Oportunidade única'],
            'necessidade': ['Caminho claro', 'Mentor necessário', 'Método vs caos']
        }
        
        return techniques.get(phase_name, ['Técnica padrão'])
    
    def _generate_complete_script(
        self, 
        emotional_orchestration: Dict[str, Any], 
        context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Gera roteiro completo do pré-pitch"""
        
        try:
            segmento = context_data.get('segmento', 'negócios')
            
            prompt = f"""
Crie um roteiro completo de pré-pitch invisível para o segmento {segmento}.

ORQUESTRAÇÃO EMOCIONAL:
{json.dumps(emotional_orchestration, indent=2, ensure_ascii=False)[:2000]}

CONTEXTO:
- Segmento: {segmento}
- Produto: {context_data.get('produto', 'Não informado')}
- Público: {context_data.get('publico', 'Não informado')}

RETORNE APENAS JSON VÁLIDO:

```json
{{
  "abertura": {{
    "tempo": "3-5 minutos",
    "objetivo": "Quebrar padrão e despertar consciência",
    "script": "Roteiro detalhado da abertura",
    "frases_chave": ["Frase 1", "Frase 2"],
    "transicao": "Como conectar com próxima fase"
  }},
  "desenvolvimento": {{
    "tempo": "8-12 minutos",
    "objetivo": "Amplificar dor e desejo",
    "script": "Roteiro detalhado do desenvolvimento",
    "momentos_criticos": ["Momento 1", "Momento 2"],
    "escalada_emocional": "Como aumentar intensidade"
  }},
  "pre_climax": {{
    "tempo": "3-4 minutos",
    "objetivo": "Criar tensão máxima",
    "script": "Roteiro detalhado do pré-clímax",
    "ponto_virada": "Momento exato da virada",
    "preparacao_pitch": "Como preparar para oferta"
  }},
  "fechamento": {{
    "tempo": "2-3 minutos",
    "objetivo": "Transição perfeita para pitch",
    "script": "Roteiro detalhado do fechamento",
    "ponte_oferta": "Frase de transição para oferta",
    "estado_mental_ideal": "Como devem estar mentalmente"
  }}
}}
```
"""
            
            response = ai_manager.generate_analysis(prompt, max_tokens=2500)
            
            if response:
                clean_response = response.strip()
                if "```json" in clean_response:
                    start = clean_response.find("```json") + 7
                    end = clean_response.rfind("```")
                    clean_response = clean_response[start:end].strip()
                
                try:
                    script = json.loads(clean_response)
                    logger.info("✅ Roteiro completo gerado com IA")
                    return script
                except json.JSONDecodeError:
                    logger.warning("⚠️ IA retornou JSON inválido para roteiro")
            
            # Fallback para roteiro básico
            return self._create_basic_script(context_data)
            
        except Exception as e:
            logger.error(f"❌ Erro ao gerar roteiro: {str(e)}")
            return self._create_basic_script(context_data)
    
    def _create_basic_script(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Cria roteiro básico como fallback"""
        
        segmento = context_data.get('segmento', 'negócios')
        
        return {
            'abertura': {
                'tempo': '3-5 minutos',
                'objetivo': 'Quebrar padrão e despertar consciência',
                'script': f"Deixa eu te fazer uma pergunta sobre {segmento}... Há quanto tempo você está no mesmo nível?",
                'frases_chave': [
                    f"A verdade sobre {segmento} que ninguém te conta",
                    "Isso vai doer, mas precisa ser dito"
                ],
                'transicao': "E sabe por que isso acontece?"
            },
            'desenvolvimento': {
                'tempo': '8-12 minutos',
                'objetivo': 'Amplificar dor e desejo',
                'script': f"Cada dia que passa sem otimizar {segmento} é dinheiro saindo do seu bolso...",
                'momentos_criticos': [
                    "Cálculo da perda financeira",
                    "Comparação com concorrentes"
                ],
                'escalada_emocional': "Aumentar pressão gradualmente"
            },
            'pre_climax': {
                'tempo': '3-4 minutos',
                'objetivo': 'Criar tensão máxima',
                'script': f"Agora você tem duas escolhas em {segmento}...",
                'ponto_virada': "Momento da decisão binária",
                'preparacao_pitch': "Preparar para revelar solução"
            },
            'fechamento': {
                'tempo': '2-3 minutos',
                'objetivo': 'Transição perfeita para pitch',
                'script': "Eu vou te mostrar exatamente como sair dessa situação...",
                'ponte_oferta': "Mas antes, preciso saber se você está pronto...",
                'estado_mental_ideal': "Ansioso pela solução"
            }
        }
    
    def _create_format_variations(
        self, 
        complete_script: Dict[str, Any], 
        context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Cria variações por formato"""
        
        return {
            'webinar': {
                'duracao_total': '15-20 minutos',
                'adaptacoes': [
                    'Usar chat para engajamento',
                    'Pausas para perguntas retóricas',
                    'Slides de apoio visual'
                ],
                'timing': 'Últimos 20 minutos antes da oferta'
            },
            'evento_presencial': {
                'duracao_total': '25-35 minutos',
                'adaptacoes': [
                    'Interação direta com audiência',
                    'Movimentação no palco',
                    'Provas visuais físicas'
                ],
                'timing': 'Distribuído ao longo do evento'
            },
            'cpl_3_aulas': {
                'duracao_total': '10-15 minutos',
                'adaptacoes': [
                    'Construção gradual ao longo das aulas',
                    'Callbacks entre aulas',
                    'Intensificação na aula 3'
                ],
                'timing': 'Final da aula 3'
            },
            'lives_aquecimento': {
                'duracao_total': '5-8 minutos por live',
                'adaptacoes': [
                    'Sementes em cada live',
                    'Preparação subliminar',
                    'Crescimento de intensidade'
                ],
                'timing': 'Distribuído nas lives'
            }
        }
    
    def _create_emotional_escalation(self, sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cria escalada emocional"""
        
        return {
            'curva_intensidade': [
                {'fase': seq['fase'], 'intensidade': seq['intensidade']} 
                for seq in sequence
            ],
            'pontos_pico': [
                seq['fase'] for seq in sequence 
                if seq['intensidade'] in ['Máxima', 'Definitiva']
            ],
            'momentos_alivio': [
                seq['fase'] for seq in sequence 
                if seq['intensidade'] == 'Esperançosa'
            ]
        }
    
    def _identify_critical_points(self, sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identifica pontos críticos"""
        
        critical_points = []
        
        for seq in sequence:
            if seq['intensidade'] in ['Máxima', 'Definitiva']:
                critical_points.append({
                    'fase': seq['fase'],
                    'momento': f"Durante {seq['objetivo'].lower()}",
                    'risco': 'Perda de audiência se muito intenso',
                    'oportunidade': 'Máximo impacto emocional',
                    'gestao': 'Monitorar reações e ajustar intensidade'
                })
        
        return critical_points
    
    def _create_phase_transitions(self, sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cria transições entre fases"""
        
        transitions = []
        
        for i in range(len(sequence) - 1):
            current_phase = sequence[i]['fase']
            next_phase = sequence[i + 1]['fase']
            
            transition_key = f"{current_phase}_para_{next_phase}"
            transition_text = self.transition_templates.get(
                transition_key, 
                f"Transição de {current_phase} para {next_phase}"
            )
            
            transitions.append({
                'de': current_phase,
                'para': next_phase,
                'script': transition_text,
                'tempo': '15-30 segundos',
                'tecnica': 'Ponte emocional suave'
            })
        
        return transitions
    
    def _create_success_metrics(self) -> Dict[str, Any]:
        """Cria métricas de sucesso"""
        
        return {
            'indicadores_durante': [
                'Silêncio absoluto durante ativação',
                'Comentários emocionais no chat',
                'Perguntas sobre quando abre inscrições',
                'Concordância física (acenar cabeça)'
            ],
            'indicadores_apos': [
                'Ansiedade visível para a oferta',
                'Perguntas sobre preço/formato',
                'Comentários "já quero comprar"',
                'Objeções minimizadas'
            ],
            'sinais_resistencia': [
                'Questionamentos técnicos excessivos',
                'Mudança de assunto',
                'Objeções imediatas',
                'Linguagem corporal fechada'
            ],
            'metricas_conversao': {
                'engajamento': 'Tempo de atenção por fase',
                'emocional': 'Reações emocionais geradas',
                'comportamental': 'Ações tomadas após ativação',
                'conversao': 'Taxa de conversão pós-pré-pitch'
            }
        }
    
    def _calculate_total_duration(self, orchestration: Dict[str, Any]) -> str:
        """Calcula duração total"""
        
        sequence = orchestration.get('sequencia_psicologica', [])
        
        total_min = 0
        total_max = 0
        
        for phase in sequence:
            duration = phase.get('duracao', '3-4 minutos')
            
            # Extrai números da duração
            import re
            numbers = re.findall(r'\d+', duration)
            if len(numbers) >= 2:
                total_min += int(numbers[0])
                total_max += int(numbers[1])
            elif len(numbers) == 1:
                total_min += int(numbers[0])
                total_max += int(numbers[0])
        
        return f"{total_min}-{total_max} minutos"
    
    def _calculate_max_intensity(self, orchestration: Dict[str, Any]) -> str:
        """Calcula intensidade máxima"""
        
        sequence = orchestration.get('sequencia_psicologica', [])
        
        intensities = [phase.get('intensidade', 'Baixa') for phase in sequence]
        
        if 'Máxima' in intensities:
            return 'Máxima'
        elif 'Alta' in intensities:
            return 'Alta'
        elif 'Crescente' in intensities:
            return 'Crescente'
        else:
            return 'Média'

# Instância global
pre_pitch_architect = PrePitchArchitect()