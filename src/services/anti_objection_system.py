#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v2.0 - Anti-Objection System
Sistema de Engenharia Psicológica Anti-Objeção
"""

import logging
from typing import Dict, List, Any, Optional
from services.ai_manager import ai_manager

logger = logging.getLogger(__name__)

class AntiObjectionSystem:
    """Sistema de Engenharia Psicológica Anti-Objeção"""
    
    def __init__(self):
        """Inicializa o sistema anti-objeção"""
        self.universal_objections = self._load_universal_objections()
        self.hidden_objections = self._load_hidden_objections()
        self.neutralization_techniques = self._load_neutralization_techniques()
        
        logger.info("Anti-Objection System inicializado com arsenal completo")
    
    def _load_universal_objections(self) -> Dict[str, Dict[str, Any]]:
        """Carrega as 3 objeções universais"""
        return {
            'tempo': {
                'objecao': 'Não tenho tempo / Isso não é prioridade para mim',
                'raiz_emocional': 'Medo de mais uma responsabilidade / Falta de clareza sobre importância',
                'contra_ataque': 'Técnica do Cálculo da Sangria + Consequência Exponencial',
                'scripts': [
                    'Cada [período] que você adia resolver [problema], você está perdendo [quantia específica]',
                    'O problema não para de crescer enquanto você está ocupado com outras coisas',
                    'Esta oportunidade existe agora por [razão específica], depois pode não existir mais'
                ]
            },
            'dinheiro': {
                'objecao': 'Não tenho dinheiro / Minha vida não está tão ruim que precise investir',
                'raiz_emocional': 'Medo de perder dinheiro / Prioridades desalinhadas / Não vê valor',
                'contra_ataque': 'Comparação Cruel + ROI Absurdo + Custo de Oportunidade',
                'scripts': [
                    'Você gasta R$X em [coisa supérflua] mas hesita em investir [valor] em algo que muda sua vida',
                    'Se você conseguir apenas [resultado mínimo], já pagou o investimento [X] vezes',
                    'O que você vai perder NÃO fazendo isso é muito maior que o investimento'
                ]
            },
            'confianca': {
                'objecao': 'Me dê uma razão para acreditar (em você/produto/provas/mim mesmo)',
                'raiz_emocional': 'Histórico de fracassos / Medo de mais uma decepção / Baixa autoestima',
                'contra_ataque': 'Autoridade Técnica + Prova Social Qualificada + Garantia Agressiva',
                'scripts': [
                    'Eu já [credencial específica] e consegui [resultado específico] usando exatamente isso',
                    'Pessoas exatamente como você conseguiram [resultado] em [tempo] seguindo este método',
                    'Estou tão confiante que assumo todo o risco: [garantia específica]'
                ]
            }
        }
    
    def _load_hidden_objections(self) -> Dict[str, Dict[str, Any]]:
        """Carrega as 5 objeções ocultas críticas"""
        return {
            'autossuficiencia': {
                'objecao_oculta': 'Acho que consigo sozinho',
                'perfil_tipico': 'Pessoas com formação superior, experiência na área, ego profissional',
                'raiz_emocional': 'Orgulho / Medo de parecer incompetente',
                'sinais': ['Menções de "tentar sozinho"', 'Resistência a ajuda', 'Linguagem técnica excessiva'],
                'contra_ataque': 'O Expert que Precisou de Expert + Aceleração vs Tentativa',
                'scripts': [
                    'Mesmo sendo [autoridade], precisei de ajuda para [resultado específico]',
                    'A diferença entre tentar sozinho e ter orientação é [comparação temporal/financeira]'
                ]
            },
            'sinal_fraqueza': {
                'objecao_oculta': 'Aceitar ajuda é admitir fracasso',
                'perfil_tipico': 'Homens, líderes, pessoas com imagem a zelar',
                'raiz_emocional': 'Medo de julgamento / Perda de status / Humilhação',
                'sinais': ['Minimização de problemas', '"Está tudo bem"', 'Resistência a expor vulnerabilidade'],
                'contra_ataque': 'Reframe de Inteligência + Histórias de Heróis Vulneráveis',
                'scripts': [
                    'Pessoas inteligentes buscam atalhos. Pessoas burras insistem no caminho difícil',
                    'Os maiores CEOs do mundo têm coaches. Coincidência?'
                ]
            },
            'medo_novo': {
                'objecao_oculta': 'Não tenho pressa / Quando for a hora certa',
                'perfil_tipico': 'Pessoas estagnadas mas "confortáveis", medo do desconhecido',
                'raiz_emocional': 'Ansiedade sobre nova realidade / Zona de conforto',
                'sinais': ['"Quando for a hora certa"', 'Procrastinação disfarçada', 'Conformismo'],
                'contra_ataque': 'Dor da Estagnação + Janela Histórica',
                'scripts': [
                    'A única coisa pior que a dor da mudança é a dor do arrependimento',
                    'Esta oportunidade existe por [contexto específico]. Quem não aproveitar agora...'
                ]
            },
            'prioridades_desequilibradas': {
                'objecao_oculta': 'Não é dinheiro (mas gasta em outras coisas)',
                'perfil_tipico': 'Pessoas que gastam em lazer/consumo mas "não têm dinheiro" para evolução',
                'raiz_emocional': 'Não reconhece educação como prioridade / Vício em gratificação imediata',
                'sinais': ['Menções de gastos em outras áreas', 'Justificativas financeiras contraditórias'],
                'contra_ataque': 'Comparação Cruel + Cálculo de Oportunidade Perdida',
                'scripts': [
                    'R$200/mês em streaming vs R$2000 uma vez para nunca mais passar aperto',
                    'Você investe mais no seu carro que na sua mente'
                ]
            },
            'autoestima_destruida': {
                'objecao_oculta': 'Não confio em mim / Sou eu o problema',
                'perfil_tipico': 'Pessoas com múltiplas tentativas fracassadas, baixa confiança pessoal',
                'raiz_emocional': 'Histórico de fracassos / Medo de mais um fracasso',
                'sinais': ['"Já tentei antes"', 'Histórico de fracassos', 'Vitimização', 'Autodesqualificação'],
                'contra_ataque': 'Casos de Pessoas "Piores" + Diferencial do Método',
                'scripts': [
                    'Se [pessoa pior situação] conseguiu, você também consegue',
                    'O problema não era você, era a falta de método certo'
                ]
            }
        }
    
    def _load_neutralization_techniques(self) -> Dict[str, Dict[str, Any]]:
        """Carrega técnicas de neutralização"""
        return {
            'concordar_valorizar_apresentar': {
                'estrutura': 'Você tem razão... Por isso criei...',
                'quando_usar': 'Objeções lógicas válidas',
                'exemplo': 'Você tem razão em ser cauteloso com investimentos. Por isso criei uma garantia de 60 dias...'
            },
            'inversao_perspectiva': {
                'estrutura': 'Na verdade é o oposto do que você imagina...',
                'quando_usar': 'Crenças limitantes',
                'exemplo': 'Na verdade, pessoas que mais precisam de ajuda são as que mais resistem a ela...'
            },
            'memorias_reviravolta': {
                'estrutura': 'Lembre de quando você decidiu sem certeza...',
                'quando_usar': 'Medo de decisão',
                'exemplo': 'Lembre quando você decidiu [mudança importante] sem ter certeza absoluta...'
            },
            'confronto_controlado': {
                'estrutura': 'Quantas vezes você perdeu oportunidade por isso?',
                'quando_usar': 'Padrões autodestrutivos',
                'exemplo': 'Quantas vezes você já perdeu oportunidades por "pensar demais"?'
            },
            'nova_crenca': {
                'estrutura': 'Isso é uma crença limitante, vou te mostrar outro ângulo...',
                'quando_usar': 'Crenças arraigadas',
                'exemplo': 'Isso é uma crença limitante. Vou te mostrar como pessoas "sem tempo" criaram tempo...'
            }
        }
    
    def generate_complete_anti_objection_system(
        self, 
        objections_list: List[str], 
        avatar_data: Dict[str, Any], 
        context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Gera sistema completo anti-objeção"""
        
        try:
            logger.info(f"🛡️ Gerando sistema anti-objeção para {len(objections_list)} objeções")
            
            # Analisa objeções específicas do avatar
            analyzed_objections = self._analyze_specific_objections(objections_list, avatar_data)
            
            # Mapeia para objeções universais e ocultas
            mapped_objections = self._map_to_universal_objections(analyzed_objections)
            
            # Cria arsenal de contra-ataques
            counter_attacks = self._create_counter_attacks(mapped_objections, avatar_data, context_data)
            
            # Gera scripts personalizados
            personalized_scripts = self._generate_personalized_scripts(counter_attacks, avatar_data, context_data)
            
            # Cria arsenal de emergência
            emergency_arsenal = self._create_emergency_arsenal(avatar_data, context_data)
            
            return {
                'objecoes_universais': self._customize_universal_objections(avatar_data, context_data),
                'objecoes_ocultas': self._identify_hidden_objections(avatar_data),
                'contra_ataques_personalizados': counter_attacks,
                'scripts_personalizados': personalized_scripts,
                'arsenal_emergencia': emergency_arsenal,
                'sequencia_neutralizacao': self._create_neutralization_sequence(mapped_objections),
                'metricas_eficacia': self._create_effectiveness_metrics()
            }
            
        except Exception as e:
            logger.error(f"❌ Erro ao gerar sistema anti-objeção: {str(e)}")
            raise Exception(f"SISTEMA ANTI-OBJEÇÃO FALHOU: {str(e)}")
    
    def _analyze_specific_objections(
        self, 
        objections: List[str], 
        avatar_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Analisa objeções específicas do avatar"""
        
        analyzed = []
        
        for objection in objections:
            analysis = {
                'objecao_original': objection,
                'categoria': self._categorize_objection(objection),
                'intensidade': self._assess_objection_intensity(objection),
                'raiz_emocional': self._identify_emotional_root(objection),
                'frequencia_esperada': self._estimate_frequency(objection, avatar_data)
            }
            analyzed.append(analysis)
        
        return analyzed
    
    def _categorize_objection(self, objection: str) -> str:
        """Categoriza objeção"""
        
        objection_lower = objection.lower()
        
        if any(word in objection_lower for word in ['tempo', 'ocupado', 'prioridade']):
            return 'tempo'
        elif any(word in objection_lower for word in ['dinheiro', 'caro', 'investimento', 'preço']):
            return 'dinheiro'
        elif any(word in objection_lower for word in ['confiança', 'funciona', 'resultado', 'prova']):
            return 'confianca'
        elif any(word in objection_lower for word in ['sozinho', 'conseguir', 'tentar']):
            return 'autossuficiencia'
        elif any(word in objection_lower for word in ['ajuda', 'fraco', 'admitir']):
            return 'sinal_fraqueza'
        else:
            return 'geral'
    
    def _assess_objection_intensity(self, objection: str) -> str:
        """Avalia intensidade da objeção"""
        
        high_intensity_words = ['nunca', 'impossível', 'jamais', 'ódio', 'detesto']
        medium_intensity_words = ['difícil', 'complicado', 'problema', 'preocupação']
        
        objection_lower = objection.lower()
        
        if any(word in objection_lower for word in high_intensity_words):
            return 'Alta'
        elif any(word in objection_lower for word in medium_intensity_words):
            return 'Média'
        else:
            return 'Baixa'
    
    def _identify_emotional_root(self, objection: str) -> str:
        """Identifica raiz emocional da objeção"""
        
        objection_lower = objection.lower()
        
        if any(word in objection_lower for word in ['medo', 'receio', 'ansioso']):
            return 'Medo do desconhecido'
        elif any(word in objection_lower for word in ['fracasso', 'errado', 'tentei']):
            return 'Histórico de fracassos'
        elif any(word in objection_lower for word in ['orgulho', 'sozinho', 'independente']):
            return 'Orgulho ferido'
        elif any(word in objection_lower for word in ['confiança', 'dúvida', 'ceticismo']):
            return 'Desconfiança'
        else:
            return 'Resistência geral à mudança'
    
    def _estimate_frequency(self, objection: str, avatar_data: Dict[str, Any]) -> str:
        """Estima frequência da objeção"""
        
        # Baseado no perfil psicográfico
        personalidade = avatar_data.get('perfil_psicografico', {}).get('personalidade', '')
        
        if 'conservador' in personalidade.lower():
            return 'Alta'
        elif 'cauteloso' in personalidade.lower():
            return 'Média'
        else:
            return 'Baixa'
    
    def _map_to_universal_objections(self, analyzed_objections: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Mapeia objeções específicas para universais"""
        
        mapped = {
            'tempo': [],
            'dinheiro': [],
            'confianca': [],
            'ocultas': []
        }
        
        for objection in analyzed_objections:
            category = objection['categoria']
            if category in mapped:
                mapped[category].append(objection)
            else:
                mapped['ocultas'].append(objection)
        
        return mapped
    
    def _create_counter_attacks(
        self, 
        mapped_objections: Dict[str, List[Dict[str, Any]]], 
        avatar_data: Dict[str, Any], 
        context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Cria contra-ataques personalizados"""
        
        counter_attacks = {}
        
        for category, objections in mapped_objections.items():
            if not objections:
                continue
            
            if category in self.universal_objections:
                universal_data = self.universal_objections[category]
                counter_attacks[category] = self._customize_universal_counter_attack(
                    universal_data, objections, avatar_data, context_data
                )
            elif category == 'ocultas':
                counter_attacks['ocultas'] = self._create_hidden_counter_attacks(
                    objections, avatar_data, context_data
                )
        
        return counter_attacks
    
    def _customize_universal_counter_attack(
        self, 
        universal_data: Dict[str, Any], 
        specific_objections: List[Dict[str, Any]], 
        avatar_data: Dict[str, Any], 
        context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Customiza contra-ataque universal"""
        
        segmento = context_data.get('segmento', 'negócios')
        
        customized = universal_data.copy()
        
        # Customiza scripts para o segmento
        customized_scripts = []
        for script in universal_data['scripts']:
            if '[período]' in script:
                script = script.replace('[período]', 'mês')
            if '[problema]' in script:
                script = script.replace('[problema]', f'sua situação em {segmento}')
            if '[quantia específica]' in script:
                script = script.replace('[quantia específica]', 'R$ 5.000 em oportunidades')
            
            customized_scripts.append(script)
        
        customized['scripts_customizados'] = customized_scripts
        customized['objecoes_especificas'] = [obj['objecao_original'] for obj in specific_objections]
        
        return customized
    
    def _create_hidden_counter_attacks(
        self, 
        hidden_objections: List[Dict[str, Any]], 
        avatar_data: Dict[str, Any], 
        context_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Cria contra-ataques para objeções ocultas"""
        
        counter_attacks = []
        
        for objection in hidden_objections:
            # Identifica qual objeção oculta mais se aproxima
            best_match = self._find_best_hidden_match(objection)
            
            if best_match:
                counter_attack = self.hidden_objections[best_match].copy()
                counter_attack['objecao_especifica'] = objection['objecao_original']
                counter_attack['customizacao'] = self._customize_for_context(counter_attack, context_data)
                counter_attacks.append(counter_attack)
        
        return counter_attacks
    
    def _find_best_hidden_match(self, objection: Dict[str, Any]) -> Optional[str]:
        """Encontra melhor match para objeção oculta"""
        
        objection_text = objection['objecao_original'].lower()
        
        # Mapeia palavras-chave para objeções ocultas
        keyword_mapping = {
            'autossuficiencia': ['sozinho', 'conseguir', 'tentar', 'independente'],
            'sinal_fraqueza': ['ajuda', 'fraco', 'admitir', 'problema'],
            'medo_novo': ['hora certa', 'depois', 'futuro', 'quando'],
            'prioridades_desequilibradas': ['dinheiro', 'gasto', 'prioridade', 'investimento'],
            'autoestima_destruida': ['fracasso', 'tentei', 'não consegui', 'problema sou eu']
        }
        
        best_match = None
        max_matches = 0
        
        for hidden_type, keywords in keyword_mapping.items():
            matches = sum(1 for keyword in keywords if keyword in objection_text)
            if matches > max_matches:
                max_matches = matches
                best_match = hidden_type
        
        return best_match if max_matches > 0 else None
    
    def _customize_for_context(self, counter_attack: Dict[str, Any], context_data: Dict[str, Any]) -> str:
        """Customiza contra-ataque para contexto"""
        
        segmento = context_data.get('segmento', 'negócios')
        
        return f"Customizado para {segmento}: {counter_attack['contra_ataque']}"
    
    def _generate_personalized_scripts(
        self, 
        counter_attacks: Dict[str, Any], 
        avatar_data: Dict[str, Any], 
        context_data: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Gera scripts personalizados usando IA"""
        
        try:
            segmento = context_data.get('segmento', 'negócios')
            personalidade = avatar_data.get('perfil_psicografico', {}).get('personalidade', '')
            
            prompt = f"""
Crie scripts personalizados para neutralizar objeções no segmento {segmento}.

PERFIL DO AVATAR:
- Personalidade: {personalidade}
- Principais dores: {avatar_data.get('dores_viscerais', [])[:3]}
- Linguagem: {avatar_data.get('linguagem_interna', {})}

OBJEÇÕES IDENTIFICADAS:
{json.dumps(counter_attacks, indent=2, ensure_ascii=False)[:1000]}

RETORNE APENAS JSON VÁLIDO:

```json
{{
  "scripts_tempo": [
    "Script 1 personalizado para objeção de tempo",
    "Script 2 personalizado para objeção de tempo"
  ],
  "scripts_dinheiro": [
    "Script 1 personalizado para objeção de dinheiro",
    "Script 2 personalizado para objeção de dinheiro"
  ],
  "scripts_confianca": [
    "Script 1 personalizado para objeção de confiança",
    "Script 2 personalizado para objeção de confiança"
  ],
  "scripts_emergencia": [
    "Script de emergência 1",
    "Script de emergência 2"
  ]
}}
```
"""
            
            response = ai_manager.generate_analysis(prompt, max_tokens=1500)
            
            if response:
                clean_response = response.strip()
                if "```json" in clean_response:
                    start = clean_response.find("```json") + 7
                    end = clean_response.rfind("```")
                    clean_response = clean_response[start:end].strip()
                
                try:
                    scripts = json.loads(clean_response)
                    logger.info("✅ Scripts personalizados gerados com IA")
                    return scripts
                except json.JSONDecodeError:
                    logger.warning("⚠️ IA retornou JSON inválido para scripts")
            
            # Fallback para scripts básicos
            return self._create_basic_scripts(avatar_data, context_data)
            
        except Exception as e:
            logger.error(f"❌ Erro crítico ao gerar scripts personalizados: {str(e)}")
            raise RuntimeError(f"Falha crítica: não foi possível gerar scripts anti-objeção. Dados insuficientes para análise do segmento: {context_data.get('segmento', 'não especificado')}")
    
    def _validate_script_quality(self, scripts: Dict[str, List[str]], context_data: Dict[str, Any]) -> bool:
        """Valida qualidade dos scripts gerados"""
        segmento = context_data.get('segmento', '')
        
        if not scripts or len(scripts) < 3:
            logger.error("❌ Scripts insuficientes gerados")
            return False
        
        total_content = 0
        for category, script_list in scripts.items():
            if not script_list or len(script_list) < 2:
                logger.error(f"❌ Categoria {category} com scripts insuficientes")
                return False
            
            for script in script_list:
                if len(script) < 50:  # Scripts muito curtos
                    logger.error(f"❌ Script muito curto: {script[:30]}...")
                    return False
                total_content += len(script)
        
        if total_content < 1000:  # Mínimo de conteúdo total
            logger.error(f"❌ Scripts anti-objeção muito curtos: {total_content} caracteres. Mínimo: 1000")
            return False
        
        # Verifica se há menção ao segmento específico
        segment_mentioned = False
        for script_list in scripts.values():
            for script in script_list:
                if segmento.lower() in script.lower():
                    segment_mentioned = True
                    break
            if segment_mentioned:
                break
        
        if not segment_mentioned and segmento:
            logger.warning(f"⚠️ Scripts não mencionam segmento específico: {segmento}")
        
        return "A única diferença entre você e quem já conseguiu é a decisão de agir"
    
    def _customize_universal_objections(
        self, 
        avatar_data: Dict[str, Any], 
        context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Customiza objeções universais para o contexto"""
        
        customized = {}
        
        for category, objection_data in self.universal_objections.items():
            customized[category] = objection_data.copy()
            
            # Customiza para o segmento
            segmento = context_data.get('segmento', 'negócios')
            customized[category]['contexto_segmento'] = segmento
            
            # Adiciona exemplos específicos
            customized[category]['exemplos_especificos'] = self._create_specific_examples(
                category, avatar_data, context_data
            )
        
        return customized
    
    def _identify_hidden_objections(self, avatar_data: Dict[str, Any]) -> Dict[str, Any]:
        """Identifica objeções ocultas baseadas no avatar"""
        
        identified = {}
        
        # Analisa perfil para identificar objeções ocultas prováveis
        personalidade = avatar_data.get('perfil_psicografico', {}).get('personalidade', '').lower()
        valores = avatar_data.get('perfil_psicografico', {}).get('valores', '').lower()
        
        # Autossuficiência
        if any(trait in personalidade for trait in ['independente', 'autoconfiante', 'determinado']):
            identified['autossuficiencia'] = self.hidden_objections['autossuficiencia'].copy()
            identified['autossuficiencia']['probabilidade'] = 'Alta'
        
        # Sinal de fraqueza
        if any(trait in valores for trait in ['imagem', 'status', 'reconhecimento']):
            identified['sinal_fraqueza'] = self.hidden_objections['sinal_fraqueza'].copy()
            identified['sinal_fraqueza']['probabilidade'] = 'Média'
        
        # Medo do novo
        if any(trait in personalidade for trait in ['conservador', 'cauteloso', 'tradicional']):
            identified['medo_novo'] = self.hidden_objections['medo_novo'].copy()
            identified['medo_novo']['probabilidade'] = 'Alta'
        
        return identified
    
    def _create_specific_examples(
        self, 
        category: str, 
        avatar_data: Dict[str, Any], 
        context_data: Dict[str, Any]
    ) -> List[str]:
        """Cria exemplos específicos para cada categoria"""
        
        segmento = context_data.get('segmento', 'negócios')
        
        examples = {
            'tempo': [
                f"Cada mês sem otimizar {segmento} = R$ 10.000 em oportunidades perdidas",
                f"Profissionais de {segmento} que adiaram mudanças perderam 40% do market share"
            ],
            'dinheiro': [
                f"R$ 200/mês em ferramentas vs R$ 2.000 uma vez para dominar {segmento}",
                f"ROI médio em {segmento} com método correto: 500% em 12 meses"
            ],
            'confianca': [
                f"Mais de 500 profissionais de {segmento} já aplicaram com sucesso",
                f"Garantia específica para {segmento}: resultados em 60 dias ou dinheiro de volta"
            ]
        }
        
        return examples.get(category, [f"Exemplo específico para {category} em {segmento}"])
    
    def _create_emergency_arsenal(
        self, 
        avatar_data: Dict[str, Any], 
        context_data: Dict[str, Any]
    ) -> List[str]:
        """Cria arsenal de emergência para objeções de última hora"""
        
        return [
            "Vamos ser honestos: você vai continuar adiando até quando?",
            "A única diferença entre você e quem já conseguiu é a decisão de agir",
            "Quantas oportunidades você já perdeu por 'pensar demais'?",
            "O medo de errar está te impedindo de acertar",
            "Você prefere o arrependimento de ter tentado ou de não ter tentado?",
            "Cada 'não' que você diz para evolução é um 'sim' para estagnação",
            "O tempo que você está perdendo pensando, outros estão usando para agir",
            "Sua zona de conforto é uma prisão disfarçada de segurança"
        ]
    
    def _create_neutralization_sequence(self, mapped_objections: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Cria sequência de neutralização"""
        
        return [
            "1. IDENTIFICAR: Qual objeção está sendo verbalizada ou sinalizada",
            "2. CONCORDAR: Validar a preocupação como legítima",
            "3. VALORIZAR: Mostrar que pessoas inteligentes pensam assim",
            "4. APRESENTAR: Oferecer nova perspectiva ou solução",
            "5. CONFIRMAR: Verificar se a objeção foi neutralizada",
            "6. ANCORAR: Reforçar a nova crença instalada"
        ]
    
    def _create_effectiveness_metrics(self) -> Dict[str, Any]:
        """Cria métricas de eficácia do sistema"""
        
        return {
            'indicadores_neutralizacao': [
                'Mudança na linguagem corporal (abertura)',
                'Perguntas sobre próximos passos',
                'Redução de questionamentos',
                'Concordância verbal ou física'
            ],
            'sinais_resistencia_persistente': [
                'Repetição da mesma objeção',
                'Mudança de assunto',
                'Linguagem corporal fechada',
                'Questionamentos técnicos excessivos'
            ],
            'metricas_conversao': {
                'pre_neutralizacao': 'Taxa de conversão antes do sistema',
                'pos_neutralizacao': 'Taxa de conversão após aplicação',
                'tempo_medio_neutralizacao': 'Tempo médio para neutralizar objeção',
                'objecoes_mais_resistentes': 'Ranking das objeções mais difíceis'
            }
        }

# Instância global
anti_objection_system = AntiObjectionSystem()