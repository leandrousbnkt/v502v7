#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v2.0 - Visual Proofs Generator
Sistema Completo de Provas Visuais Instantâneas
"""

import logging
from typing import Dict, List, Any, Optional
from services.ai_manager import ai_manager

logger = logging.getLogger(__name__)

class VisualProofsGenerator:
    """Gerador de Provas Visuais Instantâneas - Sistema Completo"""
    
    def __init__(self):
        """Inicializa o gerador de provas visuais"""
        self.proof_categories = {
            'urgencia': 'Provas que criam urgência temporal',
            'crenca': 'Provas que instalam novas crenças',
            'objecao': 'Provas que destroem objeções',
            'transformacao': 'Provas que mostram transformação',
            'metodo': 'Provas que validam o método',
            'autoridade': 'Provas que estabelecem autoridade',
            'social': 'Provas sociais visuais'
        }
        
        self.experiment_library = self._load_experiment_library()
        logger.info("Visual Proofs Generator inicializado com biblioteca de experimentos")
    
    def _load_experiment_library(self) -> Dict[str, List[Dict[str, Any]]]:
        """Carrega biblioteca de experimentos visuais"""
        return {
            'urgencia': [
                {
                    'nome': 'Ampulheta do Dinheiro',
                    'conceito': 'Tempo = Dinheiro perdido',
                    'experimento': 'Ampulheta com moedas caindo representando oportunidades perdidas',
                    'materiais': ['Ampulheta transparente', 'Moedas', 'Calculadora'],
                    'impacto': 'Alto'
                },
                {
                    'nome': 'Vela da Oportunidade',
                    'conceito': 'Janela de oportunidade se fechando',
                    'experimento': 'Vela acesa que vai se apagando durante apresentação',
                    'materiais': ['Vela', 'Fósforos', 'Cronômetro'],
                    'impacto': 'Médio'
                }
            ],
            'crenca': [
                {
                    'nome': 'Metamorfose da Lagarta',
                    'conceito': 'Transformação é possível',
                    'experimento': 'Mostrar processo de metamorfose visual',
                    'materiais': ['Imagens sequenciais', 'Casulo real', 'Borboleta'],
                    'impacto': 'Alto'
                },
                {
                    'nome': 'Semente Gigante',
                    'conceito': 'Potencial oculto',
                    'experimento': 'Semente pequena vs árvore gigante que ela pode gerar',
                    'materiais': ['Semente', 'Foto de árvore gigante', 'Régua'],
                    'impacto': 'Médio'
                }
            ],
            'objecao': [
                {
                    'nome': 'Cofrinho Furado',
                    'conceito': 'Economizar sem sistema é inútil',
                    'experimento': 'Cofrinho com furos vs cofre lacrado',
                    'materiais': ['Cofrinho com furos', 'Cofre', 'Água colorida'],
                    'impacto': 'Alto'
                },
                {
                    'nome': 'GPS vs Mapa Rasgado',
                    'conceito': 'Método vs tentativa',
                    'experimento': 'Comparar navegação com GPS vs mapa danificado',
                    'materiais': ['GPS/celular', 'Mapa rasgado', 'Cronômetro'],
                    'impacto': 'Alto'
                }
            ],
            'transformacao': [
                {
                    'nome': 'Antes e Depois Extremo',
                    'conceito': 'Transformação radical possível',
                    'experimento': 'Comparação visual dramática de transformação',
                    'materiais': ['Fotos antes/depois', 'Espelho', 'Timeline visual'],
                    'impacto': 'Alto'
                },
                {
                    'nome': 'Carvão para Diamante',
                    'conceito': 'Pressão certa gera valor',
                    'experimento': 'Mostrar carvão e diamante - mesma origem, pressão diferente',
                    'materiais': ['Carvão', 'Diamante/cristal', 'Lupa'],
                    'impacto': 'Médio'
                }
            ],
            'metodo': [
                {
                    'nome': 'Receita vs Ingredientes',
                    'conceito': 'Sistema vs componentes soltos',
                    'experimento': 'Ingredientes separados vs prato pronto',
                    'materiais': ['Ingredientes diversos', 'Prato finalizado', 'Receita'],
                    'impacto': 'Alto'
                },
                {
                    'nome': 'Orquestra vs Ruído',
                    'conceito': 'Coordenação vs caos',
                    'experimento': 'Áudio de orquestra vs instrumentos desorganizados',
                    'materiais': ['Alto-falante', 'Áudios contrastantes', 'Partitura'],
                    'impacto': 'Médio'
                }
            ],
            'autoridade': [
                {
                    'nome': 'Biblioteca de Resultados',
                    'conceito': 'Experiência acumulada',
                    'experimento': 'Pilha de cases/certificados vs página em branco',
                    'materiais': ['Documentos reais', 'Pasta vazia', 'Balança'],
                    'impacto': 'Alto'
                }
            ],
            'social': [
                {
                    'nome': 'Multidão Seguindo',
                    'conceito': 'Prova social visual',
                    'experimento': 'Demonstração de como multidão segue direção',
                    'materiais': ['Fotos de multidão', 'Setas direcionais', 'Estatísticas'],
                    'impacto': 'Médio'
                }
            ]
        }
    
    def generate_complete_proofs_system(
        self, 
        concepts_to_prove: List[str], 
        avatar_data: Dict[str, Any], 
        context_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Gera sistema completo de provas visuais"""
        
        try:
            logger.info(f"🎭 Gerando provas visuais para {len(concepts_to_prove)} conceitos")
            
            # Analisa conceitos e categoriza
            categorized_concepts = self._categorize_concepts(concepts_to_prove)
            
            # Seleciona experimentos apropriados
            selected_experiments = self._select_optimal_experiments(categorized_concepts, avatar_data)
            
            # Customiza experimentos para o contexto
            customized_proofs = []
            for experiment in selected_experiments:
                customized_proof = self._customize_experiment(experiment, avatar_data, context_data)
                customized_proofs.append(customized_proof)
            
            # Adiciona experimentos únicos gerados por IA
            ai_generated_proofs = self._generate_ai_custom_proofs(concepts_to_prove, avatar_data, context_data)
            customized_proofs.extend(ai_generated_proofs)
            
            # Ordena por impacto e relevância
            final_proofs = self._optimize_proof_sequence(customized_proofs)
            
            logger.info(f"✅ {len(final_proofs)} provas visuais geradas")
            return final_proofs
            
        except Exception as e:
            logger.error(f"❌ Erro ao gerar provas visuais: {str(e)}")
            raise Exception(f"PROVAS VISUAIS FALHARAM: {str(e)}")
    
    def _categorize_concepts(self, concepts: List[str]) -> Dict[str, List[str]]:
        """Categoriza conceitos por tipo de prova necessária"""
        
        categorized = {category: [] for category in self.proof_categories.keys()}
        
        for concept in concepts:
            concept_lower = concept.lower()
            
            # Categorização baseada em palavras-chave
            if any(word in concept_lower for word in ['urgente', 'rápido', 'tempo', 'agora']):
                categorized['urgencia'].append(concept)
            elif any(word in concept_lower for word in ['acreditar', 'possível', 'conseguir']):
                categorized['crenca'].append(concept)
            elif any(word in concept_lower for word in ['objeção', 'dúvida', 'medo', 'resistência']):
                categorized['objecao'].append(concept)
            elif any(word in concept_lower for word in ['transformar', 'mudar', 'evoluir']):
                categorized['transformacao'].append(concept)
            elif any(word in concept_lower for word in ['método', 'sistema', 'processo']):
                categorized['metodo'].append(concept)
            elif any(word in concept_lower for word in ['autoridade', 'especialista', 'expert']):
                categorized['autoridade'].append(concept)
            else:
                categorized['social'].append(concept)
        
        return categorized
    
    def _select_optimal_experiments(
        self, 
        categorized_concepts: Dict[str, List[str]], 
        avatar_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Seleciona experimentos ótimos para os conceitos"""
        
        selected = []
        
        for category, concepts in categorized_concepts.items():
            if not concepts:
                continue
            
            # Seleciona experimentos da biblioteca
            available_experiments = self.experiment_library.get(category, [])
            
            # Prioriza experimentos de alto impacto
            high_impact = [exp for exp in available_experiments if exp.get('impacto') == 'Alto']
            medium_impact = [exp for exp in available_experiments if exp.get('impacto') == 'Médio']
            
            # Seleciona até 2 experimentos por categoria
            experiments_to_use = (high_impact + medium_impact)[:2]
            
            for experiment in experiments_to_use:
                experiment['concepts_addressed'] = concepts[:3]  # Máximo 3 conceitos por experimento
                selected.append(experiment)
        
        return selected[:10]  # Máximo 10 experimentos
    
    def _customize_experiment(
        self, 
        experiment: Dict[str, Any], 
        avatar_data: Dict[str, Any], 
        context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Customiza experimento para o contexto específico"""
        
        segmento = context_data.get('segmento', 'negócios')
        
        customized = experiment.copy()
        
        # Customiza baseado no segmento
        if 'medicina' in segmento.lower():
            customized = self._customize_for_medicine(customized)
        elif 'digital' in segmento.lower():
            customized = self._customize_for_digital(customized)
        elif 'consultoria' in segmento.lower():
            customized = self._customize_for_consulting(customized)
        
        # Adiciona roteiro completo
        customized['roteiro_completo'] = self._create_complete_script(customized, avatar_data, context_data)
        
        # Adiciona variações
        customized['variacoes'] = {
            'online': self._create_online_variation(customized),
            'presencial': self._create_presential_variation(customized),
            'grande_publico': self._create_large_audience_variation(customized)
        }
        
        return customized
    
    def _customize_for_medicine(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Customiza experimento para área médica"""
        
        if experiment['nome'] == 'GPS vs Mapa Rasgado':
            experiment['analogia'] = "É como fazer cirurgia com protocolo vs improvisar - vidas dependem da precisão"
            experiment['materiais'].append('Estetoscópio (prop)')
        
        return experiment
    
    def _customize_for_digital(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Customiza experimento para área digital"""
        
        if experiment['nome'] == 'Cofrinho Furado':
            experiment['analogia'] = "É como ter um site que converte 0,5% vs 15% - mesmo tráfego, resultados opostos"
            experiment['materiais'].append('Gráfico de conversão')
        
        return experiment
    
    def _customize_for_consulting(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Customiza experimento para consultoria"""
        
        if experiment['nome'] == 'Receita vs Ingredientes':
            experiment['analogia'] = "É como dar consultoria com metodologia vs dar conselhos soltos"
            experiment['materiais'].append('Metodologia impressa')
        
        return experiment
    
    def _create_complete_script(
        self, 
        experiment: Dict[str, Any], 
        avatar_data: Dict[str, Any], 
        context_data: Dict[str, Any]
    ) -> str:
        """Cria roteiro completo para o experimento"""
        
        return f"""
ROTEIRO COMPLETO - {experiment['nome']}

SETUP (30 segundos):
"Deixa eu te mostrar algo que vai mudar como você vê {experiment['conceito']}..."
[Preparar materiais visivelmente]

EXECUÇÃO (60-90 segundos):
1. [Demonstrar situação atual problemática]
2. [Criar tensão/expectativa]
3. [Revelar solução/transformação]

CLÍMAX (15 segundos):
"Viu a diferença? É EXATAMENTE isso que acontece quando..."

BRIDGE (30 segundos):
"Agora me diz: você quer continuar sendo o [situação problemática] ou quer ser o [situação ideal]?"

ANCORAGEM:
"Toda vez que você pensar em {experiment['conceito']}, lembre desta demonstração."
"""
    
    def _create_online_variation(self, experiment: Dict[str, Any]) -> str:
        """Cria variação para formato online"""
        return f"VERSÃO ONLINE: Usar close-up da câmera, materiais maiores, narração mais detalhada"
    
    def _create_presential_variation(self, experiment: Dict[str, Any]) -> str:
        """Cria variação para formato presencial"""
        return f"VERSÃO PRESENCIAL: Envolver audiência, usar voluntários, amplificar gestos"
    
    def _create_large_audience_variation(self, experiment: Dict[str, Any]) -> str:
        """Cria variação para grande público"""
        return f"VERSÃO GRANDE PÚBLICO: Usar projeção, materiais gigantes, microfone sem fio"
    
    def _generate_ai_custom_proofs(
        self, 
        concepts: List[str], 
        avatar_data: Dict[str, Any], 
        context_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Gera provas visuais customizadas usando IA"""
        
        try:
            segmento = context_data.get('segmento', 'negócios')
            
            prompt = f"""
Crie 3 experimentos visuais únicos e impactantes para o segmento {segmento}.

CONCEITOS A PROVAR:
{chr(10).join(concepts[:5])}

AVATAR:
- Perfil: {avatar_data.get('perfil_demografico', {})}
- Dores: {avatar_data.get('dores_viscerais', [])[:3]}
- Desejos: {avatar_data.get('desejos_secretos', [])[:3]}

RETORNE APENAS JSON VÁLIDO:

```json
[
  {{
    "nome": "Nome impactante do experimento",
    "conceito_alvo": "Conceito específico que prova",
    "experimento": "Descrição detalhada da demonstração física",
    "analogia": "Como conecta com a vida do avatar",
    "materiais": ["Lista de materiais necessários"],
    "roteiro_completo": "Roteiro passo a passo detalhado",
    "impacto_esperado": "Alto/Médio/Baixo",
    "momento_ideal": "Quando usar na apresentação"
  }}
]
```
"""
            
            response = ai_manager.generate_analysis(prompt, max_tokens=2000)
            
            if response:
                # Extrai JSON da resposta
                clean_response = response.strip()
                if "```json" in clean_response:
                    start = clean_response.find("```json") + 7
                    end = clean_response.rfind("```")
                    clean_response = clean_response[start:end].strip()
                
                try:
                    ai_proofs = json.loads(clean_response)
                    if isinstance(ai_proofs, list):
                        logger.info(f"✅ IA gerou {len(ai_proofs)} provas visuais customizadas")
                        return ai_proofs
                except json.JSONDecodeError:
                    logger.warning("⚠️ IA retornou JSON inválido para provas visuais")
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Erro ao gerar provas com IA: {str(e)}")
            return []
    
    def _optimize_proof_sequence(self, proofs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Otimiza sequência de provas por impacto"""
        
        # Ordena por impacto (Alto > Médio > Baixo)
        impact_order = {'Alto': 3, 'Médio': 2, 'Baixo': 1}
        
        proofs.sort(key=lambda x: impact_order.get(x.get('impacto', 'Baixo'), 1), reverse=True)
        
        # Adiciona sequenciamento estratégico
        for i, proof in enumerate(proofs):
            proof['sequencia'] = i + 1
            proof['momento_sugerido'] = self._suggest_timing(proof, i, len(proofs))
        
        return proofs[:12]  # Máximo 12 provas
    
    def _suggest_timing(self, proof: Dict[str, Any], index: int, total: int) -> str:
        """Sugere momento ideal para cada prova"""
        
        if index < total * 0.3:
            return "Abertura - Quebra de padrão"
        elif index < total * 0.6:
            return "Desenvolvimento - Construção de crença"
        elif index < total * 0.8:
            return "Pré-pitch - Preparação para oferta"
        else:
            return "Fechamento - Urgência final"

# Instância global
visual_proofs_generator = VisualProofsGenerator()