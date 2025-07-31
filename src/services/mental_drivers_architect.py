#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v2.0 - Mental Drivers Architect
Arquiteto de Drivers Mentais - Sistema de Ancoragem Psicológica
"""

import logging
from typing import Dict, List, Any, Optional
from services.ai_manager import ai_manager

logger = logging.getLogger(__name__)

class MentalDriversArchitect:
    """Arquiteto de Drivers Mentais - Sistema de Ancoragem Psicológica"""
    
    def __init__(self):
        """Inicializa o arquiteto de drivers mentais"""
        self.universal_drivers = self._load_universal_drivers()
        logger.info(f"Mental Drivers Architect inicializado com {len(self.universal_drivers)} drivers universais")
    
    def _load_universal_drivers(self) -> List[Dict[str, Any]]:
        """Carrega os 19 drivers mentais universais"""
        return [
            {
                "id": 1,
                "nome": "Ferida Exposta",
                "categoria": "Emocional Primário",
                "gatilho": "Dor não resolvida",
                "mecanica": "Trazer à consciência o que foi reprimido",
                "template_ativacao": "Você ainda [comportamento doloroso] mesmo sabendo que [consequência]?"
            },
            {
                "id": 2,
                "nome": "Troféu Secreto", 
                "categoria": "Emocional Primário",
                "gatilho": "Desejo inconfessável",
                "mecanica": "Validar ambições proibidas",
                "template_ativacao": "Não é sobre dinheiro, é sobre [desejo real oculto]"
            },
            {
                "id": 3,
                "nome": "Inveja Produtiva",
                "categoria": "Emocional Primário", 
                "gatilho": "Comparação com pares",
                "mecanica": "Transformar inveja em combustível",
                "template_ativacao": "Enquanto você [situação atual], outros como você [resultado desejado]"
            },
            {
                "id": 4,
                "nome": "Relógio Psicológico",
                "categoria": "Emocional Primário",
                "gatilho": "Urgência existencial",
                "mecanica": "Tempo como recurso finito", 
                "template_ativacao": "Quantos [período] você ainda vai [desperdício]?"
            },
            {
                "id": 5,
                "nome": "Identidade Aprisionada",
                "categoria": "Emocional Primário",
                "gatilho": "Conflito entre quem é e quem poderia ser",
                "mecanica": "Expor a máscara social",
                "template_ativacao": "Você não é [rótulo limitante], você é [potencial real]"
            },
            {
                "id": 6,
                "nome": "Custo Invisível",
                "categoria": "Emocional Primário",
                "gatilho": "Perda não percebida",
                "mecanica": "Quantificar o preço da inação",
                "template_ativacao": "Cada dia sem [solução] custa [perda específica]"
            },
            {
                "id": 7,
                "nome": "Ambição Expandida",
                "categoria": "Emocional Primário",
                "gatilho": "Sonhos pequenos demais",
                "mecanica": "Elevar o teto mental de possibilidades",
                "template_ativacao": "Se o esforço é o mesmo, por que você está pedindo tão pouco?"
            },
            {
                "id": 8,
                "nome": "Diagnóstico Brutal",
                "categoria": "Emocional Primário",
                "gatilho": "Confronto com a realidade atual",
                "mecanica": "Criar indignação produtiva com status quo",
                "template_ativacao": "Olhe seus números/situação. Até quando você vai aceitar isso?"
            },
            {
                "id": 9,
                "nome": "Ambiente Vampiro",
                "categoria": "Emocional Primário",
                "gatilho": "Consciência do entorno tóxico",
                "mecanica": "Revelar como ambiente atual suga energia/potencial",
                "template_ativacao": "Seu ambiente te impulsiona ou te mantém pequeno?"
            },
            {
                "id": 10,
                "nome": "Mentor Salvador",
                "categoria": "Emocional Primário",
                "gatilho": "Necessidade de orientação externa",
                "mecanica": "Ativar desejo por figura de autoridade que acredita neles",
                "template_ativacao": "Você precisa de alguém que veja seu potencial quando você não consegue"
            },
            {
                "id": 11,
                "nome": "Coragem Necessária",
                "categoria": "Emocional Primário",
                "gatilho": "Medo paralisante disfarçado",
                "mecanica": "Transformar desculpas em decisões corajosas",
                "template_ativacao": "Não é sobre condições perfeitas, é sobre decidir apesar do medo"
            },
            {
                "id": 12,
                "nome": "Mecanismo Revelado",
                "categoria": "Racional Complementar",
                "gatilho": "Compreensão do como",
                "mecanica": "Desmistificar o complexo",
                "template_ativacao": "É simplesmente [analogia simples], não [complicação percebida]"
            },
            {
                "id": 13,
                "nome": "Prova Matemática",
                "categoria": "Racional Complementar",
                "gatilho": "Certeza numérica",
                "mecanica": "Equação irrefutável",
                "template_ativacao": "Se você fizer X por Y dias = Resultado Z garantido"
            },
            {
                "id": 14,
                "nome": "Padrão Oculto",
                "categoria": "Racional Complementar",
                "gatilho": "Insight revelador",
                "mecanica": "Mostrar o que sempre esteve lá",
                "template_ativacao": "Todos que conseguiram [resultado] fizeram [padrão específico]"
            },
            {
                "id": 15,
                "nome": "Exceção Possível",
                "categoria": "Racional Complementar",
                "gatilho": "Quebra de limitação",
                "mecanica": "Provar que regras podem ser quebradas",
                "template_ativacao": "Diziam que [limitação], mas [prova contrária]"
            },
            {
                "id": 16,
                "nome": "Atalho Ético",
                "categoria": "Racional Complementar",
                "gatilho": "Eficiência sem culpa",
                "mecanica": "Validar o caminho mais rápido",
                "template_ativacao": "Por que sofrer [tempo longo] se existe [atalho comprovado]?"
            },
            {
                "id": 17,
                "nome": "Decisão Binária",
                "categoria": "Racional Complementar",
                "gatilho": "Simplificação radical",
                "mecanica": "Eliminar zona cinzenta",
                "template_ativacao": "Ou você [ação desejada] ou aceita [consequência dolorosa]"
            },
            {
                "id": 18,
                "nome": "Oportunidade Oculta",
                "categoria": "Racional Complementar",
                "gatilho": "Vantagem não percebida",
                "mecanica": "Revelar demanda/chance óbvia mas ignorada",
                "template_ativacao": "O mercado está gritando por [solução] e ninguém está ouvindo"
            },
            {
                "id": 19,
                "nome": "Método vs Sorte",
                "categoria": "Racional Complementar",
                "gatilho": "Caos vs sistema",
                "mecanica": "Contrastar tentativa aleatória com caminho estruturado",
                "template_ativacao": "Sem método você está cortando mata com foice. Com método, está na autoestrada"
            }
        ]
    
    def generate_complete_drivers_system(
        self, 
        avatar_data: Dict[str, Any], 
        context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Gera sistema completo de drivers mentais customizados"""
        
        try:
            # Seleciona os drivers mais relevantes para o contexto
            selected_drivers = self._select_optimal_drivers(avatar_data, context_data)
            
            # Customiza cada driver selecionado
            customized_drivers = []
            for driver in selected_drivers:
                customized = self._customize_driver(driver, avatar_data, context_data)
                customized_drivers.append(customized)
            
            # Cria sequenciamento estratégico
            sequencing = self._create_strategic_sequencing(customized_drivers)
            
            return {
                "drivers_customizados": customized_drivers,
                "sequenciamento_estrategico": sequencing,
                "fases_implementacao": self._create_implementation_phases(customized_drivers),
                "scripts_ativacao": self._create_activation_scripts(customized_drivers),
                "metricas_eficacia": self._create_effectiveness_metrics(customized_drivers)
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar sistema de drivers: {str(e)}")
            return self._generate_fallback_drivers_system(context_data)
    
    def _select_optimal_drivers(self, avatar_data: Dict[str, Any], context_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Seleciona os drivers mais relevantes para o contexto"""
        
        # Drivers essenciais baseados no avatar
        essential_drivers = [
            "Diagnóstico Brutal",  # Sempre crítico para despertar
            "Ambição Expandida",   # Amplifica desejos
            "Relógio Psicológico", # Cria urgência
            "Método vs Sorte",     # Oferece caminho
            "Decisão Binária",     # Força escolha
            "Custo Invisível",     # Quantifica perdas
            "Coragem Necessária"   # Remove última barreira
        ]
        
        selected = []
        for driver_name in essential_drivers:
            driver = next((d for d in self.universal_drivers if d["nome"] == driver_name), None)
            if driver:
                selected.append(driver)
        
        return selected
    
    def _customize_driver(self, driver: Dict[str, Any], avatar_data: Dict[str, Any], context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Customiza um driver específico para o contexto"""
        
        segmento = context_data.get('segmento', 'negócios')
        produto = context_data.get('produto', 'produto/serviço')
        
        # Customizações específicas por driver
        customizations = {
            "Diagnóstico Brutal": {
                "pergunta_abertura": f"Há quanto tempo você está travado no mesmo nível em {segmento}?",
                "historia_analogia": f"É como um profissional de {segmento} que trabalha 12 horas por dia mas ganha o mesmo há 3 anos. Todo esse esforço, toda essa dedicação, mas os resultados não acompanham.",
                "metafora_visual": f"Imagine um hamster numa roda dourada. Ele corre muito, se esforça muito, mas continua no mesmo lugar. Esse é o profissional de {segmento} sem sistema.",
                "comando_acao": "Pare de aceitar mediocridade disfarçada de esforço"
            },
            "Ambição Expandida": {
                "pergunta_abertura": f"Por que você está pedindo tão pouco do seu negócio em {segmento}?",
                "historia_analogia": f"É como ter um Ferrari e usar apenas a primeira marcha. Você tem todo o potencial em {segmento}, mas está limitando artificialmente seus resultados.",
                "metafora_visual": f"Visualize seu negócio em {segmento} operando em 100% da capacidade. Cada processo otimizado, cada oportunidade capturada.",
                "comando_acao": "Eleve suas expectativas ao nível do seu potencial real"
            },
            "Relógio Psicológico": {
                "pergunta_abertura": f"Quantos anos você ainda vai desperdiçar sem dominar {segmento}?",
                "historia_analogia": f"Cada mês que passa sem otimizar seu negócio em {segmento} é um mês que seus concorrentes estão ganhando vantagem. O tempo não para para você se organizar.",
                "metafora_visual": f"Imagine um cronômetro gigante sobre seu negócio em {segmento}. A cada segundo que passa, oportunidades escapam.",
                "comando_acao": "Aja agora ou aceite ficar para trás permanentemente"
            }
        }
        
        driver_name = driver["nome"]
        custom = customizations.get(driver_name, {})
        
        return {
            "nome": driver_name,
            "gatilho_central": driver["gatilho"],
            "definicao_visceral": driver["mecanica"],
            "momento_instalacao": self._determine_installation_moment(driver_name),
            "roteiro_ativacao": {
                "pergunta_abertura": custom.get("pergunta_abertura", driver["template_ativacao"]),
                "historia_analogia": custom.get("historia_analogia", f"História customizada para {segmento}"),
                "metafora_visual": custom.get("metafora_visual", f"Metáfora visual para {segmento}"),
                "comando_acao": custom.get("comando_acao", "Comando de ação específico")
            },
            "frases_ancoragem": self._generate_anchor_phrases(driver_name, segmento),
            "prova_logica": self._generate_logical_proof(driver_name, context_data),
            "loop_reforco": f"Toda vez que você pensar em {segmento}, lembre: {driver['template_ativacao']}",
            "categoria": driver["categoria"],
            "poder_impacto": self._calculate_impact_power(driver_name, avatar_data)
        }
    
    def _determine_installation_moment(self, driver_name: str) -> str:
        """Determina o momento ideal para instalar cada driver"""
        
        moments = {
            "Diagnóstico Brutal": "Abertura - Para quebrar padrão e despertar consciência",
            "Ambição Expandida": "Desenvolvimento - Após despertar, amplificar desejos",
            "Relógio Psicológico": "Meio - Para criar pressão temporal",
            "Método vs Sorte": "Pré-pitch - Para posicionar solução",
            "Decisão Binária": "Fechamento - Para forçar escolha",
            "Custo Invisível": "Desenvolvimento - Para quantificar perdas",
            "Coragem Necessária": "Fechamento - Para remover última barreira"
        }
        
        return moments.get(driver_name, "Desenvolvimento - Momento padrão")
    
    def _generate_anchor_phrases(self, driver_name: str, segmento: str) -> List[str]:
        """Gera frases de ancoragem específicas"""
        
        phrases = {
            "Diagnóstico Brutal": [
                f"Mediocridade em {segmento} não é destino, é escolha",
                f"Seus resultados em {segmento} são o espelho das suas decisões",
                f"Aceitar menos em {segmento} é roubar de si mesmo"
            ],
            "Ambição Expandida": [
                f"Seu potencial em {segmento} não tem teto, suas crenças sim",
                f"Pequenos sonhos em {segmento} geram pequenos resultados",
                f"Se vai sonhar com {segmento}, sonhe grande"
            ],
            "Relógio Psicológico": [
                f"Cada dia sem otimizar {segmento} é um dia perdido para sempre",
                f"O tempo não espera você estar pronto para {segmento}",
                f"Procrastinação em {segmento} é autossabotagem disfarçada"
            ]
        }
        
        return phrases.get(driver_name, [f"Frase de ancoragem para {driver_name} em {segmento}"])
    
    def _generate_logical_proof(self, driver_name: str, context_data: Dict[str, Any]) -> Dict[str, str]:
        """Gera prova lógica para cada driver"""
        
        segmento = context_data.get('segmento', 'negócios')
        
        proofs = {
            "Diagnóstico Brutal": {
                "estatistica": f"87% dos profissionais de {segmento} estão presos no operacional",
                "caso_exemplo": f"Empresário de {segmento} que trabalhava 80h/semana e faturava o mesmo há 3 anos",
                "demonstracao": "Análise dos seus números atuais vs potencial real"
            },
            "Ambição Expandida": {
                "estatistica": f"Apenas 3% dos profissionais de {segmento} exploram seu potencial máximo",
                "caso_exemplo": f"Empresa de {segmento} que triplicou faturamento mudando apenas mindset",
                "demonstracao": "Cálculo do seu potencial real vs resultados atuais"
            },
            "Relógio Psicológico": {
                "estatistica": f"Cada ano de atraso em {segmento} custa em média R$ 100.000 em oportunidades",
                "caso_exemplo": f"Profissional que perdeu liderança no mercado de {segmento} por procrastinar",
                "demonstracao": "Cálculo do custo de cada mês de inação"
            }
        }
        
        return proofs.get(driver_name, {
            "estatistica": f"Dados específicos sobre {driver_name} em {segmento}",
            "caso_exemplo": f"Caso real de {driver_name} aplicado em {segmento}",
            "demonstracao": f"Como provar {driver_name} na prática"
        })
    
    def _calculate_impact_power(self, driver_name: str, avatar_data: Dict[str, Any]) -> str:
        """Calcula poder de impacto do driver para este avatar"""
        
        # Drivers de alto impacto para perfis empresariais
        high_impact_drivers = [
            "Diagnóstico Brutal",
            "Ambição Expandida", 
            "Relógio Psicológico",
            "Decisão Binária"
        ]
        
        if driver_name in high_impact_drivers:
            return "Alto"
        else:
            return "Médio"
    
    def _create_strategic_sequencing(self, drivers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cria sequenciamento estratégico dos drivers"""
        
        return {
            "fase_1_despertar": {
                "objetivo": "Quebrar padrão e despertar consciência",
                "drivers": ["Diagnóstico Brutal", "Oportunidade Oculta"],
                "duracao": "5-7 minutos",
                "intensidade": "Alta"
            },
            "fase_2_desejo": {
                "objetivo": "Amplificar desejos e possibilidades",
                "drivers": ["Ambição Expandida", "Troféu Secreto"],
                "duracao": "8-10 minutos", 
                "intensidade": "Crescente"
            },
            "fase_3_pressao": {
                "objetivo": "Criar urgência e pressão",
                "drivers": ["Relógio Psicológico", "Custo Invisível"],
                "duracao": "5-7 minutos",
                "intensidade": "Máxima"
            },
            "fase_4_direcao": {
                "objetivo": "Oferecer caminho e solução",
                "drivers": ["Método vs Sorte", "Mentor Salvador"],
                "duracao": "6-8 minutos",
                "intensidade": "Esperançosa"
            },
            "fase_5_decisao": {
                "objetivo": "Forçar decisão e ação",
                "drivers": ["Decisão Binária", "Coragem Necessária"],
                "duracao": "3-5 minutos",
                "intensidade": "Definitiva"
            }
        }
    
    def _create_implementation_phases(self, drivers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cria fases de implementação dos drivers"""
        
        return {
            "pre_lancamento": {
                "semana_1": "Instalar Diagnóstico Brutal em conteúdos",
                "semana_2": "Ativar Ambição Expandida em stories",
                "semana_3": "Começar Relógio Psicológico sutilmente",
                "semana_4": "Intensificar todos os drivers"
            },
            "durante_evento": {
                "abertura": "Diagnóstico Brutal + Oportunidade Oculta",
                "desenvolvimento": "Ambição Expandida + Custo Invisível",
                "pre_pitch": "Método vs Sorte + Mentor Salvador",
                "fechamento": "Decisão Binária + Coragem Necessária"
            },
            "pos_evento": {
                "follow_up_1": "Reforçar Relógio Psicológico",
                "follow_up_2": "Ativar Custo Invisível",
                "follow_up_3": "Decisão Binária final"
            }
        }
    
    def _create_activation_scripts(self, drivers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cria scripts de ativação prontos"""
        
        scripts = {}
        
        for driver in drivers:
            driver_name = driver["nome"]
            scripts[driver_name] = {
                "abertura": f"Deixa eu te fazer uma pergunta sobre {driver_name.lower()}...",
                "desenvolvimento": driver["roteiro_ativacao"]["historia_analogia"],
                "fechamento": driver["roteiro_ativacao"]["comando_acao"],
                "reativacao": f"Lembra do que falamos sobre {driver_name.lower()}?"
            }
        
        return scripts
    
    def _create_effectiveness_metrics(self, drivers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cria métricas de eficácia dos drivers"""
        
        return {
            "indicadores_sucesso": [
                "Silêncio absoluto durante ativação",
                "Comentários emocionais no chat",
                "Perguntas sobre quando abre inscrições",
                "Concordância física (acenar cabeça)"
            ],
            "sinais_resistencia": [
                "Questionamentos técnicos excessivos",
                "Mudança de assunto",
                "Objeções imediatas",
                "Linguagem corporal fechada"
            ],
            "metricas_conversao": {
                "engajamento": "Tempo de atenção por driver",
                "emocional": "Reações emocionais geradas",
                "comportamental": "Ações tomadas após ativação",
                "conversao": "Taxa de conversão pós-drivers"
            },
            "otimizacao": {
                "teste_ab": "Testar diferentes versões dos drivers",
                "personalizacao": "Adaptar por perfil de audiência",
                "timing": "Otimizar momentos de ativação",
                "intensidade": "Ajustar força dos drivers"
            }
        }
    
    def _generate_fallback_drivers_system(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Gera sistema de drivers de fallback"""
        
        segmento = context_data.get('segmento', 'negócios')
        
        return {
            "drivers_customizados": [
                {
                    "nome": "Diagnóstico Brutal",
                    "gatilho_central": "Confronto com realidade",
                    "roteiro_ativacao": {
                        "pergunta_abertura": f"Há quanto tempo você está estagnado em {segmento}?",
                        "comando_acao": "Pare de aceitar mediocridade"
                    }
                }
            ],
            "sequenciamento_estrategico": {
                "fase_1_despertar": {
                    "drivers": ["Diagnóstico Brutal"],
                    "objetivo": "Despertar consciência"
                }
            }
        }

# Instância global
mental_drivers_architect = MentalDriversArchitect()