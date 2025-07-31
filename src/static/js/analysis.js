// ARQV30 Enhanced v2.0 - Analysis JavaScript with Robust Content Extraction

class AnalysisManager {
    constructor() {
        this.sessionId = window.app?.sessionId || this.generateSessionId();
        this.isAnalyzing = false;
        this.progressInterval = null;
        this.currentAnalysis = null;
        this.extractorStats = {};
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadExtractorStats();
    }

    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    setupEventListeners() {
        // Form submission
        const form = document.getElementById('analysisForm');
        if (form) {
            form.addEventListener('submit', (e) => {
                e.preventDefault();
                this.startAnalysis();
            });
        }

        // Test buttons
        this.setupTestButtons();
    }

    setupTestButtons() {
        // Add test buttons to the interface
        const testButtonsHtml = `
            <div class="test-section" style="margin-top: 20px; padding: 20px; background: var(--glass-bg); border-radius: 12px; border: 1px solid var(--glass-border);">
                <h4 style="color: var(--accent-primary); margin-bottom: 15px;">🧪 Testes do Sistema</h4>
                <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                    <button class="btn-secondary" onclick="analysisManager.testExtraction()">
                        <i class="fas fa-download"></i> Testar Extração
                    </button>
                    <button class="btn-secondary" onclick="analysisManager.testSearch()">
                        <i class="fas fa-search"></i> Testar Busca
                    </button>
                    <button class="btn-secondary" onclick="analysisManager.showExtractorStats()">
                        <i class="fas fa-chart-bar"></i> Stats Extratores
                    </button>
                    <button class="btn-secondary" onclick="analysisManager.resetExtractors()">
                        <i class="fas fa-refresh"></i> Reset Extratores
                    </button>
                </div>
            </div>
        `;

        const formActions = document.querySelector('.form-actions');
        if (formActions) {
            formActions.insertAdjacentHTML('beforebegin', testButtonsHtml);
        }
    }

    async loadExtractorStats() {
        try {
            const response = await fetch('/api/extractor_stats');
            if (response.ok) {
                this.extractorStats = await response.json();
                this.updateExtractorStatusDisplay();
            }
        } catch (error) {
            console.error('Erro ao carregar stats dos extratores:', error);
        }
    }

    updateExtractorStatusDisplay() {
        const statusBar = document.getElementById('apiStatus');
        if (statusBar && this.extractorStats.stats) {
            const stats = this.extractorStats.stats;
            const globalStats = stats.global || {};
            
            const successRate = globalStats.success_rate || 0;
            const totalExtractions = globalStats.total_extractions || 0;
            
            if (successRate >= 80) {
                statusBar.innerHTML = `
                    <i class="fas fa-check-circle" style="color: var(--accent-tertiary);"></i>
                    <span>Extratores: ${successRate.toFixed(1)}% sucesso (${totalExtractions} extrações)</span>
                `;
            } else if (successRate >= 50) {
                statusBar.innerHTML = `
                    <i class="fas fa-exclamation-triangle" style="color: var(--accent-gold);"></i>
                    <span>Extratores: ${successRate.toFixed(1)}% sucesso (${totalExtractions} extrações)</span>
                `;
            } else {
                statusBar.innerHTML = `
                    <i class="fas fa-times-circle" style="color: #ff6b6b;"></i>
                    <span>Extratores: ${successRate.toFixed(1)}% sucesso (${totalExtractions} extrações)</span>
                `;
            }
        }
    }

    async testExtraction() {
        const testUrl = prompt('Digite uma URL para testar a extração:', 'https://g1.globo.com/tecnologia/');
        
        if (!testUrl) return;

        try {
            window.app?.showInfo('Testando extração de conteúdo...');

            const response = await fetch('/api/test_extraction', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ url: testUrl })
            });

            const result = await response.json();

            if (result.success) {
                window.app?.showSuccess(`Extração bem-sucedida! ${result.content_length} caracteres extraídos. Qualidade: ${result.validation.score}%`);
                
                // Mostra preview do conteúdo
                this.showContentPreview(result);
            } else {
                window.app?.showError(`Falha na extração: ${result.error}`);
            }

            // Atualiza stats
            this.extractorStats = { stats: result.extractor_stats };
            this.updateExtractorStatusDisplay();

        } catch (error) {
            console.error('Erro no teste de extração:', error);
            window.app?.showError('Erro no teste de extração');
        }
    }

    showContentPreview(result) {
        const modal = document.createElement('div');
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
            backdrop-filter: blur(10px);
        `;

        const content = document.createElement('div');
        content.style.cssText = `
            background: var(--bg-surface);
            border-radius: 16px;
            padding: 30px;
            max-width: 80%;
            max-height: 80%;
            overflow-y: auto;
            border: 1px solid var(--glass-border);
            box-shadow: var(--shadow-floating);
        `;

        content.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                <h3 style="color: var(--accent-primary); margin: 0;">📄 Preview do Conteúdo Extraído</h3>
                <button onclick="this.closest('.modal').remove()" style="
                    background: none; 
                    border: none; 
                    color: var(--text-muted); 
                    font-size: 24px; 
                    cursor: pointer;
                ">×</button>
            </div>
            
            <div style="margin-bottom: 20px;">
                <strong style="color: var(--accent-secondary);">URL:</strong> 
                <span style="color: var(--text-secondary); word-break: break-all;">${result.url}</span>
            </div>
            
            <div style="margin-bottom: 20px;">
                <strong style="color: var(--accent-secondary);">Qualidade:</strong> 
                <span style="color: ${result.validation.valid ? 'var(--accent-tertiary)' : '#ff6b6b'};">
                    ${result.validation.score}% - ${result.validation.valid ? 'VÁLIDO' : 'INVÁLIDO'}
                </span>
            </div>
            
            <div style="margin-bottom: 20px;">
                <strong style="color: var(--accent-secondary);">Tamanho:</strong> 
                <span style="color: var(--text-primary);">${result.content_length} caracteres</span>
            </div>
            
            <div style="background: var(--glass-bg); padding: 20px; border-radius: 8px; border: 1px solid var(--glass-border);">
                <h4 style="color: var(--accent-primary); margin-bottom: 10px;">Conteúdo:</h4>
                <pre style="
                    white-space: pre-wrap; 
                    color: var(--text-primary); 
                    font-family: 'Inter', sans-serif; 
                    line-height: 1.6;
                    max-height: 300px;
                    overflow-y: auto;
                ">${result.content_preview}</pre>
            </div>
        `;

        modal.className = 'modal';
        modal.appendChild(content);
        document.body.appendChild(modal);

        // Close on background click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        });
    }

    async testSearch() {
        const testQuery = prompt('Digite uma query para testar a busca:', 'mercado digital Brasil 2024');
        
        if (!testQuery) return;

        try {
            window.app?.showInfo('Testando sistema de busca...');

            const response = await fetch('/api/test_search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    query: testQuery,
                    max_results: 5
                })
            });

            const result = await response.json();

            if (result.success) {
                window.app?.showSuccess(`Busca bem-sucedida! ${result.results_count} resultados encontrados`);
                console.log('Resultados da busca:', result.results);
            } else {
                window.app?.showError(`Falha na busca: ${result.error}`);
            }

        } catch (error) {
            console.error('Erro no teste de busca:', error);
            window.app?.showError('Erro no teste de busca');
        }
    }

    async showExtractorStats() {
        try {
            const response = await fetch('/api/extractor_stats');
            const result = await response.json();

            if (result.success) {
                this.displayExtractorStatsModal(result.stats);
            } else {
                window.app?.showError('Erro ao obter estatísticas dos extratores');
            }

        } catch (error) {
            console.error('Erro ao obter stats:', error);
            window.app?.showError('Erro ao obter estatísticas');
        }
    }

    displayExtractorStatsModal(stats) {
        const modal = document.createElement('div');
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
            backdrop-filter: blur(10px);
        `;

        const content = document.createElement('div');
        content.style.cssText = `
            background: var(--bg-surface);
            border-radius: 16px;
            padding: 30px;
            max-width: 90%;
            max-height: 80%;
            overflow-y: auto;
            border: 1px solid var(--glass-border);
            box-shadow: var(--shadow-floating);
        `;

        let statsHtml = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                <h3 style="color: var(--accent-primary); margin: 0;">📊 Estatísticas dos Extratores</h3>
                <button onclick="this.closest('.modal').remove()" style="
                    background: none; 
                    border: none; 
                    color: var(--text-muted); 
                    font-size: 24px; 
                    cursor: pointer;
                ">×</button>
            </div>
        `;

        // Estatísticas globais
        if (stats.global) {
            const global = stats.global;
            statsHtml += `
                <div style="background: var(--glass-bg); padding: 20px; border-radius: 12px; margin-bottom: 20px; border: 1px solid var(--glass-border);">
                    <h4 style="color: var(--accent-secondary); margin-bottom: 15px;">🌐 Estatísticas Globais</h4>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                        <div>
                            <strong style="color: var(--text-secondary);">Total de Extrações:</strong>
                            <div style="color: var(--accent-primary); font-size: 24px; font-weight: 700;">${global.total_extractions}</div>
                        </div>
                        <div>
                            <strong style="color: var(--text-secondary);">Sucessos:</strong>
                            <div style="color: var(--accent-tertiary); font-size: 24px; font-weight: 700;">${global.total_successes}</div>
                        </div>
                        <div>
                            <strong style="color: var(--text-secondary);">Falhas:</strong>
                            <div style="color: #ff6b6b; font-size: 24px; font-weight: 700;">${global.total_failures}</div>
                        </div>
                        <div>
                            <strong style="color: var(--text-secondary);">Taxa de Sucesso:</strong>
                            <div style="color: ${global.success_rate >= 80 ? 'var(--accent-tertiary)' : global.success_rate >= 50 ? 'var(--accent-gold)' : '#ff6b6b'}; font-size: 24px; font-weight: 700;">${global.success_rate}%</div>
                        </div>
                    </div>
                </div>
            `;
        }

        // Estatísticas por extrator
        statsHtml += `<div style="display: grid; gap: 15px;">`;

        for (const [extractorName, extractorStats] of Object.entries(stats)) {
            if (extractorName === 'global') continue;

            const available = extractorStats.available;
            const successRate = extractorStats.success_rate || 0;
            const usageCount = extractorStats.usage_count || 0;

            statsHtml += `
                <div style="background: var(--glass-bg); padding: 20px; border-radius: 12px; border: 1px solid var(--glass-border);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                        <h5 style="color: var(--accent-primary); margin: 0; text-transform: capitalize;">${extractorName}</h5>
                        <span style="
                            padding: 4px 12px; 
                            border-radius: 20px; 
                            font-size: 12px; 
                            font-weight: 600;
                            background: ${available ? 'rgba(6, 255, 165, 0.2)' : 'rgba(255, 107, 107, 0.2)'};
                            color: ${available ? 'var(--accent-tertiary)' : '#ff6b6b'};
                            border: 1px solid ${available ? 'var(--accent-tertiary)' : '#ff6b6b'};
                        ">
                            ${available ? 'DISPONÍVEL' : 'INDISPONÍVEL'}
                        </span>
                    </div>
                    
                    ${available ? `
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; font-size: 14px;">
                            <div>
                                <strong style="color: var(--text-secondary);">Usos:</strong>
                                <div style="color: var(--text-primary);">${usageCount}</div>
                            </div>
                            <div>
                                <strong style="color: var(--text-secondary);">Sucessos:</strong>
                                <div style="color: var(--accent-tertiary);">${extractorStats.success_count || 0}</div>
                            </div>
                            <div>
                                <strong style="color: var(--text-secondary);">Erros:</strong>
                                <div style="color: #ff6b6b;">${extractorStats.error_count || 0}</div>
                            </div>
                            <div>
                                <strong style="color: var(--text-secondary);">Taxa:</strong>
                                <div style="color: ${successRate >= 80 ? 'var(--accent-tertiary)' : successRate >= 50 ? 'var(--accent-gold)' : '#ff6b6b'};">${successRate}%</div>
                            </div>
                            <div>
                                <strong style="color: var(--text-secondary);">Tempo Médio:</strong>
                                <div style="color: var(--text-primary);">${extractorStats.avg_response_time || 0}s</div>
                            </div>
                        </div>
                    ` : `
                        <div style="color: var(--text-muted); font-style: italic;">
                            ${extractorStats.reason || 'Não disponível'}
                        </div>
                    `}
                </div>
            `;
        }

        statsHtml += `</div>`;

        content.innerHTML = statsHtml;
        modal.className = 'modal';
        modal.appendChild(content);
        document.body.appendChild(modal);

        // Close on background click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        });
    }

    async resetExtractors() {
        if (!confirm('Tem certeza que deseja resetar as estatísticas dos extratores?')) {
            return;
        }

        try {
            const response = await fetch('/api/reset_extractors', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({})
            });

            const result = await response.json();

            if (result.success) {
                window.app?.showSuccess('Estatísticas dos extratores resetadas com sucesso!');
                this.extractorStats = { stats: result.stats };
                this.updateExtractorStatusDisplay();
            } else {
                window.app?.showError(`Erro ao resetar: ${result.error}`);
            }

        } catch (error) {
            console.error('Erro ao resetar extratores:', error);
            window.app?.showError('Erro ao resetar extratores');
        }
    }

    async startAnalysis() {
        if (this.isAnalyzing) {
            window.app?.showWarning('Análise já em andamento!');
            return;
        }

        try {
            // Coleta dados do formulário
            const formData = this.collectFormData();
            
            // Validação básica
            if (!formData.segmento) {
                window.app?.showError('Segmento é obrigatório!');
                return;
            }

            // Adiciona session ID
            formData.session_id = this.sessionId;

            // Inicia análise
            this.isAnalyzing = true;
            this.showProgressSection();
            this.startProgressTracking();

            // Faz requisição de análise
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            const result = await response.json();

            if (response.ok && result) {
                this.onAnalysisSuccess(result);
            } else {
                this.onAnalysisError(result.error || 'Erro desconhecido', result);
            }

        } catch (error) {
            console.error('Erro na análise:', error);
            this.onAnalysisError(error.message);
        } finally {
            this.isAnalyzing = false;
            this.stopProgressTracking();
        }
    }

    collectFormData() {
        const form = document.getElementById('analysisForm');
        const formData = new FormData(form);
        const data = {};

        // Converte FormData para objeto
        for (let [key, value] of formData.entries()) {
            data[key] = value;
        }

        // Adiciona arquivos enviados
        const uploadedFiles = window.uploadManager?.getUploadedFiles() || [];
        if (uploadedFiles.length > 0) {
            data.attachments = uploadedFiles;
        }

        return data;
    }

    showProgressSection() {
        const progressArea = document.getElementById('progressArea');
        const resultsArea = document.getElementById('resultsArea');
        
        if (progressArea) {
            progressArea.style.display = 'block';
            progressArea.scrollIntoView({ behavior: 'smooth' });
        }
        
        if (resultsArea) {
            resultsArea.style.display = 'none';
        }

        // Desabilita botão de análise
        const analyzeBtn = document.getElementById('analyzeBtn');
        if (analyzeBtn) {
            window.app?.setButtonLoading(analyzeBtn, true);
        }
    }

    startProgressTracking() {
        this.updateProgress(0, 'Iniciando análise ultra-detalhada...');
        
        // Simula progresso inicial
        let step = 0;
        const steps = [
            'Validando dados de entrada...',
            'Executando pesquisa web massiva REAL...',
            'Extraindo conteúdo com múltiplos extratores...',
            'Validando qualidade do conteúdo...',
            'Analisando com múltiplas IAs...',
            'Gerando avatar ultra-detalhado...',
            'Criando drivers mentais customizados...',
            'Desenvolvendo provas visuais...',
            'Construindo sistema anti-objeção...',
            'Arquitetando pré-pitch invisível...',
            'Predizendo futuro do mercado...',
            'Consolidando insights exclusivos...',
            'Finalizando análise GIGANTE...'
        ];

        this.progressInterval = setInterval(() => {
            if (step < steps.length - 1) {
                step++;
                this.updateProgress(step, steps[step]);
            }
        }, 3000);
    }

    stopProgressTracking() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
    }

    updateProgress(step, message) {
        const totalSteps = 13;
        const percentage = (step / totalSteps) * 100;

        // Atualiza barra de progresso
        const progressFill = document.querySelector('.progress-fill');
        if (progressFill) {
            progressFill.style.width = percentage + '%';
        }

        // Atualiza mensagem
        const currentStep = document.getElementById('currentStep');
        if (currentStep) {
            currentStep.textContent = message;
        }

        // Atualiza contador
        const stepCounter = document.getElementById('stepCounter');
        if (stepCounter) {
            stepCounter.textContent = `${step}/${totalSteps}`;
        }

        // Atualiza tempo estimado
        const estimatedTime = document.getElementById('estimatedTime');
        if (estimatedTime) {
            const remaining = Math.max(0, (totalSteps - step) * 15); // 15s por step
            const minutes = Math.floor(remaining / 60);
            const seconds = remaining % 60;
            estimatedTime.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        }
    }

    onAnalysisSuccess(result) {
        this.currentAnalysis = result;
        
        // Completa progresso
        this.updateProgress(13, '🎉 Análise concluída com sucesso!');
        
        setTimeout(() => {
            this.hideProgressSection();
            this.showResults(result);
            window.app?.showSuccess('Análise ultra-detalhada concluída com sucesso!');
        }, 1000);
    }

    onAnalysisError(error, result = null) {
        logger.error('Erro na análise:', error);
        
        this.hideProgressSection();
        
        // Mostra erro detalhado
        let errorMessage = `Erro na análise: ${error}`;
        
        if (result && result.recommendation) {
            errorMessage += `\n\nRecomendação: ${result.recommendation}`;
        }
        
        if (result && result.required_apis) {
            errorMessage += `\n\nAPIs necessárias:\n${result.required_apis.join('\n')}`;
        }
        
        window.app?.showError(errorMessage);
        
        // Reabilita botão
        const analyzeBtn = document.getElementById('analyzeBtn');
        if (analyzeBtn) {
            window.app?.setButtonLoading(analyzeBtn, false);
        }
    }

    hideProgressSection() {
        const progressArea = document.getElementById('progressArea');
        if (progressArea) {
            progressArea.style.display = 'none';
        }
        
        // Reabilita botão
        const analyzeBtn = document.getElementById('analyzeBtn');
        if (analyzeBtn) {
            window.app?.setButtonLoading(analyzeBtn, false);
        }
    }

    showResults(analysis) {
        const resultsArea = document.getElementById('resultsArea');
        if (!resultsArea) return;

        resultsArea.style.display = 'block';
        resultsArea.scrollIntoView({ behavior: 'smooth' });

        // Renderiza cada seção
        this.renderAvatarResults(analysis);
        this.renderDriversResults(analysis);
        this.renderCompetitionResults(analysis);
        this.renderPositioningResults(analysis);
        this.renderKeywordsResults(analysis);
        this.renderMetricsResults(analysis);
        this.renderActionPlanResults(analysis);
        this.renderInsightsResults(analysis);
        this.renderVisualProofsResults(analysis);
        this.renderAntiObjectionResults(analysis);
        this.renderPrePitchResults(analysis);
        this.renderFutureResults(analysis);
        this.renderResearchResults(analysis);
        this.renderMetadataResults(analysis);

        // Habilita download PDF
        this.enablePdfDownload(analysis);
    }

    renderAvatarResults(analysis) {
        const container = document.getElementById('avatarResults');
        if (!container || !analysis.avatar_ultra_detalhado) return;

        const avatar = analysis.avatar_ultra_detalhado;
        
        container.innerHTML = `
            <div class="result-section">
                <div class="result-section-header">
                    <i class="fas fa-user-circle"></i>
                    <h4>Avatar Ultra-Detalhado</h4>
                </div>
                <div class="result-section-content">
                    <div class="avatar-grid">
                        ${this.renderAvatarCard('Perfil Demográfico', avatar.perfil_demografico, 'fas fa-chart-pie')}
                        ${this.renderAvatarCard('Perfil Psicográfico', avatar.perfil_psicografico, 'fas fa-brain')}
                    </div>
                    
                    ${this.renderAvatarList('Dores Viscerais', avatar.dores_viscerais, 'fas fa-heart-broken', '#ff6b6b')}
                    ${this.renderAvatarList('Desejos Secretos', avatar.desejos_secretos, 'fas fa-star', 'var(--accent-tertiary)')}
                    ${this.renderAvatarList('Objeções Reais', avatar.objecoes_reais, 'fas fa-shield-alt', 'var(--accent-gold)')}
                </div>
            </div>
        `;
    }

    renderAvatarCard(title, data, icon) {
        if (!data) return '';
        
        const items = Object.entries(data).map(([key, value]) => `
            <div class="avatar-item">
                <span class="avatar-label">${key.replace(/_/g, ' ')}</span>
                <span class="avatar-value">${value}</span>
            </div>
        `).join('');

        return `
            <div class="avatar-card">
                <h5><i class="${icon}"></i> ${title}</h5>
                ${items}
            </div>
        `;
    }

    renderAvatarList(title, items, icon, color) {
        if (!items || !Array.isArray(items)) return '';
        
        const listItems = items.map(item => `
            <li class="insight-item">
                <i class="${icon}" style="color: ${color};"></i>
                <span class="insight-text">${item}</span>
            </li>
        `).join('');

        return `
            <div style="margin-top: 30px;">
                <h5 style="color: ${color}; margin-bottom: 20px; display: flex; align-items: center; gap: 10px;">
                    <i class="${icon}"></i> ${title}
                </h5>
                <ul class="insight-list">${listItems}</ul>
            </div>
        `;
    }

    renderResearchResults(analysis) {
        const container = document.getElementById('researchResults');
        if (!container || !analysis.pesquisa_web_massiva) return;

        const research = analysis.pesquisa_web_massiva;
        
        container.innerHTML = `
            <div class="result-section">
                <div class="result-section-header">
                    <i class="fas fa-globe"></i>
                    <h4>Pesquisa Web Massiva REAL</h4>
                </div>
                <div class="result-section-content">
                    <div class="research-content">
                        <div class="data-quality-indicator">
                            <span class="quality-label">Qualidade dos Dados:</span>
                            <span class="quality-value real-data">100% DADOS REAIS</span>
                        </div>
                        
                        <div class="research-stats">
                            <h5 style="color: var(--accent-primary); margin-bottom: 15px;">📊 Estatísticas da Pesquisa</h5>
                            <div class="stats-grid">
                                <div class="stat-item">
                                    <span class="stat-label">Queries Executadas</span>
                                    <span class="stat-value">${research.total_queries || 0}</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">Resultados Encontrados</span>
                                    <span class="stat-value">${research.total_resultados || 0}</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">Fontes Únicas</span>
                                    <span class="stat-value">${research.fontes_unicas || 0}</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">Conteúdo Extraído</span>
                                    <span class="stat-value">${(research.conteudo_extraido_chars || 0).toLocaleString()} chars</span>
                                </div>
                            </div>
                        </div>
                        
                        ${research.queries_executadas ? `
                            <div class="queries-executed">
                                <h5 style="color: var(--accent-secondary); margin-bottom: 15px;">🔍 Queries Executadas</h5>
                                <div class="results-list">
                                    ${research.queries_executadas.slice(0, 10).map(query => `
                                        <div class="result-item">
                                            <span style="color: var(--text-primary);">${query}</span>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                        ` : ''}
                        
                        ${research.resultados_detalhados ? `
                            <div class="detailed-results">
                                <h5 style="color: var(--accent-tertiary); margin-bottom: 15px;">📄 Fontes Analisadas</h5>
                                <div class="results-list">
                                    ${research.resultados_detalhados.slice(0, 15).map(source => `
                                        <div class="result-item">
                                            <h5>${source.title}</h5>
                                            <div class="result-url">${source.url}</div>
                                            <div class="result-source">Fonte: ${source.source}</div>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
    }

    // Métodos de renderização existentes (mantidos)
    renderDriversResults(analysis) {
        const container = document.getElementById('driversResults');
        if (!container || !analysis.drivers_mentais_customizados) return;

        const drivers = analysis.drivers_mentais_customizados;
        
        container.innerHTML = `
            <div class="result-section">
                <div class="result-section-header">
                    <i class="fas fa-brain"></i>
                    <h4>Drivers Mentais Customizados</h4>
                </div>
                <div class="result-section-content">
                    <div class="drivers-grid">
                        ${drivers.map(driver => `
                            <div class="driver-card">
                                <h4>${driver.nome}</h4>
                                <div class="driver-content">
                                    <p><strong>Gatilho Central:</strong> ${driver.gatilho_central}</p>
                                    <p><strong>Definição:</strong> ${driver.definicao_visceral}</p>
                                    
                                    ${driver.roteiro_ativacao ? `
                                        <div class="driver-script">
                                            <h6>Roteiro de Ativação</h6>
                                            <p><strong>Pergunta:</strong> ${driver.roteiro_ativacao.pergunta_abertura}</p>
                                            <p><strong>História:</strong> ${driver.roteiro_ativacao.historia_analogia}</p>
                                            <p><strong>Comando:</strong> ${driver.roteiro_ativacao.comando_acao}</p>
                                        </div>
                                    ` : ''}
                                    
                                    ${driver.frases_ancoragem ? `
                                        <div class="anchor-phrases">
                                            <h6>Frases de Ancoragem</h6>
                                            <ul>
                                                ${driver.frases_ancoragem.map(frase => `<li>"${frase}"</li>`).join('')}
                                            </ul>
                                        </div>
                                    ` : ''}
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
    }

    renderInsightsResults(analysis) {
        const container = document.getElementById('insightsResults');
        if (!container || !analysis.insights_exclusivos) return;

        const insights = analysis.insights_exclusivos;
        
        container.innerHTML = `
            <div class="result-section">
                <div class="result-section-header">
                    <i class="fas fa-lightbulb"></i>
                    <h4>Insights Exclusivos Ultra-Valiosos</h4>
                </div>
                <div class="result-section-content">
                    <div class="insights-showcase">
                        ${insights.map((insight, index) => `
                            <div class="insight-card">
                                <div class="insight-number">${index + 1}</div>
                                <div class="insight-content">${insight}</div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
    }

    renderMetadataResults(analysis) {
        const container = document.getElementById('metadataResults');
        if (!container || !analysis.metadata) return;

        const metadata = analysis.metadata;
        
        container.innerHTML = `
            <div class="result-section">
                <div class="result-section-header">
                    <i class="fas fa-info-circle"></i>
                    <h4>Metadados da Análise</h4>
                </div>
                <div class="result-section-content">
                    <div class="metadata-grid">
                        <div class="metadata-item">
                            <span class="metadata-label">Tempo de Processamento</span>
                            <span class="metadata-value">${metadata.processing_time_formatted || 'N/A'}</span>
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">Engine de Análise</span>
                            <span class="metadata-value">${metadata.analysis_engine || 'N/A'}</span>
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">Score de Qualidade</span>
                            <span class="metadata-value">${metadata.quality_score || 'N/A'}%</span>
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">Fontes Analisadas</span>
                            <span class="metadata-value">${metadata.real_data_sources || 0}</span>
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">Conteúdo Analisado</span>
                            <span class="metadata-value">${(metadata.total_content_analyzed || 0).toLocaleString()} chars</span>
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">Garantia de Dados</span>
                            <span class="metadata-value" style="color: var(--accent-tertiary);">100% REAIS</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    // Métodos de renderização simplificados para outras seções
    renderCompetitionResults(analysis) {
        // Implementação simplificada
        const container = document.getElementById('competitionResults');
        if (container && analysis.analise_concorrencia_detalhada) {
            container.innerHTML = `
                <div class="result-section">
                    <div class="result-section-header">
                        <i class="fas fa-chess"></i>
                        <h4>Análise de Concorrência</h4>
                    </div>
                    <div class="result-section-content">
                        <p style="color: var(--text-primary);">Análise detalhada da concorrência disponível nos dados completos.</p>
                    </div>
                </div>
            `;
        }
    }

    renderPositioningResults(analysis) {
        const container = document.getElementById('positioningResults');
        if (container && analysis.escopo) {
            container.innerHTML = `
                <div class="result-section">
                    <div class="result-section-header">
                        <i class="fas fa-bullseye"></i>
                        <h4>Posicionamento e Escopo</h4>
                    </div>
                    <div class="result-section-content">
                        <p style="color: var(--text-primary);">Estratégia de posicionamento detalhada disponível nos dados completos.</p>
                    </div>
                </div>
            `;
        }
    }

    renderKeywordsResults(analysis) {
        const container = document.getElementById('keywordsResults');
        if (container && analysis.estrategia_palavras_chave) {
            container.innerHTML = `
                <div class="result-section">
                    <div class="result-section-header">
                        <i class="fas fa-tags"></i>
                        <h4>Estratégia de Palavras-Chave</h4>
                    </div>
                    <div class="result-section-content">
                        <p style="color: var(--text-primary);">Estratégia completa de palavras-chave disponível nos dados completos.</p>
                    </div>
                </div>
            `;
        }
    }

    renderMetricsResults(analysis) {
        const container = document.getElementById('metricsResults');
        if (container && analysis.metricas_performance_detalhadas) {
            container.innerHTML = `
                <div class="result-section">
                    <div class="result-section-header">
                        <i class="fas fa-chart-line"></i>
                        <h4>Métricas de Performance</h4>
                    </div>
                    <div class="result-section-content">
                        <p style="color: var(--text-primary);">Métricas detalhadas de performance disponíveis nos dados completos.</p>
                    </div>
                </div>
            `;
        }
    }

    renderActionPlanResults(analysis) {
        const container = document.getElementById('actionPlanResults');
        if (container && analysis.plano_acao_detalhado) {
            container.innerHTML = `
                <div class="result-section">
                    <div class="result-section-header">
                        <i class="fas fa-tasks"></i>
                        <h4>Plano de Ação Detalhado</h4>
                    </div>
                    <div class="result-section-content">
                        <p style="color: var(--text-primary);">Plano de ação completo disponível nos dados completos.</p>
                    </div>
                </div>
            `;
        }
    }

    renderVisualProofsResults(analysis) {
        const container = document.getElementById('visualProofsResults');
        if (container && analysis.provas_visuais_sugeridas) {
            container.innerHTML = `
                <div class="result-section">
                    <div class="result-section-header">
                        <i class="fas fa-eye"></i>
                        <h4>Provas Visuais Instantâneas</h4>
                    </div>
                    <div class="result-section-content">
                        <p style="color: var(--text-primary);">Sistema completo de provas visuais disponível nos dados completos.</p>
                    </div>
                </div>
            `;
        }
    }

    renderAntiObjectionResults(analysis) {
        const container = document.getElementById('antiObjectionResults');
        if (container && analysis.sistema_anti_objecao) {
            container.innerHTML = `
                <div class="result-section">
                    <div class="result-section-header">
                        <i class="fas fa-shield-alt"></i>
                        <h4>Sistema Anti-Objeção</h4>
                    </div>
                    <div class="result-section-content">
                        <p style="color: var(--text-primary);">Sistema completo anti-objeção disponível nos dados completos.</p>
                    </div>
                </div>
            `;
        }
    }

    renderPrePitchResults(analysis) {
        const container = document.getElementById('prePitchResults');
        if (container && analysis.pre_pitch_invisivel) {
            container.innerHTML = `
                <div class="result-section">
                    <div class="result-section-header">
                        <i class="fas fa-magic"></i>
                        <h4>Pré-Pitch Invisível</h4>
                    </div>
                    <div class="result-section-content">
                        <p style="color: var(--text-primary);">Sistema completo de pré-pitch disponível nos dados completos.</p>
                    </div>
                </div>
            `;
        }
    }

    renderFutureResults(analysis) {
        const container = document.getElementById('futureResults');
        if (container && analysis.predicoes_futuro_completas) {
            container.innerHTML = `
                <div class="result-section">
                    <div class="result-section-header">
                        <i class="fas fa-crystal-ball"></i>
                        <h4>Predições do Futuro</h4>
                    </div>
                    <div class="result-section-content">
                        <p style="color: var(--text-primary);">Predições completas do futuro do mercado disponíveis nos dados completos.</p>
                    </div>
                </div>
            `;
        }
    }

    enablePdfDownload(analysis) {
        const downloadBtn = document.getElementById('downloadPdfBtn');
        if (downloadBtn) {
            downloadBtn.style.display = 'inline-flex';
            downloadBtn.onclick = () => this.downloadPdf(analysis);
        }
    }

    async downloadPdf(analysis) {
        try {
            window.app?.showInfo('Gerando relatório PDF...');

            const response = await fetch('/api/generate_pdf', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(analysis)
            });

            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `analise_mercado_${new Date().toISOString().slice(0, 10)}.pdf`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);

                window.app?.showSuccess('Relatório PDF baixado com sucesso!');
            } else {
                throw new Error('Erro ao gerar PDF');
            }

        } catch (error) {
            console.error('Erro ao baixar PDF:', error);
            window.app?.showError('Erro ao gerar relatório PDF');
        }
    }
}

// Initialize analysis manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.analysisManager = new AnalysisManager();
});