// ARQV30 Enhanced v2.0 - Main JavaScript with Dark Neumorphic UI

class ARQVApp {
    constructor() {
        this.sessionId = this.generateSessionId();
        this.isOnline = false;
        this.systemStatus = {};
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.checkSystemStatus();
        this.initializeUI();
        this.startStatusMonitoring();
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
                if (window.analysisManager) {
                    window.analysisManager.startAnalysis();
                }
            });
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                e.preventDefault();
                const analyzeBtn = document.getElementById('analyzeBtn');
                if (analyzeBtn && !analyzeBtn.disabled) {
                    analyzeBtn.click();
                }
            }
        });

        // Smooth scrolling for internal links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                        console.log('✅ Service Worker registered successfully');
                        
                        // Check for updates
                        registration.addEventListener('updatefound', () => {
                            const newWorker = registration.installing;
                            newWorker.addEventListener('statechange', () => {
                                if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                                    // New version available
                                    if (window.app) {
                                        window.app.showInfo('Nova versão disponível! Recarregue a página para atualizar.');
                                    }
                                }
                            });
                        });
                    target.scrollIntoView({
                        behavior: 'smooth',
                        console.warn('⚠️ Service Worker registration failed:', registrationError);
                    });
                }
            });
        });

        // Enhanced form interactions
        this.setupFormEnhancements();
    }

    setupFormEnhancements() {
        // Auto-resize textareas
        document.querySelectorAll('textarea').forEach(textarea => {
            textarea.addEventListener('input', () => {
                textarea.style.height = 'auto';
                textarea.style.height = textarea.scrollHeight + 'px';
            });
        });

        // Input focus effects
        document.querySelectorAll('input, textarea, select').forEach(input => {
            input.addEventListener('focus', () => {
                input.parentElement.classList.add('focused');
            });

            input.addEventListener('blur', () => {
                input.parentElement.classList.remove('focused');
            });

            // Real-time validation
            input.addEventListener('input', () => {
                this.validateField(input);
            });
        });

        // Auto-generate query if not provided
        const segmentoInput = document.getElementById('segmento');
        const produtoInput = document.getElementById('produto');
        const queryInput = document.getElementById('query');

        if (segmentoInput && queryInput) {
            const updateQuery = () => {
                if (!queryInput.value.trim()) {
                    const segmento = segmentoInput.value.trim();
                    const produto = produtoInput?.value.trim() || '';
                    
                    if (segmento) {
                        let autoQuery = `análise mercado ${segmento}`;
                        if (produto) {
                            autoQuery += ` ${produto}`;
                        }
                        autoQuery += ' Brasil tendências oportunidades 2024';
                        queryInput.value = autoQuery;
                        queryInput.style.fontStyle = 'italic';
                        queryInput.style.opacity = '0.8';
                    }
                }
            };

            segmentoInput.addEventListener('blur', updateQuery);
            if (produtoInput) {
                produtoInput.addEventListener('blur', updateQuery);
            }

            queryInput.addEventListener('focus', () => {
                queryInput.style.fontStyle = 'normal';
                queryInput.style.opacity = '1';
            });
        } else {
            console.warn('⚠️ Service Worker not supported in this browser');
        }
    }

    validateField(input) {
        const value = input.value.trim();
        const fieldName = input.name || input.id;
        
        // Remove previous validation classes
        input.classList.remove('valid', 'invalid');
        
        // Basic validation rules
        let isValid = true;
        
        if (input.hasAttribute('required') && !value) {
            isValid = false;
        }
        
        if (input.type === 'email' && value && !this.isValidEmail(value)) {
            isValid = false;
        }
        
        if (input.type === 'number' && value && isNaN(value)) {
            isValid = false;
        }
        
        // Apply validation class
        input.classList.add(isValid ? 'valid' : 'invalid');
        
        return isValid;
    }

    isValidEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }

    async checkSystemStatus() {
        try {
            const response = await fetch('/api/app_status');
            if (response.ok) {
                this.systemStatus = await response.json();
                this.updateStatusIndicator(true);
                this.isOnline = true;
            } else {
                throw new Error('Status check failed');
            }
        } catch (error) {
            console.error('System status check failed:', error);
            this.updateStatusIndicator(false);
            this.isOnline = false;
        }
    }

    updateStatusIndicator(isOnline) {
        const indicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');
        
        if (indicator && statusText) {
            indicator.className = 'status-indicator';
            
            if (isOnline) {
                indicator.classList.add('online');
                statusText.textContent = 'Sistema Online';
            } else {
                indicator.classList.add('offline');
                statusText.textContent = 'Sistema Offline';
            }
        }
    }

    startStatusMonitoring() {
        // Check status every 30 seconds
        setInterval(() => {
            this.checkSystemStatus();
        }, 30000);
    }

    initializeUI() {
        // Add loading states to buttons
        document.querySelectorAll('.btn-primary').forEach(btn => {
            btn.addEventListener('click', () => {
                if (!btn.disabled) {
                    this.setButtonLoading(btn, true);
                }
            });
        });

        // Initialize tooltips
        this.initializeTooltips();

        // Initialize expandable sections
        this.initializeExpandableSections();

        // Add cyber effects
        this.addCyberEffects();

        // Initialize particle background
        this.initializeParticleBackground();
    }

    setButtonLoading(button, loading) {
        if (loading) {
            button.disabled = true;
            button.classList.add('loading');
            button.dataset.originalText = button.innerHTML;
            button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processando...';
        } else {
            button.disabled = false;
            button.classList.remove('loading');
            if (button.dataset.originalText) {
                button.innerHTML = button.dataset.originalText;
            }
        }
    }

    initializeTooltips() {
        // Simple tooltip implementation
        document.querySelectorAll('[data-tooltip]').forEach(element => {
            element.addEventListener('mouseenter', (e) => {
                this.showTooltip(e.target, e.target.dataset.tooltip);
            });

            element.addEventListener('mouseleave', () => {
                this.hideTooltip();
            });
        });
    }

    showTooltip(element, text) {
        const tooltip = document.createElement('div');
        tooltip.className = 'tooltip-popup';
        tooltip.textContent = text;
        tooltip.style.cssText = `
            position: absolute;
            background: var(--bg-elevated);
            color: var(--text-primary);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            z-index: 1000;
            box-shadow: var(--shadow-floating);
            border: 1px solid var(--glass-border);
            backdrop-filter: var(--glass-blur);
        `;

        document.body.appendChild(tooltip);

        const rect = element.getBoundingClientRect();
        tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
        tooltip.style.top = rect.top - tooltip.offsetHeight - 8 + 'px';

        this.currentTooltip = tooltip;
    }

    hideTooltip() {
        if (this.currentTooltip) {
            this.currentTooltip.remove();
            this.currentTooltip = null;
        }
    }

    initializeExpandableSections() {
        document.querySelectorAll('.expandable-header').forEach(header => {
            header.addEventListener('click', () => {
                const section = header.closest('.expandable-section');
                section.classList.toggle('expanded');
            });
        });
    }

    addCyberEffects() {
        // Add glitch effect to title
        const title = document.querySelector('.hero-title');
        if (title) {
            title.setAttribute('data-text', title.textContent);
            title.classList.add('glitch');
        }

        // Add scan lines to containers
        document.querySelectorAll('.form-container, .results-container').forEach(container => {
            container.classList.add('scan-lines');
        });

        // Add cyber borders to important elements
        document.querySelectorAll('.section-title').forEach(title => {
            title.classList.add('cyber-border');
        });
    }

    initializeParticleBackground() {
        // Simple particle system for background
        const canvas = document.createElement('canvas');
        canvas.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            pointer-events: none;
            opacity: 0.3;
        `;
        document.body.appendChild(canvas);

        const ctx = canvas.getContext('2d');
        const particles = [];

        const resizeCanvas = () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        };

        const createParticle = () => ({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 0.5,
            vy: (Math.random() - 0.5) * 0.5,
            size: Math.random() * 2 + 1,
            opacity: Math.random() * 0.5 + 0.2
        });

        const initParticles = () => {
            particles.length = 0;
            for (let i = 0; i < 50; i++) {
                particles.push(createParticle());
            }
        };

        const updateParticles = () => {
            particles.forEach(particle => {
                particle.x += particle.vx;
                particle.y += particle.vy;

                if (particle.x < 0 || particle.x > canvas.width) particle.vx *= -1;
                if (particle.y < 0 || particle.y > canvas.height) particle.vy *= -1;
            });
        };

        const drawParticles = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            particles.forEach(particle => {
                ctx.beginPath();
                ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(0, 212, 255, ${particle.opacity})`;
                ctx.fill();
            });

            // Draw connections
            particles.forEach((particle, i) => {
                particles.slice(i + 1).forEach(otherParticle => {
                    const dx = particle.x - otherParticle.x;
                    const dy = particle.y - otherParticle.y;
                    const distance = Math.sqrt(dx * dx + dy * dy);

                    if (distance < 100) {
                        ctx.beginPath();
                        ctx.moveTo(particle.x, particle.y);
                        ctx.lineTo(otherParticle.x, otherParticle.y);
                        ctx.strokeStyle = `rgba(0, 212, 255, ${0.1 * (1 - distance / 100)})`;
                        ctx.lineWidth = 1;
                        ctx.stroke();
                    }
                });
            });
        };

        const animate = () => {
            updateParticles();
            drawParticles();
            requestAnimationFrame(animate);
        };

        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();
        initParticles();
        animate();
    }

    showError(message) {
        this.showAlert(message, 'error');
    }

    showSuccess(message) {
        this.showAlert(message, 'success');
    }

    showWarning(message) {
        this.showAlert(message, 'warning');
    }

    showInfo(message) {
        this.showAlert(message, 'info');
    }

    showAlert(message, type = 'info') {
        // Remove existing alerts
        document.querySelectorAll('.alert').forEach(alert => alert.remove());

        const alert = document.createElement('div');
        alert.className = `alert alert-${type}`;
        
        const icon = {
            error: 'fas fa-exclamation-triangle',
            success: 'fas fa-check-circle',
            warning: 'fas fa-exclamation-circle',
            info: 'fas fa-info-circle'
        }[type];

        alert.innerHTML = `
            <div style="display: flex; align-items: center; gap: 12px;">
                <i class="${icon}"></i>
                <span>${message}</span>
                <button onclick="this.parentElement.parentElement.remove()" style="
                    background: none; 
                    border: none; 
                    color: inherit; 
                    font-size: 18px; 
                    cursor: pointer;
                    margin-left: auto;
                ">×</button>
            </div>
        `;

        document.body.appendChild(alert);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alert.parentElement) {
                alert.remove();
            }
        }, 5000);
    }

    // Utility methods
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }

    copyToClipboard(text) {
        navigator.clipboard.writeText(text).then(() => {
            this.showSuccess('Copiado para a área de transferência!');
        }).catch(() => {
            this.showError('Erro ao copiar para a área de transferência');
        });
    }

    downloadJSON(data, filename) {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    // Theme management
    toggleTheme() {
        document.body.classList.toggle('light-theme');
        localStorage.setItem('theme', document.body.classList.contains('light-theme') ? 'light' : 'dark');
    }

    loadTheme() {
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'light') {
            document.body.classList.add('light-theme');
        }
    }

    // Performance monitoring
    startPerformanceMonitoring() {
        if ('performance' in window) {
            const observer = new PerformanceObserver((list) => {
                list.getEntries().forEach((entry) => {
                    if (entry.entryType === 'navigation') {
                        console.log('Page load time:', entry.loadEventEnd - entry.loadEventStart, 'ms');
                    }
                });
            });
            observer.observe({ entryTypes: ['navigation'] });
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new ARQVApp();
    
    // Add CSS for dynamic styles
    const dynamicStyles = document.createElement('style');
    dynamicStyles.textContent = `
        .input-group.focused input,
        .input-group.focused textarea,
        .input-group.focused select {
            border-color: var(--accent-primary);
            box-shadow: var(--shadow-inset), 0 0 0 3px rgba(0, 212, 255, 0.2);
            background: rgba(0, 212, 255, 0.05);
        }
        
        .input-group input.valid,
        .input-group textarea.valid,
        .input-group select.valid {
            border-color: var(--accent-tertiary);
        }
        
        .input-group input.invalid,
        .input-group textarea.invalid,
        .input-group select.invalid {
            border-color: #ff6b6b;
            box-shadow: var(--shadow-inset), 0 0 0 3px rgba(255, 107, 107, 0.2);
        }
        
        .tooltip-popup {
            animation: fadeIn 0.2s ease-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    `;
    document.head.appendChild(dynamicStyles);
});

// Global error handler
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    if (window.app) {
        window.app.showError('Ocorreu um erro inesperado. Verifique o console para mais detalhes.');
    }
});

// Service worker registration (if available)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then((registration) => {
                console.log('SW registered: ', registration);
            })
            .catch((registrationError) => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}