// ARQV30 Enhanced v2.0 - Upload JavaScript

class FileUploadManager {
    constructor() {
        this.maxFileSize = 16 * 1024 * 1024; // 16MB
        this.allowedTypes = [
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.ms-excel',
            'text/csv',
            'text/plain',
            'application/json'
        ];

        this.allowedExtensions = [
            '.pdf', '.doc', '.docx', '.xlsx', '.xls', '.csv', '.txt', '.json'
        ];

        this.init();
    }

    init() {
        this.setupDragAndDrop();
        this.setupFileInput();
    }

    setupDragAndDrop() {
        const uploadArea = document.getElementById('uploadArea');
        if (!uploadArea) return;

        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, this.preventDefaults, false);
            document.body.addEventListener(eventName, this.preventDefaults, false);
        });

        // Highlight drop area when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => this.highlight(uploadArea), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => this.unhighlight(uploadArea), false);
        });

        // Handle dropped files
        uploadArea.addEventListener('drop', (e) => this.handleDrop(e), false);
    }

    setupFileInput() {
        const fileInput = document.getElementById('fileInput');
        if (!fileInput) return;

        fileInput.addEventListener('change', (e) => {
            const files = Array.from(e.target.files);
            this.handleFiles(files);
        });
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    highlight(element) {
        element.classList.add('dragover');
    }

    unhighlight(element) {
        element.classList.remove('dragover');
    }

    handleDrop(e) {
        const dt = e.dataTransfer;
        const files = Array.from(dt.files);
        this.handleFiles(files);
    }

    handleFiles(files) {
        files.forEach(file => this.processFile(file));
    }

    processFile(file) {
        // Validate file
        const validation = this.validateFile(file);
        if (!validation.valid) {
            this.showError(validation.message);
            return;
        }

        // Show file in UI immediately
        this.addFileToUI(file);

        // Upload file
        this.uploadFile(file);
    }

    validateFile(file) {
        // Check file size
        if (file.size > this.maxFileSize) {
            return {
                valid: false,
                message: `Arquivo "${file.name}" é muito grande. Tamanho máximo: 16MB.`
            };
        }

        // Check file type
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        const isValidType = this.allowedTypes.includes(file.type) || 
                           this.allowedExtensions.includes(fileExtension);

        if (!isValidType) {
            return {
                valid: false,
                message: `Tipo de arquivo "${file.name}" não suportado. Tipos aceitos: PDF, DOC, DOCX, XLSX, XLS, CSV, TXT, JSON.`
            };
        }

        return { valid: true };
    }

    addFileToUI(file) {
        const container = document.getElementById('uploadedFiles');
        if (!container) return;

        const fileId = 'file_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);

        const fileElement = document.createElement('div');
        fileElement.className = 'file-item';
        fileElement.setAttribute('data-file-id', fileId);

        fileElement.innerHTML = `
            <div class="file-info">
                <i class="fas fa-file-alt"></i>
                <div>
                    <div class="file-name">${file.name}</div>
                    <div class="file-size">${this.formatFileSize(file.size)} • <span class="file-status">Processando...</span></div>
                </div>
            </div>
            <div class="file-actions">
                <div class="file-progress">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: 0%"></div>
                    </div>
                </div>
                <button class="file-remove" onclick="uploadManager.removeFile('${fileId}')">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;

        container.appendChild(fileElement);

        // Store file reference
        fileElement._fileData = {
            id: fileId,
            file: file,
            element: fileElement
        };
    }

    async uploadFile(file) {
        const fileElement = document.querySelector(`[data-file-id="${file._fileData?.id}"]`) || 
                           Array.from(document.querySelectorAll('.file-item')).find(el => 
                               el.querySelector('.file-name').textContent === file.name
                           );

        if (!fileElement) return;

        const progressFill = fileElement.querySelector('.progress-fill');
        const statusElement = fileElement.querySelector('.file-status');

        try {
            // Prepare form data
            const formData = new FormData();
            formData.append('file', file);
            formData.append('session_id', window.app?.sessionId || 'default_session');

            // Upload with progress tracking
            const xhr = new XMLHttpRequest();

            // Track upload progress
            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const percentComplete = (e.loaded / e.total) * 100;
                    if (progressFill) {
                        progressFill.style.width = percentComplete + '%';
                    }
                }
            });

            // Handle completion
            xhr.addEventListener('load', () => {
                if (xhr.status === 200) {
                    const response = JSON.parse(xhr.responseText);
                    if (response.success) {
                        this.onUploadSuccess(fileElement, response);
                    } else {
                        this.onUploadError(fileElement, response.error || 'Erro desconhecido');
                    }
                } else {
                    this.onUploadError(fileElement, 'Erro no servidor');
                }
            });

            // Handle errors
            xhr.addEventListener('error', () => {
                this.onUploadError(fileElement, 'Erro de conexão');
            });

            // Start upload
            xhr.open('POST', '/api/upload_attachment');
            xhr.send(formData);

        } catch (error) {
            console.error('Upload error:', error);
            this.onUploadError(fileElement, error.message);
        }
    }

    onUploadSuccess(fileElement, response) {
        // Update file status
        const statusElement = fileElement.querySelector('.file-status');
        if (statusElement) {
            statusElement.textContent = 'Concluído';
            statusElement.style.color = '#48bb78';
        }

        // Hide progress bar
        const progressContainer = fileElement.querySelector('.file-progress');
        if (progressContainer) {
            progressContainer.style.display = 'none';
        }

        // Add success indicator
        const fileInfo = fileElement.querySelector('.file-info');
        if (fileInfo) {
            const successIcon = document.createElement('i');
            successIcon.className = 'fas fa-check-circle';
            successIcon.style.color = '#48bb78';
            successIcon.style.marginLeft = '8px';
            fileInfo.appendChild(successIcon);
        }

        // Store response data
        fileElement._responseData = response;

        this.showSuccess(`Arquivo "${response.filename}" processado com sucesso!`);
    }

    onUploadError(fileElement, errorMessage) {
        const statusElement = fileElement.querySelector('.file-status');
        const progressContainer = fileElement.querySelector('.file-progress');

        if (statusElement) {
            statusElement.textContent = 'Erro no processamento';
            statusElement.style.color = '#f56565';
        }

        if (progressContainer) {
            progressContainer.style.display = 'none';
        }

        // Add error indicator
        const fileInfo = fileElement.querySelector('.file-info');
        if (fileInfo) {
            const errorIcon = document.createElement('i');
            errorIcon.className = 'fas fa-exclamation-circle';
            errorIcon.style.color = '#f56565';
            errorIcon.style.marginLeft = '8px';
            fileInfo.appendChild(errorIcon);
        }

        this.showError(`Erro ao processar arquivo: ${errorMessage}`);
    }

    removeFile(fileId) {
        const fileElement = document.querySelector(`[data-file-id="${fileId}"]`);
        if (fileElement) {
            fileElement.remove();
        }

        // Update app's uploaded files list if available
        if (window.app && window.app.removeFile) {
            window.app.removeFile(fileId);
        }
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';

        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));

        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    getFileIcon(filename) {
        const extension = filename.split('.').pop().toLowerCase();

        const iconMap = {
            'pdf': 'fas fa-file-pdf',
            'doc': 'fas fa-file-word',
            'docx': 'fas fa-file-word',
            'xls': 'fas fa-file-excel',
            'xlsx': 'fas fa-file-excel',
            'csv': 'fas fa-file-csv',
            'txt': 'fas fa-file-alt',
            'json': 'fas fa-file-code'
        };

        return iconMap[extension] || 'fas fa-file';
    }

    getUploadedFiles() {
        const fileElements = document.querySelectorAll('.file-item');
        const files = [];

        fileElements.forEach(element => {
            if (element._responseData) {
                files.push(element._responseData);
            }
        });

        return files;
    }

    clearAllFiles() {
        const container = document.getElementById('uploadedFiles');
        if (container) {
            container.innerHTML = '';
        }
    }

    showError(message) {
        if (window.app && window.app.showError) {
            window.app.showError(message);
        } else {
            console.error(message);
            alert(message);
        }
    }

    showSuccess(message) {
        if (window.app && window.app.showSuccess) {
            window.app.showSuccess(message);
        } else {
            console.log(message);
        }
    }
}

// Initialize upload manager
document.addEventListener('DOMContentLoaded', () => {
    window.uploadManager = new FileUploadManager();
});

// Add CSS for upload components
document.addEventListener('DOMContentLoaded', () => {
    if (!document.querySelector('#upload-styles')) {
        const styles = document.createElement('style');
        styles.id = 'upload-styles';
        styles.textContent = `
            .file-progress {
                margin-top: 8px;
                margin-bottom: 4px;
            }

            .progress-bar {
                width: 100%;
                height: 4px;
                background-color: #e2e8f0;
                border-radius: 2px;
                overflow: hidden;
            }

            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, #4299e1 0%, #38b2ac 100%);
                border-radius: 2px;
                transition: width 0.3s ease;
                width: 0%;
            }

            .file-actions {
                display: flex;
                flex-direction: column;
                align-items: flex-end;
                gap: 8px;
            }

            .file-item {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                padding: 12px 16px;
                background: #f7fafc;
                border-radius: 8px;
                margin-bottom: 8px;
                border: 1px solid #e2e8f0;
                transition: all 0.2s ease;
            }

            .file-item:hover {
                background: #edf2f7;
                border-color: #cbd5e0;
            }

            .file-info {
                display: flex;
                align-items: center;
                gap: 12px;
                flex: 1;
            }

            .file-info i {
                color: #4299e1;
                font-size: 18px;
            }

            .file-name {
                font-weight: 500;
                color: #2d3748;
                margin-bottom: 2px;
            }

            .file-size {
                font-size: 12px;
                color: #718096;
            }

            .file-status {
                font-weight: 500;
            }

            .file-remove {
                background: none;
                border: none;
                color: #a0aec0;
                cursor: pointer;
                padding: 8px;
                border-radius: 4px;
                transition: all 0.2s ease;
                flex-shrink: 0;
            }

            .file-remove:hover {
                background: rgba(245, 101, 101, 0.1);
                color: #f56565;
            }

            .upload-area.dragover {
                border-color: #4299e1;
                background-color: rgba(66, 153, 225, 0.05);
            }

            .upload-area.dragover .upload-content i {
                color: #4299e1;
            }
        `;
        document.head.appendChild(styles);
    }
});