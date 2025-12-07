/**
 * Orchester & Zuschauer Personenzähler
 * Personenerkennung mit YOLO (Backend) oder COCO-SSD (Browser)
 */

class PersonCounter {
    constructor() {
        this.model = null;
        this.isModelLoaded = false;
        this.currentMode = 'browser'; // 'browser' or 'backend'
        this.currentModelName = 'mobilenet_v2';
        this.minConfidence = 0.5;
        this.backendUrl = 'http://localhost:5000';

        // Model display names
        this.modelNames = {
            // Browser models (COCO-SSD)
            'lite_mobilenet_v2': 'Lite MobileNet v2',
            'mobilenet_v1': 'MobileNet v1',
            'mobilenet_v2': 'MobileNet v2',
            // Backend models (YOLO)
            'yolov8n.pt': 'YOLOv8 Nano',
            'yolov8s.pt': 'YOLOv8 Small',
            'yolov8m.pt': 'YOLOv8 Medium',
            'yolov8l.pt': 'YOLOv8 Large',
            'yolov8x.pt': 'YOLOv8 XLarge'
        };

        // DOM Elements
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.resultSection = document.getElementById('resultSection');
        this.loadingSection = document.getElementById('loadingSection');
        this.loadingText = document.getElementById('loadingText');
        this.outputCanvas = document.getElementById('outputCanvas');
        this.inputImage = document.getElementById('inputImage');
        this.personCount = document.getElementById('personCount');
        this.processingTime = document.getElementById('processingTime');
        this.avgConfidence = document.getElementById('avgConfidence');
        this.detectionList = document.getElementById('detectionList');
        this.modeSelect = document.getElementById('modeSelect');
        this.modelSelect = document.getElementById('modelSelect');
        this.modelStatus = document.getElementById('modelStatus');
        this.confidenceSlider = document.getElementById('confidenceSlider');
        this.confidenceValue = document.getElementById('confidenceValue');
        this.backendUrlInput = document.getElementById('backendUrl');
        this.backendUrlContainer = document.getElementById('backendUrlContainer');

        // Store last processed image for re-processing
        this.lastImageSrc = null;
        this.lastFile = null;

        this.init();
    }

    async init() {
        this.setupEventListeners();
        this.updateModelOptions();
        await this.loadBrowserModel(this.currentModelName);
    }

    setupEventListeners() {
        // File input
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

        // Drag and drop
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.classList.add('dragover');
        });

        this.uploadArea.addEventListener('dragleave', () => {
            this.uploadArea.classList.remove('dragover');
        });

        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.processFile(files[0]);
            }
        });

        // Demo buttons
        document.querySelectorAll('.demo-btn').forEach(btn => {
            btn.addEventListener('click', () => this.loadDemoImage(btn.dataset.demo));
        });

        // Mode selection
        this.modeSelect.addEventListener('change', async (e) => {
            this.currentMode = e.target.value;
            this.updateModelOptions();
            this.updateStatus();

            // Show/hide backend URL input
            this.backendUrlContainer.style.display = this.currentMode === 'backend' ? 'block' : 'none';

            if (this.currentMode === 'browser') {
                await this.loadBrowserModel(this.currentModelName);
            } else {
                this.isModelLoaded = true; // Backend handles model loading
            }

            // Re-process if image available
            if (this.lastImageSrc || this.lastFile) {
                this.reprocessLastImage();
            }
        });

        // Model selection
        this.modelSelect.addEventListener('change', async (e) => {
            const newModel = e.target.value;
            this.currentModelName = newModel;

            if (this.currentMode === 'browser') {
                await this.loadBrowserModel(newModel);
            }

            this.updateStatus();

            // Re-process if image available
            if (this.lastImageSrc || this.lastFile) {
                this.reprocessLastImage();
            }
        });

        // Confidence slider
        this.confidenceSlider.addEventListener('input', (e) => {
            this.minConfidence = parseInt(e.target.value) / 100;
            this.confidenceValue.textContent = `${e.target.value}%`;

            // Re-process if image available
            if ((this.lastImageSrc || this.lastFile) && this.isModelLoaded) {
                this.reprocessLastImage();
            }
        });

        // Backend URL
        this.backendUrlInput.addEventListener('change', (e) => {
            this.backendUrl = e.target.value.replace(/\/$/, ''); // Remove trailing slash
        });
    }

    updateModelOptions() {
        const options = this.modelSelect.options;
        for (let i = 0; i < options.length; i++) {
            const option = options[i];
            const mode = option.dataset.mode;
            option.style.display = mode === this.currentMode ? '' : 'none';

            // Select first visible option if current is hidden
            if (option.selected && mode !== this.currentMode) {
                // Find first option for current mode
                for (let j = 0; j < options.length; j++) {
                    if (options[j].dataset.mode === this.currentMode) {
                        options[j].selected = true;
                        this.currentModelName = options[j].value;
                        break;
                    }
                }
            }
        }
    }

    updateStatus() {
        const modeName = this.currentMode === 'browser' ? 'Browser' : 'Backend (YOLO)';
        const modelName = this.modelNames[this.currentModelName] || this.currentModelName;
        this.modelStatus.textContent = `Modus: ${modeName} | Modell: ${modelName}`;
        this.modelStatus.className = 'model-status';
    }

    async loadBrowserModel(modelName) {
        this.isModelLoaded = false;
        this.modelStatus.textContent = `Lade ${this.modelNames[modelName]}...`;
        this.modelStatus.className = 'model-status loading';
        this.showLoading(`${this.modelNames[modelName]} wird geladen...`);

        try {
            if (this.model) {
                this.model = null;
            }

            this.model = await cocoSsd.load({ base: modelName });
            this.currentModelName = modelName;
            this.isModelLoaded = true;
            this.updateStatus();
            this.hideLoading();
            console.log(`COCO-SSD ${modelName} erfolgreich geladen`);
        } catch (error) {
            console.error('Fehler beim Laden des Modells:', error);
            this.modelStatus.textContent = 'Fehler beim Laden des Modells';
            this.modelStatus.className = 'model-status error';
            this.loadingText.textContent = 'Fehler beim Laden des Modells. Bitte Seite neu laden.';
        }
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    processFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Bitte wählen Sie eine Bilddatei aus.');
            return;
        }

        this.lastFile = file;

        if (this.currentMode === 'backend') {
            this.processWithBackend(file);
        } else {
            const reader = new FileReader();
            reader.onload = (e) => {
                this.lastImageSrc = e.target.result;
                this.processWithBrowser(e.target.result);
            };
            reader.readAsDataURL(file);
        }
    }

    reprocessLastImage() {
        if (this.currentMode === 'backend' && this.lastFile) {
            this.processWithBackend(this.lastFile);
        } else if (this.lastImageSrc) {
            this.processWithBrowser(this.lastImageSrc);
        }
    }

    async loadDemoImage(type) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        switch (type) {
            case 'orchestra':
                canvas.width = 800;
                canvas.height = 600;
                this.drawOrchestraDemo(ctx, canvas.width, canvas.height);
                break;
            case 'audience':
                canvas.width = 900;
                canvas.height = 700;
                this.drawAudienceDemo(ctx, canvas.width, canvas.height);
                break;
            case 'group':
                canvas.width = 400;
                canvas.height = 300;
                this.drawGroupDemo(ctx, canvas.width, canvas.height);
                break;
        }

        const dataUrl = canvas.toDataURL('image/jpeg');
        this.lastImageSrc = dataUrl;

        if (this.currentMode === 'backend') {
            // Convert canvas to blob for backend
            canvas.toBlob((blob) => {
                this.lastFile = new File([blob], 'demo.jpg', { type: 'image/jpeg' });
                this.processWithBackend(this.lastFile);
            }, 'image/jpeg');
        } else {
            this.processWithBrowser(dataUrl);
        }
    }

    // ===== Browser Processing (COCO-SSD) =====

    async processWithBrowser(imageSrc) {
        if (!this.isModelLoaded) {
            alert('Modell wird noch geladen. Bitte warten...');
            return;
        }

        this.showLoading('Personen werden erkannt...');

        const img = new Image();
        img.crossOrigin = 'anonymous';

        img.onload = async () => {
            const startTime = performance.now();

            this.outputCanvas.width = img.width;
            this.outputCanvas.height = img.height;
            const ctx = this.outputCanvas.getContext('2d');
            ctx.drawImage(img, 0, 0);

            const predictions = await this.model.detect(img);
            const persons = predictions.filter(
                p => p.class === 'person' && p.score >= this.minConfidence
            );

            const endTime = performance.now();
            const processingTimeMs = endTime - startTime;

            this.drawDetections(ctx, persons, img.width, img.height);
            this.updateResults(persons, processingTimeMs);

            this.hideLoading();
            this.resultSection.style.display = 'block';
            this.resultSection.scrollIntoView({ behavior: 'smooth' });
        };

        img.onerror = () => {
            this.hideLoading();
            alert('Fehler beim Laden des Bildes.');
        };

        img.src = imageSrc;
    }

    // ===== Backend Processing (YOLO) =====

    async processWithBackend(file) {
        this.showLoading('Sende an YOLO-Backend...');

        const formData = new FormData();
        formData.append('image', file);

        const url = `${this.backendUrl}/api/detect?model=${this.currentModelName}&confidence=${this.minConfidence}&annotate=true`;

        try {
            const startTime = performance.now();

            const response = await fetch(url, {
                method: 'POST',
                body: formData
            });

            const endTime = performance.now();

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Backend-Fehler');
            }

            const result = await response.json();

            if (result.annotated_image) {
                // Display the annotated image from backend
                const img = new Image();
                img.onload = () => {
                    this.outputCanvas.width = img.width;
                    this.outputCanvas.height = img.height;
                    const ctx = this.outputCanvas.getContext('2d');
                    ctx.drawImage(img, 0, 0);

                    // Convert backend detections format
                    const detections = result.detections.map(d => ({
                        score: d.confidence,
                        bbox: d.bbox
                    }));

                    this.updateResults(detections, endTime - startTime);

                    this.hideLoading();
                    this.resultSection.style.display = 'block';
                    this.resultSection.scrollIntoView({ behavior: 'smooth' });
                };
                img.src = result.annotated_image;
            }

        } catch (error) {
            this.hideLoading();
            console.error('Backend-Fehler:', error);

            if (error.message.includes('Failed to fetch')) {
                alert(`Backend nicht erreichbar unter ${this.backendUrl}\n\nStarten Sie den Server mit:\npython api_server.py`);
            } else {
                alert(`Fehler: ${error.message}`);
            }
        }
    }

    // ===== Drawing Functions =====

    drawOrchestraDemo(ctx, width, height) {
        ctx.fillStyle = '#282830';
        ctx.fillRect(0, 0, width, height);
        ctx.fillStyle = '#4a3728';
        ctx.fillRect(0, height / 2, width, height / 2);

        const rows = 3;
        const personsPerRow = 8;

        for (let row = 0; row < rows; row++) {
            const yBase = 200 + row * 120;
            const xSpacing = width / (personsPerRow + 1);

            for (let i = 0; i < personsPerRow; i++) {
                const x = xSpacing * (i + 1) + (Math.random() - 0.5) * 20;
                this.drawPerson(ctx, x, yBase, 15 + row * 2);
            }
        }
    }

    drawAudienceDemo(ctx, width, height) {
        ctx.fillStyle = '#1e1923';
        ctx.fillRect(0, 0, width, height);

        const rows = 5;
        const seatsPerRow = 12;

        for (let row = 0; row < rows; row++) {
            const yBase = 100 + row * 110;
            const xSpacing = width / (seatsPerRow + 1);

            for (let i = 0; i < seatsPerRow; i++) {
                if (Math.random() > 0.1) {
                    const x = xSpacing * (i + 1);
                    this.drawPerson(ctx, x, yBase, 12, true);
                }
            }

            ctx.fillStyle = '#3c2820';
            ctx.fillRect(20, yBase + 60, width - 40, 15);
        }
    }

    drawGroupDemo(ctx, width, height) {
        ctx.fillStyle = '#6496c8';
        ctx.fillRect(0, 0, width, height * 2 / 3);
        ctx.fillStyle = '#507850';
        ctx.fillRect(0, height * 2 / 3, width, height / 3);

        const numPersons = 5;
        const xSpacing = width / (numPersons + 1);

        for (let i = 0; i < numPersons; i++) {
            const x = xSpacing * (i + 1);
            this.drawPerson(ctx, x, height * 2 / 3, 15, true);
        }
    }

    drawPerson(ctx, x, yBase, headRadius, colorful = false) {
        ctx.fillStyle = `rgb(${200 + Math.random() * 20}, ${160 + Math.random() * 20}, ${140 + Math.random() * 20})`;
        ctx.beginPath();
        ctx.arc(x, yBase - 30, headRadius, 0, Math.PI * 2);
        ctx.fill();

        if (colorful) {
            ctx.fillStyle = `rgb(${50 + Math.random() * 150}, ${50 + Math.random() * 150}, ${50 + Math.random() * 150})`;
        } else {
            ctx.fillStyle = '#141414';
        }
        ctx.fillRect(x - 20, yBase - 30 + headRadius, 40, 60);
    }

    drawDetections(ctx, detections, imgWidth, imgHeight) {
        const scaleFactor = Math.max(imgWidth, imgHeight) / 800;
        const fontSize = Math.max(14, Math.round(16 * scaleFactor));
        const lineWidth = Math.max(2, Math.round(3 * scaleFactor));
        const padding = Math.max(4, Math.round(5 * scaleFactor));

        detections.forEach((detection, index) => {
            const [x, y, width, height] = detection.bbox;
            const confidence = detection.score;

            let color, bgColor;
            if (confidence >= 0.7) {
                color = '#22c55e';
                bgColor = 'rgba(34, 197, 94, 0.9)';
            } else if (confidence >= 0.5) {
                color = '#f59e0b';
                bgColor = 'rgba(245, 158, 11, 0.9)';
            } else {
                color = '#ef4444';
                bgColor = 'rgba(239, 68, 68, 0.9)';
            }

            ctx.strokeStyle = color;
            ctx.lineWidth = lineWidth;
            ctx.strokeRect(x, y, width, height);

            const cornerLength = Math.min(width, height) * 0.2;
            ctx.lineWidth = lineWidth + 2;

            ctx.beginPath();
            ctx.moveTo(x, y + cornerLength);
            ctx.lineTo(x, y);
            ctx.lineTo(x + cornerLength, y);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(x + width - cornerLength, y);
            ctx.lineTo(x + width, y);
            ctx.lineTo(x + width, y + cornerLength);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(x, y + height - cornerLength);
            ctx.lineTo(x, y + height);
            ctx.lineTo(x + cornerLength, y + height);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(x + width - cornerLength, y + height);
            ctx.lineTo(x + width, y + height);
            ctx.lineTo(x + width, y + height - cornerLength);
            ctx.stroke();

            const label = `#${index + 1}`;
            const confLabel = `${(confidence * 100).toFixed(0)}%`;

            ctx.font = `bold ${fontSize}px Arial, sans-serif`;
            const numberWidth = ctx.measureText(label).width;
            const confWidth = ctx.measureText(confLabel).width;
            const labelHeight = fontSize + padding * 2;

            this.drawRoundedRect(ctx, x, y - labelHeight - 2, numberWidth + padding * 2, labelHeight, 4, bgColor);
            ctx.fillStyle = '#ffffff';
            ctx.fillText(label, x + padding, y - padding - 4);

            this.drawRoundedRect(ctx, x + width - confWidth - padding * 2, y - labelHeight - 2, confWidth + padding * 2, labelHeight, 4, bgColor);
            ctx.fillStyle = '#ffffff';
            ctx.fillText(confLabel, x + width - confWidth - padding, y - padding - 4);
        });

        this.drawTotalCountOverlay(ctx, detections.length, imgWidth, imgHeight, scaleFactor);
    }

    drawRoundedRect(ctx, x, y, width, height, radius, fillColor) {
        ctx.fillStyle = fillColor;
        ctx.beginPath();
        ctx.moveTo(x + radius, y);
        ctx.lineTo(x + width - radius, y);
        ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
        ctx.lineTo(x + width, y + height - radius);
        ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
        ctx.lineTo(x + radius, y + height);
        ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
        ctx.lineTo(x, y + radius);
        ctx.quadraticCurveTo(x, y, x + radius, y);
        ctx.closePath();
        ctx.fill();
    }

    drawTotalCountOverlay(ctx, count, imgWidth, imgHeight, scaleFactor) {
        const fontSize = Math.max(24, Math.round(32 * scaleFactor));
        const padding = Math.max(10, Math.round(15 * scaleFactor));

        const text = `${count} Person${count !== 1 ? 'en' : ''} erkannt`;
        ctx.font = `bold ${fontSize}px Arial, sans-serif`;
        const textWidth = ctx.measureText(text).width;

        const boxWidth = textWidth + padding * 2;
        const boxHeight = fontSize + padding * 2;
        const boxX = (imgWidth - boxWidth) / 2;
        const boxY = padding;

        this.drawRoundedRect(ctx, boxX, boxY, boxWidth, boxHeight, 8, 'rgba(0, 0, 0, 0.75)');

        ctx.strokeStyle = count > 0 ? '#22c55e' : '#94a3b8';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(boxX + 8, boxY);
        ctx.lineTo(boxX + boxWidth - 8, boxY);
        ctx.quadraticCurveTo(boxX + boxWidth, boxY, boxX + boxWidth, boxY + 8);
        ctx.lineTo(boxX + boxWidth, boxY + boxHeight - 8);
        ctx.quadraticCurveTo(boxX + boxWidth, boxY + boxHeight, boxX + boxWidth - 8, boxY + boxHeight);
        ctx.lineTo(boxX + 8, boxY + boxHeight);
        ctx.quadraticCurveTo(boxX, boxY + boxHeight, boxX, boxY + boxHeight - 8);
        ctx.lineTo(boxX, boxY + 8);
        ctx.quadraticCurveTo(boxX, boxY, boxX + 8, boxY);
        ctx.closePath();
        ctx.stroke();

        ctx.fillStyle = count > 0 ? '#22c55e' : '#94a3b8';
        ctx.fillText(text, boxX + padding, boxY + padding + fontSize * 0.75);
    }

    updateResults(detections, processingTimeMs) {
        const countNumber = this.personCount.querySelector('.count-number');
        countNumber.textContent = detections.length;

        this.processingTime.textContent = `${processingTimeMs.toFixed(0)} ms`;

        if (detections.length > 0) {
            const avgConf = detections.reduce((sum, d) => sum + d.score, 0) / detections.length;
            this.avgConfidence.textContent = `${(avgConf * 100).toFixed(1)}%`;
        } else {
            this.avgConfidence.textContent = '-';
        }

        this.detectionList.innerHTML = detections.map((d, i) => {
            const conf = d.score * 100;
            let confClass = 'low';
            if (conf >= 70) confClass = 'high';
            else if (conf >= 50) confClass = 'medium';

            return `
                <div class="detection-item">
                    <span class="detection-id">Person #${i + 1}</span>
                    <span class="detection-confidence ${confClass}">${conf.toFixed(1)}%</span>
                </div>
            `;
        }).join('');
    }

    showLoading(text) {
        this.loadingText.textContent = text;
        this.loadingSection.style.display = 'block';
        this.resultSection.style.display = 'none';
    }

    hideLoading() {
        this.loadingSection.style.display = 'none';
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new PersonCounter();
});
