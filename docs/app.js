/**
 * Orchester & Zuschauer Personenzähler
 * Browser-basierte Personenerkennung mit TensorFlow.js und COCO-SSD
 */

class PersonCounter {
    constructor() {
        this.model = null;
        this.isModelLoaded = false;

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

        this.init();
    }

    async init() {
        this.setupEventListeners();
        await this.loadModel();
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
    }

    async loadModel() {
        this.showLoading('KI-Modell wird geladen...');

        try {
            this.model = await cocoSsd.load();
            this.isModelLoaded = true;
            this.hideLoading();
            console.log('COCO-SSD Modell erfolgreich geladen');
        } catch (error) {
            console.error('Fehler beim Laden des Modells:', error);
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

        const reader = new FileReader();
        reader.onload = (e) => {
            this.processImage(e.target.result);
        };
        reader.readAsDataURL(file);
    }

    async loadDemoImage(type) {
        // Generate demo images using canvas
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

        this.processImage(canvas.toDataURL('image/jpeg'));
    }

    drawOrchestraDemo(ctx, width, height) {
        // Background
        ctx.fillStyle = '#282830';
        ctx.fillRect(0, 0, width, height);

        // Stage floor
        ctx.fillStyle = '#4a3728';
        ctx.fillRect(0, height / 2, width, height / 2);

        // Draw orchestra members
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
        // Background
        ctx.fillStyle = '#1e1923';
        ctx.fillRect(0, 0, width, height);

        const rows = 5;
        const seatsPerRow = 12;

        for (let row = 0; row < rows; row++) {
            const yBase = 100 + row * 110;
            const xSpacing = width / (seatsPerRow + 1);

            for (let i = 0; i < seatsPerRow; i++) {
                if (Math.random() > 0.1) { // 90% occupancy
                    const x = xSpacing * (i + 1);
                    this.drawPerson(ctx, x, yBase, 12, true);
                }
            }

            // Seat backs
            ctx.fillStyle = '#3c2820';
            ctx.fillRect(20, yBase + 60, width - 40, 15);
        }
    }

    drawGroupDemo(ctx, width, height) {
        // Sky
        ctx.fillStyle = '#6496c8';
        ctx.fillRect(0, 0, width, height * 2 / 3);

        // Ground
        ctx.fillStyle = '#507850';
        ctx.fillRect(0, height * 2 / 3, width, height / 3);

        // Draw people
        const numPersons = 5;
        const xSpacing = width / (numPersons + 1);

        for (let i = 0; i < numPersons; i++) {
            const x = xSpacing * (i + 1);
            this.drawPerson(ctx, x, height * 2 / 3, 15, true);
        }
    }

    drawPerson(ctx, x, yBase, headRadius, colorful = false) {
        // Head
        ctx.fillStyle = `rgb(${200 + Math.random() * 20}, ${160 + Math.random() * 20}, ${140 + Math.random() * 20})`;
        ctx.beginPath();
        ctx.arc(x, yBase - 30, headRadius, 0, Math.PI * 2);
        ctx.fill();

        // Body
        if (colorful) {
            ctx.fillStyle = `rgb(${50 + Math.random() * 150}, ${50 + Math.random() * 150}, ${50 + Math.random() * 150})`;
        } else {
            ctx.fillStyle = '#141414';
        }
        ctx.fillRect(x - 20, yBase - 30 + headRadius, 40, 60);
    }

    async processImage(imageSrc) {
        if (!this.isModelLoaded) {
            alert('Modell wird noch geladen. Bitte warten...');
            return;
        }

        this.showLoading('Personen werden erkannt...');

        // Load image
        const img = new Image();
        img.crossOrigin = 'anonymous';

        img.onload = async () => {
            const startTime = performance.now();

            // Set up canvas
            this.outputCanvas.width = img.width;
            this.outputCanvas.height = img.height;
            const ctx = this.outputCanvas.getContext('2d');
            ctx.drawImage(img, 0, 0);

            // Run detection
            const predictions = await this.model.detect(img);

            // Filter for persons only
            const persons = predictions.filter(p => p.class === 'person');

            const endTime = performance.now();
            const processingTimeMs = endTime - startTime;

            // Draw results
            this.drawDetections(ctx, persons);

            // Update UI
            this.updateResults(persons, processingTimeMs);

            this.hideLoading();
            this.resultSection.style.display = 'block';

            // Scroll to results
            this.resultSection.scrollIntoView({ behavior: 'smooth' });
        };

        img.onerror = () => {
            this.hideLoading();
            alert('Fehler beim Laden des Bildes.');
        };

        img.src = imageSrc;
    }

    drawDetections(ctx, detections) {
        detections.forEach((detection, index) => {
            const [x, y, width, height] = detection.bbox;
            const confidence = detection.score;

            // Determine color based on confidence
            let color;
            if (confidence >= 0.7) {
                color = '#22c55e'; // Green
            } else if (confidence >= 0.5) {
                color = '#f59e0b'; // Orange
            } else {
                color = '#ef4444'; // Red
            }

            // Draw bounding box
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, width, height);

            // Draw label background
            const label = `#${index + 1} ${(confidence * 100).toFixed(0)}%`;
            ctx.font = 'bold 14px sans-serif';
            const textWidth = ctx.measureText(label).width;

            ctx.fillStyle = color;
            ctx.fillRect(x, y - 24, textWidth + 10, 24);

            // Draw label text
            ctx.fillStyle = '#ffffff';
            ctx.fillText(label, x + 5, y - 7);
        });
    }

    updateResults(detections, processingTimeMs) {
        // Update count
        const countNumber = this.personCount.querySelector('.count-number');
        countNumber.textContent = detections.length;

        // Update stats
        this.processingTime.textContent = `${processingTimeMs.toFixed(0)} ms`;

        if (detections.length > 0) {
            const avgConf = detections.reduce((sum, d) => sum + d.score, 0) / detections.length;
            this.avgConfidence.textContent = `${(avgConf * 100).toFixed(1)}%`;
        } else {
            this.avgConfidence.textContent = '-';
        }

        // Update detection list
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
