// Factory Voice AI - Complete Frontend Application
// Supports: Milestone 1 (Audio Pipeline), Milestone 2 (STT), Milestone 3 (TTS), Milestone 4 (Chatbot)

// Polyfill for older browsers
(function() {
    if (navigator.mediaDevices?.getUserMedia) return;
    if (!navigator.mediaDevices) navigator.mediaDevices = {};
    const legacy = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
    if (legacy) {
        navigator.mediaDevices.getUserMedia = (c) => new Promise((r, e) => legacy.call(navigator, c, r, e));
    }
})();

class FactoryVoiceAI {
    constructor() {
        this.ws = null;
        this.isRecording = false;
        this.isSpeaking = false;
        this.mediaStream = null;
        this.audioContext = null;
        this.ttsAudio = null;
        
        // State management
        this.currentLanguage = 'auto';
        this.detectedLanguage = null;
        this.currentTranscription = '';
        this.interimTranscription = '';
        
        // Statistics
        this.stats = {
            chunksProcessed: 0,
            speechDetected: 0,
            transcriptions: 0,
            avgLatency: 0,
            latencies: []
        };
        
        // Speech state tracking
        this.speechState = 'silence';
        this.speechStartTime = null;
        
        // Recording buffer with speech tracking
        this.recordingBuffer = [];
        this.speechChunks = []; // Track which chunks contain speech
        this.isSavingRecording = false;
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.connectWebSocket();
        this.loadConfig();
        this.checkHTTPSRequirement();
        this.updateSystemStatus('initializing', 'Initializing...');
    }

    setupEventListeners() {
        // Control buttons
        document.getElementById('startBtn').addEventListener('click', () => this.startRecording());
        document.getElementById('stopBtn').addEventListener('click', () => this.stopRecording());
        document.getElementById('saveRecordingBtn').addEventListener('click', () => this.saveRecording());
        document.getElementById('stopTTSBtn').addEventListener('click', () => this.stopTTS());

        // Language selection
        document.querySelectorAll('.lang-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const lang = e.currentTarget.dataset.lang;
                this.selectLanguage(lang);
            });
        });
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/audio`;
        
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            this.log('WebSocket connected', 'success');
            this.updateConnectionStatus(true);
            this.updateSystemStatus('ready', 'System Ready');
        };

        this.ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleMessage(message);
        };

        this.ws.onerror = (error) => {
            this.log('WebSocket error', 'error');
            this.updateConnectionStatus(false);
        };

        this.ws.onclose = () => {
            this.log('WebSocket disconnected', 'warning');
            this.updateConnectionStatus(false);
            setTimeout(() => this.connectWebSocket(), 3000);
        };
    }

    handleMessage(message) {
        switch (message.type) {
            case 'processed_audio':
                this.handleProcessedAudio(message);
                break;
            
            case 'speech_event':
                this.handleSpeechEvent(message);
                break;
            
            case 'transcription':
                this.handleTranscription(message);
                break;
            
            case 'transcription_interim':
                this.handleInterimTranscription(message);
                break;
            
            case 'tts_audio':
                this.handleTTSAudio(message);
                break;
            
            case 'tts_start':
                this.handleTTSStart(message);
                break;
            
            case 'tts_end':
                this.handleTTSEnd(message);
                break;
            
            case 'language_detected':
                this.handleLanguageDetected(message);
                break;
            
            case 'chatbot_response':
                this.handleChatbotResponse(message);
                break;
            
            case 'calibration_complete':
                this.log('Noise calibration complete', 'success');
                break;
            
            case 'pong':
                break;
            
            default:
                this.log(`Unknown message type: ${message.type}`, 'warning');
        }
    }

    handleProcessedAudio(message) {
        this.stats.chunksProcessed++;
        
        if (message.has_speech) {
            this.stats.speechDetected++;
        }
        
        // Update speech state IMMEDIATELY (no delay)
        if (message.speech_state) {
            const prevState = this.speechState;
            const newState = message.speech_state;
            
            // Only update if state actually changed
            if (newState !== prevState) {
                this.speechState = newState;
                
                // Update UI immediately based on speech state
                if (newState === 'speech_start' || newState === 'speech') {
                    // Speech detected - update immediately
                    this.updateSystemStatus('listening', 'Speech Detected');
                    this.updateVisualStatus('speech_start');
                } else if (newState === 'silence') {
                    // Speech ended - update immediately
                    if (prevState === 'speech_start' || prevState === 'speech' || prevState === 'speech_end') {
                        this.updateSystemStatus('listening', 'Listening...');
                        this.updateVisualStatus('listening');
                    }
                } else if (newState === 'speech_end') {
                    // Speech ended - update immediately
                    this.updateSystemStatus('listening', 'Listening...');
                    this.updateVisualStatus('listening');
                }
            }
        }
        
        // Track speech chunks for saving (only save chunks with speech)
        if (this.isRecording && this.recordingBuffer.length > 0) {
            const hasSpeech = message.has_speech || 
                             (message.speech_state === 'speech_start' || 
                              message.speech_state === 'speech');
            
            // Mark the last chunk in recording buffer as speech or not
            const lastIndex = this.recordingBuffer.length - 1;
            if (this.speechChunks.length <= lastIndex) {
                // Extend array if needed
                while (this.speechChunks.length <= lastIndex) {
                    this.speechChunks.push(false);
                }
            }
            this.speechChunks[lastIndex] = hasSpeech;
        }
        
        // Update audio level from server (more accurate)
        if (message.audio_level_db !== undefined) {
            this.updateAudioLevel(message.audio_level_db);
        }
        
        // Debug: Log VAD probability and audio level
        if (message.vad_probability !== undefined) {
            if (!message.has_speech && message.vad_probability > 0.1) {
                // VAD has some probability but below threshold - log occasionally
                if (this.stats.chunksProcessed % 50 === 0) {
                    this.log(`VAD prob: ${(message.vad_probability * 100).toFixed(1)}% (threshold: ${(this.config?.vad_threshold || 0.2) * 100}%), audio: ${message.audio_level_db}dB`, 'info');
                }
            }
        }
    }

    handleSpeechEvent(message) {
        const event = message.event;
        const data = message.data;
        
        this.log(`Speech event: ${event}`, 'info');
        
        // Update UI IMMEDIATELY when speech events occur (no delay)
        if (event === 'speech_start') {
            this.speechStartTime = Date.now();
            this.currentTranscription = '';
            this.interimTranscription = '';
            this.speechState = 'speech_start';
            
            // Update UI immediately
            this.updateTranscriptionStatus('listening', 'Speech Detected');
            this.updateVisualStatus('speech_start');
            this.updateSystemStatus('listening', 'Speech Detected');
        } else if (event === 'speech_end') {
            // Speech ended - update immediately
            this.speechState = 'silence';
            this.updateTranscriptionStatus('listening', 'Listening...');
            this.updateVisualStatus('listening');
            this.updateSystemStatus('listening', 'Listening...');
        }
    }

    handleTranscription(message) {
        const text = message.text;
        const language = message.language || this.detectedLanguage;
        const confidence = message.confidence || 0;
        
        if (text) {
            this.currentTranscription = text;
            this.stats.transcriptions++;
            this.updateTranscription(text, false);
            this.updateTranscriptionStatus('complete', 'Transcription complete');
            
            // Request chatbot response (if STT is ready)
            this.requestChatbotResponse(text);
        }
    }

    handleInterimTranscription(message) {
        const text = message.text;
        if (text) {
            this.interimTranscription = text;
            this.updateTranscription(text, true);
        }
    }

    handleTTSStart(message) {
        this.isSpeaking = true;
        this.updateVisualStatus('speaking');
        this.showTTSPanel(true);
        this.updateSystemStatus('speaking', 'Speaking...');
        
        if (message.text) {
            document.getElementById('ttsContent').textContent = message.text;
        }
    }

    handleTTSAudio(message) {
        // TTS audio data would be handled here
        // For now, just show visual feedback
        this.animateTTS();
    }

    handleTTSEnd(message) {
        this.isSpeaking = false;
        this.updateVisualStatus('ready');
        this.showTTSPanel(false);
        this.updateSystemStatus('ready', 'Ready');
    }

    handleLanguageDetected(message) {
        const lang = message.language;
        const confidence = message.confidence || 0;
        
        this.detectedLanguage = lang;
        document.getElementById('detectedLangText').textContent = 
            this.getLanguageName(lang) + ` (${Math.round(confidence * 100)}%)`;
        
        // Update language button if auto-detect is on
        if (this.currentLanguage === 'auto') {
            this.highlightDetectedLanguage(lang);
        }
    }

    handleChatbotResponse(message) {
        const response = message.text;
        const language = message.language || this.currentLanguage;
        
        if (response) {
            this.requestTTS(response, language);
        }
    }

    async startRecording() {
        try {
            const hasModernAPI = navigator.mediaDevices && typeof navigator.mediaDevices.getUserMedia === 'function';
            
            if (!hasModernAPI) {
                alert('Your browser does not support microphone access. Please use Chrome, Firefox, or Edge.');
                return;
            }

            this.log('Requesting microphone access...', 'info');
            
            const audioConstraints = {
                echoCancellation: true,
                noiseSuppression: false,
                autoGainControl: true,
                sampleRate: { ideal: 16000 },
                channelCount: { ideal: 1 }
            };
            
            try {
                this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });
            } catch (e) {
                this.log('Trying with minimal constraints...', 'warning');
                this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            }

            const audioTrack = this.mediaStream.getAudioTracks()[0];
            const settings = audioTrack.getSettings();
            const actualSampleRate = settings.sampleRate || 48000;

            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: actualSampleRate
            });

            const source = this.audioContext.createMediaStreamSource(this.mediaStream);
            const bufferSize = 512;
            const processor = this.audioContext.createScriptProcessor(bufferSize, 1, 1);

            processor.onaudioprocess = (e) => {
                if (!this.isRecording) return;

                const inputData = e.inputBuffer.getChannelData(0);
                
                // Store in recording buffer for saving (keep original sample rate)
                // Note: speechChunks will be updated in handleProcessedAudio based on VAD results
                this.recordingBuffer.push(new Float32Array(inputData));
                
                // Keep buffer size reasonable (last 60 seconds)
                const maxBufferSize = this.audioContext.sampleRate * 60;
                let totalSamples = 0;
                for (let i = this.recordingBuffer.length - 1; i >= 0; i--) {
                    totalSamples += this.recordingBuffer[i].length;
                    if (totalSamples > maxBufferSize) {
                        const removed = this.recordingBuffer.length - (i + 1);
                        this.recordingBuffer = this.recordingBuffer.slice(i + 1);
                        this.speechChunks = this.speechChunks.slice(removed);
                        break;
                    }
                }
                
                let audioData = inputData;
                if (this.audioContext.sampleRate !== 16000) {
                    audioData = this.resampleAudio(inputData, this.audioContext.sampleRate, 16000);
                }
                
                this.processAudioChunk(audioData);
            };

            source.connect(processor);
            processor.connect(this.audioContext.destination);

            this.isRecording = true;
            this.recordingBuffer = []; // Reset recording buffer
            this.speechChunks = []; // Reset speech tracking
            this.updateControls(true);
            this.updateVisualStatus('listening');
            this.updateSystemStatus('listening', 'Listening...');
            this.log('Recording started', 'success');

        } catch (error) {
            this.handleMicrophoneError(error);
        }
    }

    stopRecording() {
        this.isRecording = false;
        
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }

        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }

        this.updateControls(false);
        this.updateVisualStatus('ready');
        this.updateSystemStatus('ready', 'Ready');
        this.log('Recording stopped', 'info');
    }

    processAudioChunk(audioData) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;

        const rms = Math.sqrt(audioData.reduce((sum, val) => sum + val * val, 0) / audioData.length);
        // Calculate dB: 20 * log10(rms), with -∞ for silence
        const audioLevelDb = rms > 0 ? 20 * Math.log10(rms) : -Infinity;
        this.updateAudioLevel(audioLevelDb);

        const int16Array = new Int16Array(audioData.length);
        for (let i = 0; i < audioData.length; i++) {
            const s = Math.max(-1, Math.min(1, audioData[i]));
            int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }

        const base64 = this.arrayBufferToBase64(int16Array.buffer);
        const startTime = performance.now();

        this.ws.send(JSON.stringify({
            type: 'audio_chunk',
            audio: base64,
            language: this.currentLanguage === 'auto' ? null : this.currentLanguage
        }));

        const latency = performance.now() - startTime;
        this.stats.latencies.push(latency);
        if (this.stats.latencies.length > 100) {
            this.stats.latencies.shift();
        }
        this.stats.avgLatency = Math.round(
            this.stats.latencies.reduce((a, b) => a + b, 0) / this.stats.latencies.length
        );
    }

    selectLanguage(lang) {
        this.currentLanguage = lang;
        
        document.querySelectorAll('.lang-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.getElementById(`lang${lang.charAt(0).toUpperCase() + lang.slice(1)}`).classList.add('active');
        
        this.log(`Language set to: ${this.getLanguageName(lang)}`, 'info');
    }

    highlightDetectedLanguage(lang) {
        // Visual feedback for detected language
        const langMap = { 'en': 'En', 'ja': 'Ja', 'zh': 'Zh' };
        const btnId = `lang${langMap[lang] || 'Auto'}`;
        const btn = document.getElementById(btnId);
        if (btn) {
            btn.classList.add('detected');
            setTimeout(() => btn.classList.remove('detected'), 2000);
        }
    }

    getLanguageName(code) {
        const names = {
            'auto': 'Auto-detect',
            'en': 'English',
            'ja': 'Japanese',
            'zh': 'Chinese'
        };
        return names[code] || code;
    }


    updateTranscription(text, isInterim) {
        if (isInterim) {
            document.getElementById('transcriptionInterim').textContent = text;
            document.getElementById('transcriptionContent').classList.add('has-interim');
        } else {
            document.getElementById('transcriptionContent').textContent = text;
            document.getElementById('transcriptionInterim').textContent = '';
            document.getElementById('transcriptionContent').classList.remove('has-interim');
        }
    }

    updateTranscriptionStatus(status, text) {
        const statusEl = document.getElementById('transcriptionStatus');
        const dot = statusEl.querySelector('.status-dot');
        const textSpan = statusEl.querySelector('span:last-child');
        
        statusEl.className = `transcription-status ${status}`;
        textSpan.textContent = text;
    }

    updateVisualStatus(state) {
        const circle = document.getElementById('statusCircle');
        const icon = document.getElementById('statusIcon');
        const text = document.getElementById('statusText');
        const waves = document.getElementById('statusWaves');
        
        circle.className = `status-circle ${state}`;
        
        const states = {
            'ready': { icon: '🎤', text: 'Ready', color: '#4CAF50' },
            'listening': { icon: '👂', text: 'Listening...', color: '#2196F3' },
            'speech_start': { icon: '🗣️', text: 'Speech Detected', color: '#FF9800' },
            'processing': { icon: '⚙️', text: 'Processing...', color: '#9C27B0' },
            'speaking': { icon: '🔊', text: 'Speaking...', color: '#F44336' }
        };
        
        const stateInfo = states[state] || states['ready'];
        icon.textContent = stateInfo.icon;
        text.textContent = stateInfo.text;
        circle.style.borderColor = stateInfo.color;
        
        // Animate waves for active states
        if (['listening', 'speech_start', 'speaking'].includes(state)) {
            waves.style.display = 'block';
            waves.className = `status-waves ${state}`;
        } else {
            waves.style.display = 'none';
        }
    }

    updateSystemStatus(component, status) {
        // Update individual component statuses
        const statusMap = {
            'pipeline': 'pipelineStatus',
            'stt': 'sttStatus',
            'tts': 'ttsStatus',
            'vad': 'vadStatus'
        };
        
        if (statusMap[component]) {
            document.getElementById(statusMap[component]).textContent = status;
        }
        
        // Update main system status
        if (component === 'ready' || component === 'listening' || component === 'speaking') {
            const systemStatus = document.getElementById('systemStatus');
            systemStatus.querySelector('span:last-child').textContent = status;
            systemStatus.className = `status-badge ${component}`;
        }
    }

    showTTSPanel(show) {
        document.getElementById('ttsPanel').style.display = show ? 'block' : 'none';
    }

    animateTTS() {
        const visualizer = document.getElementById('ttsVisualizer');
        const wave = visualizer.querySelector('.tts-wave');
        if (wave) {
            wave.style.animation = 'none';
            setTimeout(() => {
                wave.style.animation = 'ttsWave 1s ease-in-out infinite';
            }, 10);
        }
    }

    stopTTS() {
        if (this.ttsAudio) {
            this.ttsAudio.pause();
            this.ttsAudio = null;
        }
        this.handleTTSEnd({});
    }

    requestChatbotResponse(text) {
        // Send to backend for chatbot processing
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'chatbot_request',
                text: text,
                language: this.currentLanguage === 'auto' ? this.detectedLanguage : this.currentLanguage
            }));
        }
    }

    requestTTS(text, language) {
        // Request TTS from backend
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'tts_request',
                text: text,
                language: language || this.currentLanguage
            }));
        }
    }


    async saveRecording() {
        if (!this.isRecording) {
            alert('Please start recording first');
            return;
        }
        
        if (this.isSavingRecording) {
            this.log('Recording save already in progress...', 'warning');
            return;
        }
        
        this.isSavingRecording = true;
        this.log('Saving recording (speech only)...', 'info');
        
        try {
            // Filter to only speech chunks
            const speechOnlyChunks = [];
            for (let i = 0; i < this.recordingBuffer.length; i++) {
                if (this.speechChunks[i] === true) {
                    speechOnlyChunks.push(this.recordingBuffer[i]);
                }
            }
            
            if (speechOnlyChunks.length === 0) {
                alert('No speech detected in recording. Please speak into the microphone.');
                this.isSavingRecording = false;
                return;
            }
            
            // Add small padding around speech segments (100ms before/after)
            const paddedChunks = [];
            const paddingSamples = Math.floor((this.audioContext ? this.audioContext.sampleRate : 48000) * 0.1); // 100ms
            
            for (let i = 0; i < this.recordingBuffer.length; i++) {
                if (this.speechChunks[i] === true) {
                    // Add padding before (if previous chunk exists)
                    if (i > 0 && paddedChunks.length === 0) {
                        const prevChunk = this.recordingBuffer[i - 1];
                        const padding = prevChunk.slice(-Math.min(paddingSamples, prevChunk.length));
                        paddedChunks.push(padding);
                    }
                    
                    // Add speech chunk
                    paddedChunks.push(this.recordingBuffer[i]);
                    
                    // Add padding after (if next chunk exists and is not speech)
                    if (i < this.recordingBuffer.length - 1 && this.speechChunks[i + 1] !== true) {
                        const nextChunk = this.recordingBuffer[i + 1];
                        const padding = nextChunk.slice(0, Math.min(paddingSamples, nextChunk.length));
                        paddedChunks.push(padding);
                    }
                }
            }
            
            // Combine all speech chunks
            const totalSamples = paddedChunks.reduce((sum, chunk) => sum + chunk.length, 0);
            const combinedAudio = new Float32Array(totalSamples);
            let offset = 0;
            for (const chunk of paddedChunks) {
                combinedAudio.set(chunk, offset);
                offset += chunk.length;
            }
            
            // Convert to WAV format
            const sampleRate = this.audioContext ? this.audioContext.sampleRate : 48000;
            const wavBlob = this.audioToWav(combinedAudio, sampleRate);
            
            // Create download link
            const url = URL.createObjectURL(wavBlob);
            const a = document.createElement('a');
            a.href = url;
            const speechDuration = (combinedAudio.length / sampleRate).toFixed(1);
            a.download = `speech_${speechDuration}s_${new Date().toISOString().replace(/[:.]/g, '-')}.wav`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            this.log(`Speech recording saved: ${a.download} (${speechDuration}s)`, 'success');
            
        } catch (error) {
            this.log('Error saving recording: ' + error.message, 'error');
            alert('Failed to save recording: ' + error.message);
        } finally {
            this.isSavingRecording = false;
        }
    }
    
    audioToWav(audioData, sampleRate) {
        // Convert Float32Array to Int16Array
        const int16Array = new Int16Array(audioData.length);
        for (let i = 0; i < audioData.length; i++) {
            const s = Math.max(-1, Math.min(1, audioData[i]));
            int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        
        // Create WAV file header
        const buffer = new ArrayBuffer(44 + int16Array.length * 2);
        const view = new DataView(buffer);
        
        // WAV header
        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };
        
        writeString(0, 'RIFF');
        view.setUint32(4, 36 + int16Array.length * 2, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true); // fmt chunk size
        view.setUint16(20, 1, true); // audio format (1 = PCM)
        view.setUint16(22, 1, true); // number of channels
        view.setUint32(24, sampleRate, true); // sample rate
        view.setUint32(28, sampleRate * 2, true); // byte rate
        view.setUint16(32, 2, true); // block align
        view.setUint16(34, 16, true); // bits per sample
        writeString(36, 'data');
        view.setUint32(40, int16Array.length * 2, true);
        
        // Write audio data
        let offset = 44;
        for (let i = 0; i < int16Array.length; i++) {
            view.setInt16(offset, int16Array[i], true);
            offset += 2;
        }
        
        return new Blob([buffer], { type: 'audio/wav' });
    }

    updateConnectionStatus(connected) {
        const status = document.getElementById('connectionStatus');
        const dot = status.querySelector('.connection-dot');
        const text = status.querySelector('span:last-child');
        
        if (connected) {
            status.classList.add('connected');
            text.textContent = 'Connected';
        } else {
            status.classList.remove('connected');
            text.textContent = 'Disconnected';
        }
    }

    updateAudioLevel(levelDb) {
        const meterBar = document.getElementById('audioMeterBar');
        const levelText = document.getElementById('audioLevelText');
        
        // Convert dB to percentage for visual meter (dB range: -60 to 0)
        // -60dB = 0%, 0dB = 100%
        const minDb = -60;
        const maxDb = 0;
        const normalizedLevel = Math.max(0, Math.min(100, ((levelDb - minDb) / (maxDb - minDb)) * 100));
        
        meterBar.style.width = normalizedLevel + '%';
        
        // Color coding: green (-20 to 0), yellow (-40 to -20), red (< -40)
        let colorClass = 'low';
        if (levelDb >= -20) {
            colorClass = 'high';
        } else if (levelDb >= -40) {
            colorClass = 'medium';
        }
        meterBar.className = `audio-meter-bar ${colorClass}`;
        
        // Display dB value
        if (levelDb === -Infinity || isNaN(levelDb)) {
            levelText.textContent = '-∞ dB';
        } else {
            levelText.textContent = levelDb.toFixed(1) + ' dB';
        }
    }

    updateControls(recording) {
        document.getElementById('startBtn').disabled = recording;
        document.getElementById('stopBtn').disabled = !recording;
        document.getElementById('saveRecordingBtn').disabled = !recording;
    }


    async loadConfig() {
        try {
            const response = await fetch('/api/config');
            const config = await response.json();
            this.config = config;
        } catch (error) {
            this.log('Failed to load config: ' + error.message, 'warning');
        }
    }

    handleMicrophoneError(error) {
        let errorMessage = 'Failed to access microphone. ';
        
        if (error.name === 'NotAllowedError') {
            errorMessage += 'Please grant microphone permissions.';
        } else if (error.name === 'NotFoundError') {
            errorMessage += 'No microphone found.';
        } else {
            errorMessage += error.message;
        }
        
        this.log(errorMessage, 'error');
        alert(errorMessage);
    }

    checkHTTPSRequirement() {
        const isHTTP = location.protocol === 'http:';
        const isLocalhost = ['localhost', '127.0.0.1', '0.0.0.0', ''].includes(location.hostname);
        
        if (isHTTP && !isLocalhost) {
            this.log('⚠️ HTTPS required for microphone access on non-localhost', 'warning');
        }
    }

    resampleAudio(audioData, fromRate, toRate) {
        if (fromRate === toRate) return audioData;
        
        const ratio = fromRate / toRate;
        const newLength = Math.round(audioData.length / ratio);
        const resampled = new Float32Array(newLength);
        
        for (let i = 0; i < newLength; i++) {
            const srcIndex = i / ratio;
            const srcIndexFloor = Math.floor(srcIndex);
            const srcIndexCeil = Math.min(srcIndexFloor + 1, audioData.length - 1);
            const t = srcIndex - srcIndexFloor;
            
            resampled[i] = audioData[srcIndexFloor] * (1 - t) + audioData[srcIndexCeil] * t;
        }
        
        return resampled;
    }

    arrayBufferToBase64(buffer) {
        const bytes = new Uint8Array(buffer);
        let binary = '';
        for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }


    log(message, type = 'info') {
        const logContent = document.getElementById('logContent');
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry ${type}`;
        logEntry.textContent = `[${timestamp}] ${message}`;
        
        logContent.appendChild(logEntry);
        logContent.scrollTop = logContent.scrollHeight;
        
        while (logContent.children.length > 50) {
            logContent.removeChild(logContent.firstChild);
        }
    }
}

// Utility functions
function togglePanel(header) {
    const content = header.nextElementSibling;
    const icon = header.querySelector('.toggle-icon');
    if (content.style.display === 'none') {
        content.style.display = 'block';
        icon.textContent = '▼';
    } else {
        content.style.display = 'none';
        icon.textContent = '▶';
    }
}

function closeModal(modalId) {
    document.getElementById(modalId).style.display = 'none';
}

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    window.voiceAI = new FactoryVoiceAI();
});

window.addEventListener('beforeunload', () => {
    if (window.voiceAI) {
        window.voiceAI.stopRecording();
    }
});
