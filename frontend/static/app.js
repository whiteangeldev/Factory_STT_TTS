// Factory STT/TTS Frontend Application
class STTApp {
    constructor() {
        this.socket = null;
        this.audioContext = null;
        this.mediaStream = null;
        this.processor = null;
        this.isRecording = false;
        this.isStopping = false;
        this.audioChunks = [];
        this.transcriptionBuffer = [];
        this.inputMode = 'microphone'; // 'microphone' or 'system'
        this.transcriptionHistory = []; // Store transcription history
        this.currentInterimText = ''; // Current interim transcription
        this.lastTranscriptionTime = null; // Track transcription timestamps
        this.transcriptionCount = 0; // Count of transcriptions received
        
        // VAD smoothing for bottom indicator only
        this.vadHistory = [];
        this.currentVadState = false;
        
        // UI-only display state (separate from VAD detection to prevent flicker)
        this.currentVadDisplayState = false;
        this._vadUiHoldUntil = 0;
        this.vadUiHoldMs = 800; // Hold "Speech Detected" for 800ms during brief dips
        
        // TTS state
        this.ttsAudio = null;
        this.isTTSPlaying = false;
        this.ttsQueue = [];
        this.ttsCurrentIndex = 0; // Number of segments already started
        this.ttsObjectUrls = [];
        this.ttsActiveRequestId = null;
        this.ttsExpectedSegments = 0;
        this.ttsStreamComplete = false;
        this.ttsAudioData = []; // Store audio data (base64) for saving
        
        this.init();
    }
    
    init() {
        this.connectWebSocket();
        this.setupEventListeners();
        this.updateSystemStatus('initializing', 'Initializing...');
        this.initializeVADIndicator();
        
        // Initialize input mode UI - set microphone as default active
        const micBtn = document.getElementById('micModeBtn');
        if (micBtn && !micBtn.classList.contains('active')) {
            micBtn.classList.add('active');
        }
        this.inputMode = 'microphone';
    }
    
    initializeVADIndicator() {
        // Initialize VAD indicator to "No Speech" state
        const vadDot = document.getElementById('vadDot');
        const vadStatus = document.querySelector('.vad-text');
        if (vadDot) {
            vadDot.classList.remove('active');
        }
        if (vadStatus) {
            vadStatus.textContent = 'No Speech';
        }
    }
    
    connectWebSocket() {
        const socketUrl = window.location.origin;
        console.log('🔄 Connecting to Socket.IO:', socketUrl);
        
        this.socket = io(socketUrl, {
            transports: ['websocket', 'polling'],
            reconnection: true,
            reconnectionDelay: 1000,
            reconnectionDelayMax: 5000,
            reconnectionAttempts: Infinity
        });
        
        this.socket.on('connect', () => {
            console.log('✅ Socket.IO connected, ID:', this.socket.id);
            this.log('Socket.IO connected', 'success');
            this.updateConnectionStatus(true);
            this.updateSystemStatus('ready', 'System Ready');
        });
        
        this.socket.on('connected', (data) => {
            console.log('✅ Server confirmed connection:', data);
            this.log(data.message || 'Socket.IO ready - VAD and noise reduction active', 'success');
        });
        
        // TTS event handlers
        this.socket.on('tts_audio', (data) => {
            this.handleTTSAudio(data);
        });
        
        this.socket.on('tts_error', (data) => {
            this.handleTTSError(data);
        });
        
        this.socket.on('recording_status', (data) => {
            console.log('📊 Recording status update:', data);
            const recordBtn = document.getElementById('recordBtn');
            
            if (data.is_recording) {
                this.updateSystemStatus('listening', data.status || 'Recording...');
                if (!this.isRecording) {
                    // Server confirmed recording started
                    this.isRecording = true;
                    recordBtn.classList.add('recording');
                    document.getElementById('saveBtn').disabled = false;
                }
            } else {
                this.updateSystemStatus('ready', data.status || 'Ready');
                if (this.isRecording) {
                    // Server confirmed recording stopped
                    this.isRecording = false;
                    recordBtn.classList.remove('recording');
                }
            }
        });
        
        this.socket.on('processed_audio', (data) => {
            // Log first few events to verify reception
            if (!this._processedAudioCount) {
                this._processedAudioCount = 0;
            }
            this._processedAudioCount++;
            
            if (this._processedAudioCount <= 5) {
                console.log(`📥 [${this._processedAudioCount}] Received processed_audio:`, {
                    has_audio: !!data.audio,
                    audio_length: data.audio ? data.audio.length : 0,
                    has_speech: data.has_speech,
                    audio_level_db: data.audio_level_db,
                    mode: this.inputMode,
                    isRecording: this.isRecording,
                    current_chunks: this.audioChunks.length
                });
            } else if (this._processedAudioCount % 100 === 0) {
                // Log every 100th chunk to show progress
                console.log(`📥 Received ${this._processedAudioCount} audio chunks (${this.audioChunks.length} stored)`);
            }
            
            // Debug: log VAD data occasionally
            if (data.has_speech) {
                console.log('🎤 VAD: Speech detected', data);
            }
            this.handleProcessedAudio(data);
        });
        
        this.socket.on('speech_event', (data) => {
            console.log('🔊 Received speech_event:', data);
            // Handle both nested and flat structures
            const eventType = data.event || data.type;
            const eventData = data.data || {};
            this.handleSpeechEvent({ event: eventType, data: eventData });
        });
        
        this.socket.on('transcription', (data) => {
            console.log('📝 Received transcription:', data.text);
            this.handleTranscription(data);
        });
        
        this.socket.on('transcription_interim', (data) => {
            console.log('📝 Received interim transcription:', data.text);
            this.handleInterimTranscription(data);
        });
        
        this.socket.on('transcription_processing', (data) => {
            console.log('⏳ Transcription processing:', data.status);
            if (data.status === 'processing') {
                this.log(`Processing transcription (${data.audio_duration?.toFixed(2)}s)...`, 'info');
            }
        });
        
        this.socket.on('error', (data) => {
            console.error('❌ Server error:', data);
            this.log(`Error: ${data.message || 'Unknown error'}`, 'error');
        });
        
        this.socket.on('disconnect', (reason) => {
            console.log('🔌 Socket.IO disconnected:', reason);
            this.log(`Disconnected: ${reason}`, 'warning');
            this.updateConnectionStatus(false);
        });
        
        this.socket.on('connect_error', (error) => {
            console.error('❌ Socket.IO connection error:', error);
            this.log(`Connection error: ${error.message}`, 'error');
        });
    }
    
    setupEventListeners() {
        // Recording controls - press and hold to record
        const recordBtn = document.getElementById('recordBtn');
        const recordBtnText = document.getElementById('recordBtnText');
        
        // Mouse events
        recordBtn.addEventListener('mousedown', (e) => {
            e.preventDefault();
            if (!this.isRecording) {
                this.startRecording();
            }
        });
        
        recordBtn.addEventListener('mouseup', (e) => {
            e.preventDefault();
            if (this.isRecording) {
                this.stopRecording();
            }
        });
        
        recordBtn.addEventListener('mouseleave', (e) => {
            // Stop recording if mouse leaves button while pressed
            if (this.isRecording) {
                this.stopRecording();
            }
        });
        
        // Touch events for mobile
        recordBtn.addEventListener('touchstart', (e) => {
            e.preventDefault();
            if (!this.isRecording) {
                this.startRecording();
            }
        }, { passive: false });
        
        recordBtn.addEventListener('touchend', (e) => {
            e.preventDefault();
            if (this.isRecording) {
                this.stopRecording();
            }
        }, { passive: false });
        
        recordBtn.addEventListener('touchcancel', (e) => {
            e.preventDefault();
            if (this.isRecording) {
                this.stopRecording();
            }
        }, { passive: false });
        
        // Save button
        document.getElementById('saveBtn').addEventListener('click', () => this.saveRecording());
        
        // Input mode toggle buttons
        document.getElementById('micModeBtn').addEventListener('click', () => this.setInputMode('microphone'));
        document.getElementById('systemModeBtn').addEventListener('click', () => this.setInputMode('system'));
        
        // TTS event listeners
        document.getElementById('ttsPlayBtn').addEventListener('click', () => this.playTTS());
        document.getElementById('ttsStopBtn').addEventListener('click', () => this.stopTTS());
        document.getElementById('ttsSaveBtn').addEventListener('click', () => this.saveTTS());
        document.getElementById('ttsSpeed').addEventListener('input', (e) => {
            document.getElementById('ttsSpeedValue').textContent = `${parseFloat(e.target.value).toFixed(1)}x`;
        });
        
        // Logs toggle
        const toggleLogsBtn = document.getElementById('toggleLogsBtn');
        if (toggleLogsBtn) {
            toggleLogsBtn.addEventListener('click', () => this.toggleLogs());
        }
    }
    
    setInputMode(mode) {
        if (this.isRecording) {
            this.log('Please stop recording before changing input mode', 'warning');
            return;
        }
        
        const oldMode = this.inputMode;
        this.inputMode = mode;
        
        // Update button states
        const micBtn = document.getElementById('micModeBtn');
        const systemBtn = document.getElementById('systemModeBtn');
        
        if (micBtn && systemBtn) {
            if (mode === 'microphone') {
                micBtn.classList.add('active');
                systemBtn.classList.remove('active');
            } else {
                micBtn.classList.remove('active');
                systemBtn.classList.add('active');
            }
        }
        
        this.log(`Input mode changed to: ${mode === 'microphone' ? 'Microphone' : 'System Audio'}`, 'info');
    }
    
    toggleLogs() {
        const logContainer = document.getElementById('logArea');
        const toggleBtn = document.getElementById('toggleLogsBtn');
        if (logContainer && toggleBtn) {
            const isCollapsed = logContainer.classList.contains('collapsed');
            if (isCollapsed) {
                logContainer.classList.remove('collapsed');
                toggleBtn.querySelector('svg').style.transform = 'rotate(0deg)';
            } else {
                logContainer.classList.add('collapsed');
                toggleBtn.querySelector('svg').style.transform = 'rotate(180deg)';
            }
        }
    }
    
    async startRecording() {
        if (this.isRecording) return;
        
        if (!this.socket || !this.socket.connected) {
            this.log('Socket.IO not connected. Attempting reconnection...', 'warning');
            this.socket.connect();
            await new Promise(resolve => setTimeout(resolve, 1000));
            if (!this.socket.connected) {
                alert('Failed to connect to server. Please refresh the page.');
                return;
            }
        }
        
        // Check if mediaDevices API is available
        if (!navigator.mediaDevices) {
            const errorMsg = 'Media access is not available in this browser. Please use a modern browser (Chrome, Firefox, Edge).';
            console.error(errorMsg);
            this.log(errorMsg, 'error');
            alert(errorMsg);
            return;
        }
        
        try {
            // Try to create audio context first
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            if (!AudioContextClass) {
                throw new Error('Web Audio API is not supported in this browser');
            }
            
            this.audioContext = new AudioContextClass();
            
            // Log actual sample rate (browser may override)
            const actualSampleRate = this.audioContext.sampleRate;
            console.log(`AudioContext created with sample rate: ${actualSampleRate} Hz`);
            this.log(`Audio context: ${actualSampleRate} Hz`, 'info');
            
            // Get input mode from active button (already set by setInputMode or default)
            // Ensure mode is set correctly
            const micBtn = document.getElementById('micModeBtn');
            const systemBtn = document.getElementById('systemModeBtn');
            if (micBtn && micBtn.classList.contains('active')) {
                this.inputMode = 'microphone';
            } else if (systemBtn && systemBtn.classList.contains('active')) {
                this.inputMode = 'system';
            } else {
                // Default to microphone if no button is active
                this.inputMode = 'microphone';
                if (micBtn) micBtn.classList.add('active');
            }
            
            // Both system audio and microphone: Use server-side capture
            const inputName = this.inputMode === 'system' ? 'system audio' : 'microphone';
            console.log(`Using server-side ${inputName} capture`);
            this.log(`Starting server-side ${inputName} capture...`, 'info');
            
            // No browser audio capture needed - server handles it
            this.mediaStream = null;
            this.isRecording = true;
            this.isStopping = false;
            this.audioChunks = [];
            this.transcriptionBuffer = [];
            // Don't reset transcriptionHistory - keep it for stats
            // Don't reset transcriptionCount - count from DOM instead
            this.currentInterimText = '';
            this.lastTranscriptionTime = null;
            this.recordingStartTime = Date.now() / 1000;
            
            // Reset counters for logging
            this._processedAudioCount = 0;
            this._silenceChunkCount = 0;
            
            // CRITICAL: Reset VAD state and history when starting new recording
            // This prevents false positives from previous recording's history
            this.vadHistory = [];
            this.currentVadState = false;
            this.currentVadDisplayState = false;
            this._vadUiHoldUntil = 0;
            this._speechEnded = false;
            this._speechEndedCount = 0;
            
            console.log(`🎙️ ${inputName.charAt(0).toUpperCase() + inputName.slice(1)} mode: Ready to receive audio from server`);
            console.log('   Waiting for processed_audio events...');
            
            // Don't clear transcription area - keep previous transcriptions
            // Notify backend to start server-side capture
            if (this.socket && this.socket.connected) {
                this.socket.emit('start_recording', {
                    input_mode: this.inputMode,
                    server_capture: true
                });
            }
            
            // Update UI
            const recordBtn = document.getElementById('recordBtn');
            recordBtn.classList.add('recording');
            document.getElementById('saveBtn').disabled = false;
            
            const statusMessage = this.inputMode === 'system' ? 'Listening (system audio)...' : 'Listening (microphone)...';
            this.updateSystemStatus('listening', statusMessage);
            this.log(`${inputName.charAt(0).toUpperCase() + inputName.slice(1)} started - capturing from server`, 'success');
            
        } catch (error) {
            console.error('Error starting recording:', error);
            console.error('Error name:', error.name);
            console.error('Error message:', error.message);
            
            let errorMsg = 'Failed to access microphone. ';
            
            if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
                errorMsg += 'Please allow microphone access in your browser settings and try again.';
            } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
                errorMsg += 'No microphone found. Please connect a microphone and try again.';
            } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
                errorMsg += 'Microphone is being used by another application. Please close other apps and try again.';
            } else if (error.name === 'NotSupportedError') {
                errorMsg += 'Microphone access is not supported in this browser.';
            } else if (error.name === 'AbortError') {
                errorMsg += 'Microphone request was cancelled.';
            } else if (error.name === 'OverconstrainedError') {
                errorMsg += 'Microphone does not support required settings. Trying with basic settings...';
            } else {
                errorMsg += `Error: ${error.message || error.name}`;
            }
            
            this.log(errorMsg, 'error');
            alert(errorMsg);
            
            // Reset UI
        const recordBtn = document.getElementById('recordBtn');
        recordBtn.classList.remove('recording');
        }
    }
    
    stopRecording() {
        if (!this.isRecording) return;
        
        // Set stopping flag FIRST to prevent any new chunks from being sent
        this.isStopping = true;
        
        // Set recording flag to false to stop any in-flight processing
        this.isRecording = false;
        
        // Notify backend that recording has stopped IMMEDIATELY
        if (this.socket && this.socket.connected) {
            this.socket.emit('stop_recording');
        }
        
        // Both modes use server-side capture, so no browser audio cleanup needed
        // Server handles stopping the audio capture
        
        const recordBtn = document.getElementById('recordBtn');
        recordBtn.classList.remove('recording');
        
        this.updateSystemStatus('ready', 'Ready');
        
        // Log summary of received audio
        const totalSamples = this.audioChunks.reduce((sum, chunk) => sum + chunk.length, 0);
        const duration = (totalSamples / 16000).toFixed(2);
        const totalReceived = this._processedAudioCount || 0;
        
        console.log('🛑 Recording stopped - Summary:');
        console.log(`   Total processed_audio events received: ${totalReceived}`);
        console.log(`   Audio chunks stored: ${this.audioChunks.length}`);
        console.log(`   Total samples: ${totalSamples} (${duration}s at 16kHz)`);
        if (this.inputMode === 'system') {
            console.log(`   Silence chunks (no audio data): ${this._silenceChunkCount || 0}`);
        }
        
        this.log(`Recording stopped: ${this.audioChunks.length} chunks buffered (${duration}s)`, 'info');
        
        // Reset stopping flag after a short delay to allow any pending callbacks to finish
        setTimeout(() => {
            this.isStopping = false;
        }, 100);
    }
    
    processAudioChunk(audioData, sourceSampleRate = 16000) {
        // Both modes now use server-side capture, so this function is no longer used
        // Keeping for backward compatibility but it won't be called
        if (!this.isRecording || this.isStopping) {
            return;
        }
        
        if (!this.socket || !this.socket.connected) {
            console.warn('⚠️ Socket.IO disconnected - audio captured locally but not processed');
            // Still store audio locally even if socket is disconnected
        }
        
        // Resample to 16kHz if needed (do this BEFORE checking audio level)
        let processedData = audioData;
        if (Math.abs(sourceSampleRate - 16000) > 100) { // Only resample if significantly different
            processedData = this.resample(audioData, sourceSampleRate, 16000);
            if (!this._resampleLogged) {
                console.log(`Resampled from ${sourceSampleRate}Hz to 16kHz: ${audioData.length} -> ${processedData.length} samples`);
                this._resampleLogged = true;
            }
        }
        
        // Ensure we have valid data
        if (processedData.length === 0) {
            console.warn('Empty audio data after processing');
            return;
        }
        
        // Store for saving - ALWAYS store the resampled audio at 16kHz (processedData)
        // This ensures the saved audio matches what was sent to the server and plays at correct speed
        this.audioChunks.push(processedData.slice());
        
        // Check audio level for sending (but still store for saving)
        const maxLevel = Math.max(...Array.from(processedData).map(Math.abs));
        if (maxLevel < 0.0001) {
            // Skip sending silent chunks to reduce bandwidth, but we already stored it above
            return;
        }
        
        // Convert Float32Array to Int16Array for sending to server
        // Proper conversion: map [-1, 1] to [-32768, 32767]
        const int16Array = new Int16Array(processedData.length);
        for (let i = 0; i < processedData.length; i++) {
            // Clamp to [-1, 1] range
            const s = Math.max(-1, Math.min(1, processedData[i]));
            // Convert to int16: multiply by 32768 and clamp
            int16Array[i] = Math.max(-32768, Math.min(32767, Math.round(s * 32768)));
        }
        
        // Verify conversion (debug first few chunks)
        if (!this._conversionDebugCount) {
            this._conversionDebugCount = 0;
        }
        this._conversionDebugCount++;
        if (this._conversionDebugCount <= 3) {
            const maxFloat = Math.max(...Array.from(processedData).map(Math.abs));
            const maxInt16 = Math.max(...Array.from(int16Array).map(Math.abs));
            console.log(`[Conversion Debug ${this._conversionDebugCount}] Float max: ${maxFloat.toFixed(6)}, Int16 max: ${maxInt16}, samples: ${processedData.length}`);
        }
        
        // Convert to base64 using ArrayBuffer (proper binary encoding)
        // Use chunked approach to avoid "Maximum call stack size exceeded" for large arrays
        const uint8Array = new Uint8Array(int16Array.buffer);
        let binaryString = '';
        const chunkSize = 8192; // Process in chunks to avoid stack overflow
        for (let i = 0; i < uint8Array.length; i += chunkSize) {
            const chunk = uint8Array.slice(i, Math.min(i + chunkSize, uint8Array.length));
            binaryString += String.fromCharCode.apply(null, Array.from(chunk));
        }
        const base64 = btoa(binaryString);
        
        // Send to server (only in microphone mode, and if connected AND still recording AND not stopping)
        // Triple-check to prevent race conditions
        // System audio mode: server handles audio capture, browser doesn't send chunks
        if (this.inputMode === 'microphone' && this.isRecording && !this.isStopping && this.socket && this.socket.connected) {
            try {
                this.socket.emit('audio_chunk', { audio: base64 });
            } catch (error) {
                console.error('Error sending audio chunk:', error);
            }
        }
    }
    
    resample(input, inputSampleRate, outputSampleRate) {
        if (Math.abs(inputSampleRate - outputSampleRate) < 100) {
            return input; // Close enough, no resampling needed
        }
        
        const ratio = inputSampleRate / outputSampleRate;
        const outputLength = Math.round(input.length / ratio);
        
        // Ensure output length is reasonable
        if (outputLength <= 0 || outputLength > input.length * 2) {
            console.error(`Invalid resample output length: ${outputLength} from ${input.length} samples`);
            return input; // Return original if resampling would be invalid
        }
        
        const output = new Float32Array(outputLength);
        
        // Linear interpolation resampling
        for (let i = 0; i < outputLength; i++) {
            const index = i * ratio;
            const indexFloor = Math.floor(index);
            const indexCeil = Math.min(indexFloor + 1, input.length - 1);
            const fraction = index - indexFloor;
            
            // Clamp to valid range
            const val1 = input[Math.max(0, Math.min(indexFloor, input.length - 1))];
            const val2 = input[Math.max(0, Math.min(indexCeil, input.length - 1))];
            
            output[i] = val1 * (1 - fraction) + val2 * fraction;
        }
        
        return output;
    }
    
    handleProcessedAudio(data) {
        // Store audio chunks for saving (in system audio mode, audio comes from server)
        if (this.isRecording && this.inputMode === 'system') {
            // Debug: Log first few to verify storage is attempted
            if (!this._storageAttemptCount) {
                this._storageAttemptCount = 0;
            }
            this._storageAttemptCount++;
            if (this._storageAttemptCount <= 3) {
                console.log(`💾 [Storage Attempt ${this._storageAttemptCount}] isRecording=${this.isRecording}, inputMode=${this.inputMode}, has_audio=${!!data.audio}`);
            }
            
            // Always try to store audio chunks, even if they're silence
            if (!data.audio || data.audio === '') {
                // Create silence chunk if no audio data (shouldn't happen, but handle gracefully)
                if (!this._silenceChunkCount) this._silenceChunkCount = 0;
                this._silenceChunkCount++;
                if (this._silenceChunkCount <= 3) {
                    console.warn(`⚠️ [${this._silenceChunkCount}] Received processed_audio with NO audio data - creating silence chunk`);
                }
                // Create a silence chunk to maintain recording continuity
                const silenceChunk = new Float32Array(480); // 30ms at 16kHz
                this.audioChunks.push(silenceChunk);
                if (this._storageAttemptCount <= 3) {
                    console.log(`💾 Stored silence chunk, total chunks: ${this.audioChunks.length}`);
                }
                return;
            }
            
            try {
                // Decode base64 audio to binary string
                const audioBytes = atob(data.audio);
                
                if (audioBytes.length === 0) {
                    console.warn('⚠️ Decoded audio bytes are empty, creating silence chunk');
                    const silenceChunk = new Float32Array(480);
                    this.audioChunks.push(silenceChunk);
                    return;
                }
                
                // Create ArrayBuffer and DataView for proper byte handling
                const buffer = new ArrayBuffer(audioBytes.length);
                const uint8View = new Uint8Array(buffer);
                for (let i = 0; i < audioBytes.length; i++) {
                    uint8View[i] = audioBytes.charCodeAt(i);
                }
                
                // Create Int16Array from the buffer
                const audioArray = new Int16Array(buffer);
                
                if (audioArray.length === 0) {
                    console.warn('⚠️ Audio array is empty, creating silence chunk');
                    const silenceChunk = new Float32Array(480);
                    this.audioChunks.push(silenceChunk);
                    return;
                }
                
                // Check audio level
                const maxSample = Math.max(...Array.from(audioArray).map(Math.abs));
                const audioLevel = maxSample / 32768.0;
                
                // Convert Int16Array to Float32Array (normalize to [-1, 1])
                const float32Audio = new Float32Array(audioArray.length);
                for (let i = 0; i < audioArray.length; i++) {
                    float32Audio[i] = audioArray[i] / 32768.0;
                }
                
                // Store for saving (always store, even if silence)
                this.audioChunks.push(float32Audio);
                
                // Log first few chunks to confirm storage is working
                if (this.audioChunks.length <= 5) {
                    console.log(`💾 [${this.audioChunks.length}] Stored audio chunk: ${float32Audio.length} samples, level=${audioLevel.toFixed(4)} (${(20 * Math.log10(audioLevel + 1e-10)).toFixed(1)} dB)`);
                } else if (this.audioChunks.length % 100 === 0) {
                    // Log every 100th stored chunk
                    const totalSamples = this.audioChunks.reduce((sum, chunk) => sum + chunk.length, 0);
                    const duration = (totalSamples / 16000).toFixed(2);
                    console.log(`💾 Stored ${this.audioChunks.length} chunks (${duration}s of audio)`);
                }
            } catch (error) {
                console.error('❌ Error storing audio chunk:', error);
                console.error('  Audio data length:', data.audio ? data.audio.length : 'null');
                // Create silence chunk on error to maintain continuity
                const silenceChunk = new Float32Array(480);
                this.audioChunks.push(silenceChunk);
                console.log(`💾 Stored silence chunk after error, total chunks: ${this.audioChunks.length}`);
            }
        } else {
            // Debug: Log why chunks aren't being stored
            if (this.inputMode === 'system' && !this.isRecording) {
                if (!this._notRecordingCount) this._notRecordingCount = 0;
                this._notRecordingCount++;
                if (this._notRecordingCount <= 3) {
                    console.warn(`⚠️ [${this._notRecordingCount}] Not storing chunk: isRecording=${this.isRecording}, inputMode=${this.inputMode}`);
                }
            }
        }
        
        // Update VAD indicator (green dot in metrics) with smoothing to prevent swing
        const vadDot = document.getElementById('vadDot');
        const vadStatus = document.getElementById('vadStatus');
        
        if (data.has_speech !== undefined) {
            // If speech_end event was received, force silence state
            if (this._speechEnded) {
                // Reset flag after a few silence chunks
                if (!data.has_speech && data.speech_state === 'silence') {
                    if (!this._speechEndedCount) this._speechEndedCount = 0;
                    this._speechEndedCount++;
                    if (this._speechEndedCount >= 3) {
                        this._speechEnded = false;
                        this._speechEndedCount = 0;
                    }
                }
                // Force silence state
                if (this.currentVadState || this.currentVadDisplayState) {
                    this.currentVadState = false;
                    this.currentVadDisplayState = false;
                    this._vadUiHoldUntil = 0;
                    const vadDot = document.getElementById('vadDot');
                    const vadStatus = document.querySelector('.vad-text');
                    if (vadDot) vadDot.classList.remove('active');
                    if (vadStatus) {
                        vadStatus.textContent = 'No Speech';
                    }
                }
                return;  // Skip smoothing logic when speech has ended
            }
            
            // Add to history (keep last 5 chunks)
            this.vadHistory.push(data.has_speech);
            if (this.vadHistory.length > 5) {
                this.vadHistory.shift();
            }
            
            // Use hysteresis-based smoothing to prevent swing during speech
            // - Easier to enter speech state (2+ out of 5 chunks are speech)
            // - Harder to exit speech state, but respect backend's speech_state signal
            const speechCount = this.vadHistory.filter(v => v).length;
            let smoothedSpeech;
            
            // CRITICAL: If backend says speech ended (speech_state === 'silence'), 
            // exit speech state more aggressively (require 3+ silence chunks instead of 5)
            const backendSaysSilence = (data.speech_state === 'silence');
            
            if (this.currentVadState) {
                // Currently in speech state
                if (backendSaysSilence) {
                    // Backend says silence - exit if 3+ chunks are silence (speechCount <= 2)
                    smoothedSpeech = (speechCount >= 3);  // Stay only if 3+ chunks are speech
                } else {
                    // Backend still says speech - use normal hysteresis (exit if 4+ chunks are silence)
                    smoothedSpeech = (speechCount >= 2);  // Stay if 2+ chunks are speech
                }
            } else {
                // Currently in no-speech state - require stronger evidence to enter (4+ speech chunks)
                // This reduces false positives from background noise
                smoothedSpeech = (speechCount >= 4);  // Enter speech if 4+ out of 5 chunks are speech
            }
            
            // Update VAD detection state (core logic - unchanged)
            if (smoothedSpeech !== this.currentVadState) {
                this.currentVadState = smoothedSpeech;
            }
            
            // UI-only display state with hold mechanism to prevent flicker
            const nowUi = Date.now();
            const backendSaysSpeech = (data.speech_state === 'speech');
            
            // When speech is detected, set hold timer
            if (smoothedSpeech || backendSaysSpeech) {
                this._vadUiHoldUntil = nowUi + this.vadUiHoldMs;
            }
            
            // Determine UI display state:
            // - Show "Speech Detected" if smoothed state says speech OR we're within hold period
            // - Only switch to "No Speech" if hold period expired AND backend says silence
            let displaySpeech = smoothedSpeech || backendSaysSpeech || (
                this.currentVadDisplayState &&
                nowUi < this._vadUiHoldUntil &&
                !backendSaysSilence
            );
            
            // Update UI only if display state changed
            if (displaySpeech !== this.currentVadDisplayState) {
                this.currentVadDisplayState = displaySpeech;
                
                if (displaySpeech) {
                    // Speech detected - show green dot
                    if (vadDot) vadDot.classList.add('active');
                    if (vadStatus) {
                        vadStatus.textContent = 'Speech Detected';
                    }
                } else {
                    // No speech - hide green dot
                    if (vadDot) vadDot.classList.remove('active');
                    if (vadStatus) {
                        vadStatus.textContent = 'No Speech';
                    }
                }
            }
            
            // Update header status (top center dot) using UI display state
            if (this.isRecording) {
                if (this.currentVadDisplayState) {
                    this.updateSystemStatus('speech_detected', 'Speech Detected');
                } else {
                    this.updateSystemStatus('listening', 'Listening...');
                }
            }
        }
        
        // Update metrics
        if (data.audio_level_db !== undefined) {
            const audioLevelEl = document.getElementById('audioLevel');
            if (audioLevelEl) {
                audioLevelEl.textContent = `${data.audio_level_db.toFixed(1)} dB`;
            }
        }
        
        
        if (data.speech_state) {
            const speechStateEl = document.getElementById('speechState');
            if (speechStateEl) {
                speechStateEl.textContent = data.speech_state;
            }
        }
    }
    
    handleSpeechEvent(data) {
        const eventType = data.event;
        const eventData = data.data || {};
        
        if (eventType === 'speech_start') {
            this.updateSystemStatus('speech_detected', 'Speech Detected');
            this.log('Speech detected', 'success');
        } else if (eventType === 'speech_end') {
            // Force clear VAD state immediately when speech ends
            // Set a flag to prevent smoothing from overriding this
            this._speechEnded = true;
            this.currentVadState = false;
            this.currentVadDisplayState = false;
            this._vadUiHoldUntil = 0;
            this.vadHistory = [];  // Clear VAD history
            
            // Update VAD indicator UI immediately
            const vadDot = document.getElementById('vadDot');
            const vadStatus = document.querySelector('.vad-text');
            if (vadDot) vadDot.classList.remove('active');
            if (vadStatus) {
                vadStatus.textContent = 'No Speech';
            }
            
            this.updateSystemStatus('listening', 'Listening...');
            this.log(`Speech ended (duration: ${eventData.duration?.toFixed(2)}s)`, 'info');
        }
    }
    
    handleTranscription(data) {
        const text = data.text || '';
        if (!text) return;
        
        // Remove any interim transcription for this segment
        this.currentInterimText = '';
        const area = document.getElementById('transcriptionArea');
        
        // Remove empty state if present
        const emptyState = area.querySelector('.empty-state');
        if (emptyState) {
            emptyState.remove();
        }
        
        const existingInterim = area.querySelector('.transcription-entry.interim');
        if (existingInterim) {
            existingInterim.remove();
        }
        
        // Calculate latency if timestamp provided
        let latencyInfo = '';
        if (data.timestamp && this.lastTranscriptionTime) {
            const latency = ((data.timestamp - this.lastTranscriptionTime) * 1000).toFixed(0);
            latencyInfo = ` <span class="latency-badge">${latency}ms</span>`;
        }
        this.lastTranscriptionTime = data.timestamp || Date.now() / 1000;
        
        // Create final transcription entry
        const item = document.createElement('div');
        item.className = 'transcription-entry final';
        
        // Language detection badge
        const lang = data.language || 'auto';
        const langBadge = `<span class="lang-badge">${lang.toUpperCase()}</span>`;
        
        item.innerHTML = `
            <div class="transcription-text">${this.escapeHtml(text)}</div>
            <div class="transcription-meta">
                ${langBadge}
                ${latencyInfo}
            </div>
        `;
        
        area.appendChild(item);
        this.scrollToBottom(area);
        
        // Add to history
        this.transcriptionHistory.push({
            text: text,
            timestamp: data.timestamp || Date.now() / 1000,
            confidence: data.confidence || 1.0,
            is_final: true
        });
        
        // Update transcription count based on actual DOM entries
        this.updateTranscriptionStats();
        
        this.log(`Final transcription: ${text.substring(0, 50)}...`, 'success');
    }
    
    handleInterimTranscription(data) {
        const text = data.text || '';
        const incrementalText = data.incremental_text || null;
        if (!text) return;
        
        this.currentInterimText = text;
        const area = document.getElementById('transcriptionArea');
        
        // Remove empty state if present
        const emptyState = area.querySelector('.empty-state');
        if (emptyState) {
            emptyState.remove();
        }
        
        // Get or create interim entry
        let existingInterim = area.querySelector('.transcription-entry.interim');
        let isNewEntry = false;
        
        if (!existingInterim) {
            // Create new interim entry
            existingInterim = document.createElement('div');
            existingInterim.className = 'transcription-entry interim';
            area.appendChild(existingInterim);
            isNewEntry = true;
        }
        
        // Calculate latency if this is first interim
        let latencyInfo = '';
        if (data.timestamp && !this.lastTranscriptionTime) {
            // First transcription - measure from start
            const startTime = this.recordingStartTime || Date.now() / 1000;
            const latency = ((data.timestamp - startTime) * 1000).toFixed(0);
            latencyInfo = ` <span class="latency-badge first">${latency}ms</span>`;
        }
        
        // Update interim entry with streaming effect
        if (incrementalText && !isNewEntry) {
            // Streaming mode: append only new words
            const textElement = existingInterim.querySelector('.transcription-text');
            if (textElement) {
                // Append new words with smooth animation
                const newWordsSpan = document.createElement('span');
                newWordsSpan.className = 'new-words';
                newWordsSpan.textContent = incrementalText;
                textElement.appendChild(document.createTextNode(' '));
                textElement.appendChild(newWordsSpan);
                
                // Remove the 'new-words' class after a moment for smooth transition
                setTimeout(() => {
                    newWordsSpan.classList.remove('new-words');
                    newWordsSpan.classList.add('normal-words');
                }, 500);
            } else {
                // Fallback: update full text
                existingInterim.innerHTML = `
                    <div class="transcription-text">${this.escapeHtml(text)}</div>
                    <div class="transcription-meta">
                        <span class="lang-badge">PROCESSING</span>
                        ${latencyInfo}
                    </div>
                `;
            }
        } else {
            // First update or no incremental text: show full text
            existingInterim.innerHTML = `
                <div class="transcription-text">${this.escapeHtml(text)}</div>
                <div class="transcription-meta">
                    <span class="lang-badge">PROCESSING</span>
                    ${latencyInfo}
                </div>
            `;
        }
        
        this.scrollToBottom(area);
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    scrollToBottom(element) {
        // Smooth scroll to bottom
        element.scrollTo({
            top: element.scrollHeight,
            behavior: 'smooth'
        });
    }
    
    updateTranscriptionStats() {
        // Update transcription count based on actual DOM entries
        const statsEl = document.getElementById('transcriptionStats');
        if (statsEl) {
            const area = document.getElementById('transcriptionArea');
            if (area) {
                // Count all final transcription entries in the DOM
                const finalEntries = area.querySelectorAll('.transcription-entry.final');
                const count = finalEntries.length;
                statsEl.textContent = `${count} ${count === 1 ? 'transcription' : 'transcriptions'}`;
            } else {
                // Fallback to stored count if area doesn't exist
                statsEl.textContent = `${this.transcriptionHistory.length} ${this.transcriptionHistory.length === 1 ? 'transcription' : 'transcriptions'}`;
            }
        }
    }
    
    updateSystemStatus(state, text) {
        const statusIndicator = document.getElementById('systemStatus');
        if (statusIndicator) {
            statusIndicator.className = `status-indicator ${state}`;
            const statusText = statusIndicator.querySelector('.status-text');
            if (statusText) {
                statusText.textContent = text;
            }
            console.log(`🟢 Status updated: ${state} - ${text}`);
        }
    }
    
    updateConnectionStatus(connected) {
        // Update connection indicator if needed
    }
    
    log(message, type = 'info') {
        const logArea = document.getElementById('logArea');
        if (!logArea) return;
        
        const entry = document.createElement('div');
        entry.className = `log-entry ${type}`;
        entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        logArea.appendChild(entry);
        logArea.scrollTop = logArea.scrollHeight;
        
        // Keep only last 50 entries
        while (logArea.children.length > 50) {
            logArea.removeChild(logArea.firstChild);
        }
    }
    
    async saveRecording() {
        if (this.audioChunks.length === 0) {
            alert('No audio recorded. Please start recording first.');
            return;
        }
        
        console.log(`💾 Saving recording: ${this.audioChunks.length} chunks`);
        
        try {
            // Combine all audio chunks
            const totalLength = this.audioChunks.reduce((sum, chunk) => sum + chunk.length, 0);
            
            if (totalLength === 0) {
                alert('No audio data to save. The recording may be empty.');
                return;
            }
            
            const combinedAudio = new Float32Array(totalLength);
            let offset = 0;
            
            for (const chunk of this.audioChunks) {
                if (chunk && chunk.length > 0) {
                    combinedAudio.set(chunk, offset);
                    offset += chunk.length;
                }
            }
            
            const duration = (totalLength / 16000).toFixed(2);
            console.log(`💾 Combined audio: ${totalLength} samples (${duration}s at 16kHz)`);
            
            // Check if audio is all silence (calculate max without spreading to avoid stack overflow)
            let maxLevel = 0;
            for (let i = 0; i < combinedAudio.length; i++) {
                const abs = Math.abs(combinedAudio[i]);
                if (abs > maxLevel) {
                    maxLevel = abs;
                }
            }
            if (maxLevel < 0.001) {
                console.warn('⚠️ Warning: Recording appears to be silence (audio level < 0.001)');
                if (!confirm('The recording appears to be silence. Save anyway?')) {
                    return;
                }
            }
            
            // Convert to WAV
            const wav = this.float32ToWav(combinedAudio, 16000);
            const blob = new Blob([wav], { type: 'audio/wav' });
            const url = URL.createObjectURL(blob);
            
            // Generate filename with timestamp and mode
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const mode = this.inputMode === 'system' ? 'system' : 'mic';
            const filename = `recording_${mode}_${timestamp}.wav`;
            
            // Download
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            
            URL.revokeObjectURL(url);
            this.log(`Recording saved: ${filename} (${duration}s)`, 'success');
            console.log(`✅ Successfully saved: ${filename}`);
            
        } catch (error) {
            console.error('❌ Error saving recording:', error);
            console.error('  Stack:', error.stack);
            alert(`Failed to save recording: ${error.message}\n\nCheck console for details.`);
        }
    }
    
    float32ToWav(buffer, sampleRate) {
        const length = buffer.length;
        const arrayBuffer = new ArrayBuffer(44 + length * 2);
        const view = new DataView(arrayBuffer);
        const samples = new Int16Array(arrayBuffer, 44);
        
        // WAV header
        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };
        
        writeString(0, 'RIFF');
        view.setUint32(4, 36 + length * 2, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(36, 'data');
        view.setUint32(40, length * 2, true);
        
        // Convert float32 to int16
        for (let i = 0; i < length; i++) {
            const s = Math.max(-1, Math.min(1, buffer[i]));
            samples[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        
        return arrayBuffer;
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new STTApp();
});

// TTS methods
STTApp.prototype.playTTS = function() {
    const text = document.getElementById('ttsText').value.trim();
    const speed = parseFloat(document.getElementById('ttsSpeed').value);
    
    if (!text) {
        this.log('Please enter text to synthesize', 'warning');
        this.updateTTSStatus('Please enter text to synthesize', 'warning');
        return;
    }
    
    if (!this.socket || !this.socket.connected) {
        this.log('Not connected to server', 'error');
        this.updateTTSStatus('Not connected to server', 'error');
        return;
    }
    
    // Stop any currently playing audio
    this.stopTTS();
    this.ttsActiveRequestId = `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
    this.ttsExpectedSegments = 0;
    this.ttsStreamComplete = false;
    this.ttsCurrentIndex = 0;
    this.ttsAudioData = []; // Clear previous audio data
    
    // Update UI
    document.getElementById('ttsPlayBtn').disabled = true;
    document.getElementById('ttsStopBtn').disabled = false;
    document.getElementById('ttsSaveBtn').disabled = true; // Disable until audio is received
    this.updateTTSStatus('Detecting language and synthesizing first sentence...', 'info');
    
    // Request TTS synthesis (language will be auto-detected on server).
    // Server streams each sentence as soon as it is synthesized.
    this.socket.emit('synthesize_speech', {
        text: text,
        speed: speed,
        request_id: this.ttsActiveRequestId
    });
};

STTApp.prototype.stopTTS = function(options = {}) {
    const clearStatus = options.clearStatus !== false;

    if (this.ttsAudio) {
        this.ttsAudio.pause();
        this.ttsAudio.currentTime = 0;
        this.ttsAudio = null;
    }
    this.ttsQueue = [];
    this.ttsCurrentIndex = 0;
    this.ttsObjectUrls.forEach((url) => URL.revokeObjectURL(url));
    this.ttsObjectUrls = [];
    // Preserve ttsStreamComplete and ttsAudioData to allow saving after stopping
    // They will be cleared when starting a new TTS request in playTTS
    const wasStreamComplete = this.ttsStreamComplete;
    const hasAudioData = this.ttsAudioData.length > 0;
    this.ttsActiveRequestId = null;
    this.ttsExpectedSegments = 0;
    // Don't reset ttsStreamComplete here - preserve it so save button state is maintained
    // It will be reset in playTTS when starting a new request
    this.isTTSPlaying = false;
    document.getElementById('ttsPlayBtn').disabled = false;
    document.getElementById('ttsStopBtn').disabled = true;
    // Enable save button only if all audio was received and we have data
    // Note: If playback completed naturally, playNextTTSSegment already enabled it
    // This covers the case where user manually stops after all audio was received
    const saveBtn = document.getElementById('ttsSaveBtn');
    if (wasStreamComplete && hasAudioData) {
        saveBtn.disabled = false;
    } else {
        saveBtn.disabled = true;
    }
    if (clearStatus) {
        this.updateTTSStatus('', '');
    }
};

STTApp.prototype.createTTSAudioUrl = function(base64Audio) {
    const audioData = atob(base64Audio);
    const audioArray = new Uint8Array(audioData.length);
    for (let i = 0; i < audioData.length; i++) {
        audioArray[i] = audioData.charCodeAt(i);
    }
    const blob = new Blob([audioArray], { type: 'audio/wav' });
    const audioUrl = URL.createObjectURL(blob);
    this.ttsObjectUrls.push(audioUrl);
    return audioUrl;
};

STTApp.prototype.playNextTTSSegment = function() {
    if (!this.isTTSPlaying) return;

    if (this.ttsQueue.length === 0) {
        if (this.ttsStreamComplete) {
            // All segments received and playback completed - enable save button
            document.getElementById('ttsSaveBtn').disabled = false;
            this.log('All audio finished playing - ready to save', 'success');
            this.stopTTS({ clearStatus: false });
            this.updateTTSStatus('Playback completed', 'success');
        } else {
            const played = this.ttsCurrentIndex;
            const totalText = this.ttsExpectedSegments > 0 ? this.ttsExpectedSegments : '?';
            this.updateTTSStatus(`Synthesizing next sentence... (${played}/${totalText})`, 'info');
        }
        return;
    }

    const segment = this.ttsQueue.shift();
    const currentSegment = this.ttsCurrentIndex + 1;
    const totalSegments = this.ttsExpectedSegments > 0 ? this.ttsExpectedSegments : currentSegment;
    this.ttsCurrentIndex += 1;

    if (this.ttsAudio) {
        this.ttsAudio.pause();
        this.ttsAudio.currentTime = 0;
        this.ttsAudio = null;
    }

    this.ttsAudio = new Audio(segment.url);
    this.ttsAudio.onended = () => {
        this.ttsAudio = null;
        this.playNextTTSSegment();
    };

    this.ttsAudio.onerror = (e) => {
        console.error('TTS audio playback error:', e);
        this.stopTTS();
        this.updateTTSStatus('Playback error', 'error');
    };

    const langName = segment.language === 'en' ? 'English' : segment.language === 'zh' ? 'Chinese' : segment.language === 'ja' ? 'Japanese' : segment.language;
    this.ttsAudio.play().then(() => {
        const preview = segment.text || '';
        this.updateTTSStatus(
            `Playing ${currentSegment}/${totalSegments}: "${preview.substring(0, 50)}${preview.length > 50 ? '...' : ''}" (${langName})`,
            'success'
        );
    }).catch((err) => {
        console.error('Error playing TTS audio:', err);
        this.stopTTS();
        this.updateTTSStatus('Failed to play audio', 'error');
    });
};

STTApp.prototype.handleTTSAudio = function(data) {
    try {
        if (data && data.request_id && !this.ttsActiveRequestId) {
            // No active request (e.g., user already pressed stop), ignore late packets.
            return;
        }
        if (data && data.request_id && this.ttsActiveRequestId && data.request_id !== this.ttsActiveRequestId) {
            // Ignore stale segments from a previous request.
            return;
        }

        // Backward compatibility: non-streaming payload with all segments.
        if (Array.isArray(data.audio_segments) && data.audio_segments.length > 0) {
            this.stopTTS({ clearStatus: false });
            this.isTTSPlaying = true;
            this.ttsExpectedSegments = data.audio_segments.length;
            this.ttsStreamComplete = true;
            this.ttsQueue = [];
            data.audio_segments.forEach((segment) => {
                if (!segment || !segment.audio) return;
                this.ttsAudioData.push(segment.audio); // Store base64 audio for saving
                this.ttsQueue.push({
                    url: this.createTTSAudioUrl(segment.audio),
                    text: segment.text || data.text || '',
                    language: data.language || 'auto'
                });
            });
            document.getElementById('ttsPlayBtn').disabled = true;
            document.getElementById('ttsStopBtn').disabled = false;
            // Keep save button disabled until all audio finishes playing
            document.getElementById('ttsSaveBtn').disabled = true;
            if (!this.ttsAudio) this.playNextTTSSegment();
            return;
        }

        if (!data || !data.audio) {
            throw new Error('No playable TTS audio received');
        }

        this.ttsExpectedSegments = Number(data.segment_count) > 0 ? Number(data.segment_count) : this.ttsExpectedSegments;
        this.ttsStreamComplete = Boolean(data.is_last) || this.ttsStreamComplete;
        this.ttsAudioData.push(data.audio); // Store base64 audio for saving
        this.ttsQueue.push({
            url: this.createTTSAudioUrl(data.audio),
            text: data.text || '',
            language: data.language || 'auto'
        });
        this.isTTSPlaying = true;
        document.getElementById('ttsPlayBtn').disabled = true;
        document.getElementById('ttsStopBtn').disabled = false;
        // Keep save button disabled until all audio finishes playing
        // It will be enabled in playNextTTSSegment when playback completes
        document.getElementById('ttsSaveBtn').disabled = true;
        if (!this.ttsAudio) {
            this.playNextTTSSegment();
        }
        
    } catch (error) {
        console.error('Error handling TTS audio:', error);
        this.stopTTS();
        this.updateTTSStatus('Error processing audio', 'error');
    }
};

STTApp.prototype.handleTTSError = function(data) {
    if (data && data.request_id && this.ttsActiveRequestId && data.request_id !== this.ttsActiveRequestId) {
        return;
    }
    this.stopTTS();
    this.updateTTSStatus(data.message || 'TTS synthesis error', 'error');
    this.log(`TTS Error: ${data.message}`, 'error');
};

STTApp.prototype.updateTTSStatus = function(message, type) {
    const statusEl = document.getElementById('ttsStatus');
    if (!statusEl) return;
    
    if (!message) {
        statusEl.textContent = '';
        statusEl.className = 'tts-status';
        return;
    }
    
    statusEl.textContent = message;
    // Update class to match new CSS structure
    statusEl.className = type ? `tts-status ${type}` : 'tts-status';
};

STTApp.prototype.saveTTS = async function() {
    if (this.ttsAudioData.length === 0) {
        this.log('No TTS audio to save. Please generate audio first.', 'warning');
        this.updateTTSStatus('No audio to save', 'warning');
        return;
    }
    
    try {
        this.log('Saving TTS audio...', 'info');
        this.updateTTSStatus('Saving audio...', 'info');
        
        // Decode all base64 WAV segments and extract PCM data
        const audioBuffers = [];
        let sampleRate = 24000; // Default, will be updated from first segment
        
        for (let i = 0; i < this.ttsAudioData.length; i++) {
            const base64Audio = this.ttsAudioData[i];
            const audioBytes = atob(base64Audio);
            const buffer = new ArrayBuffer(audioBytes.length);
            const view = new Uint8Array(buffer);
            for (let j = 0; j < audioBytes.length; j++) {
                view[j] = audioBytes.charCodeAt(j);
            }
            
            // Parse WAV header to extract PCM data
            const dataView = new DataView(buffer);
            
            // Check RIFF header
            if (String.fromCharCode(dataView.getUint8(0), dataView.getUint8(1), dataView.getUint8(2), dataView.getUint8(3)) !== 'RIFF') {
                throw new Error(`Invalid WAV format in segment ${i + 1}`);
            }
            
            // Find 'data' chunk
            let dataOffset = 44; // Standard WAV header size
            let dataSize = dataView.getUint32(40, true);
            
            // If data chunk is not at standard position, search for it
            if (String.fromCharCode(dataView.getUint8(36), dataView.getUint8(37), dataView.getUint8(38), dataView.getUint8(39)) !== 'data') {
                // Search for 'data' chunk
                for (let offset = 12; offset < buffer.byteLength - 8; offset++) {
                    const chunkId = String.fromCharCode(
                        dataView.getUint8(offset),
                        dataView.getUint8(offset + 1),
                        dataView.getUint8(offset + 2),
                        dataView.getUint8(offset + 3)
                    );
                    if (chunkId === 'data') {
                        dataOffset = offset + 8;
                        dataSize = dataView.getUint32(offset + 4, true);
                        break;
                    }
                }
            }
            
            // Extract sample rate from fmt chunk (usually at offset 24)
            if (i === 0) {
                // Find 'fmt ' chunk
                for (let offset = 12; offset < buffer.byteLength - 8; offset++) {
                    const chunkId = String.fromCharCode(
                        dataView.getUint8(offset),
                        dataView.getUint8(offset + 1),
                        dataView.getUint8(offset + 2),
                        dataView.getUint8(offset + 3)
                    );
                    if (chunkId === 'fmt ') {
                        sampleRate = dataView.getUint32(offset + 12, true);
                        break;
                    }
                }
            }
            
            // Extract PCM data (Int16 samples)
            const pcmData = new Int16Array(buffer, dataOffset, dataSize / 2);
            audioBuffers.push(pcmData);
        }
        
        if (audioBuffers.length === 0) {
            throw new Error('No valid audio data found');
        }
        
        // Combine all PCM data
        const totalLength = audioBuffers.reduce((sum, buf) => sum + buf.length, 0);
        const combinedPCM = new Int16Array(totalLength);
        let offset = 0;
        for (const buffer of audioBuffers) {
            combinedPCM.set(buffer, offset);
            offset += buffer.length;
        }
        
        // Convert Int16Array to Float32Array for WAV conversion
        const float32Audio = new Float32Array(combinedPCM.length);
        for (let i = 0; i < combinedPCM.length; i++) {
            float32Audio[i] = combinedPCM[i] / 32768.0;
        }
        
        // Convert to WAV
        const wav = this.float32ToWav(float32Audio, sampleRate);
        const blob = new Blob([wav], { type: 'audio/wav' });
        const url = URL.createObjectURL(blob);
        
        // Generate filename with timestamp
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `tts_audio_${timestamp}.wav`;
        
        // Download
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        URL.revokeObjectURL(url);
        
        const duration = (combinedPCM.length / sampleRate).toFixed(2);
        this.log(`TTS audio saved: ${filename} (${duration}s, ${this.ttsAudioData.length} segments)`, 'success');
        this.updateTTSStatus(`Audio saved: ${filename}`, 'success');
        console.log(`✅ Successfully saved TTS audio: ${filename}`);
        
    } catch (error) {
        console.error('❌ Error saving TTS audio:', error);
        console.error('  Stack:', error.stack);
        this.log(`Failed to save TTS audio: ${error.message}`, 'error');
        this.updateTTSStatus('Failed to save audio', 'error');
        alert(`Failed to save TTS audio: ${error.message}\n\nCheck console for details.`);
    }
};
