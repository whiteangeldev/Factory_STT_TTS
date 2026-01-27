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
        
        this.init();
    }
    
    init() {
        this.connectWebSocket();
        this.setupEventListeners();
        this.updateSystemStatus('initializing', 'Initializing...');
        this.initializeVADIndicator();
    }
    
    initializeVADIndicator() {
        // Initialize VAD indicator to "No Speech" state
        const vadDot = document.getElementById('vadDot');
        const vadStatus = document.getElementById('vadStatus');
        if (vadDot) {
            vadDot.classList.remove('active');
        }
        if (vadStatus) {
            vadStatus.textContent = 'No Speech';
            vadStatus.style.color = '#666';
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
        
        this.socket.on('recording_status', (data) => {
            console.log('📊 Recording status update:', data);
            if (data.is_recording) {
                this.updateSystemStatus('listening', data.status || 'Recording...');
                if (!this.isRecording) {
                    // Server confirmed recording started
                    this.isRecording = true;
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    document.getElementById('saveBtn').disabled = false;
                }
            } else {
                this.updateSystemStatus('ready', data.status || 'Ready');
                if (this.isRecording) {
                    // Server confirmed recording stopped
                    this.isRecording = false;
                    document.getElementById('startBtn').disabled = false;
                    document.getElementById('stopBtn').disabled = true;
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
            console.log('🔊 Received speech_event:', data.event, data.data);
            this.handleSpeechEvent(data);
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
        document.getElementById('startBtn').addEventListener('click', () => this.startRecording());
        document.getElementById('stopBtn').addEventListener('click', () => this.stopRecording());
        document.getElementById('saveBtn').addEventListener('click', () => this.saveRecording());
        document.getElementById('inputMode').addEventListener('change', (e) => {
            const oldMode = this.inputMode;
            this.inputMode = e.target.value;
            if (this.isRecording) {
                this.log('Please stop recording before changing input mode', 'warning');
                e.target.value = oldMode;
                this.inputMode = oldMode;
            }
        });
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
            
            // Get input mode
            const modeSelect = document.getElementById('inputMode');
            this.inputMode = modeSelect ? modeSelect.value : 'microphone';
            
            let stream;
            if (this.inputMode === 'system') {
                // System audio: Use server-side capture (no browser permissions needed)
                console.log('Using server-side system audio capture');
                this.log('Starting server-side system audio capture...', 'info');
                
                // No browser audio capture needed - server handles it
                this.mediaStream = null;
                this.isRecording = true;
                this.isStopping = false;
                this.audioChunks = [];
                this.transcriptionBuffer = [];
                
                // Reset counters for logging
                this._processedAudioCount = 0;
                this._silenceChunkCount = 0;
                
                console.log('🎙️ System audio mode: Ready to receive audio from server');
                console.log('   Waiting for processed_audio events...');
                
                // Clear transcription area
                const area = document.getElementById('transcriptionArea');
                area.innerHTML = '';
                
                // Notify backend to start server-side capture
                if (this.socket && this.socket.connected) {
                    this.socket.emit('start_recording', {
                        input_mode: 'system',
                        server_capture: true
                    });
                }
                
                // Update UI
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                document.getElementById('saveBtn').disabled = false;
                
                this.updateSystemStatus('listening', 'Listening (system audio)...');
                this.log('System audio started - capturing from server (no browser permissions needed)', 'success');
                return;  // Exit early - no browser audio setup needed
            } else {
                // Request microphone access
                if (!navigator.mediaDevices.getUserMedia) {
                    throw new Error('Microphone access is not available in this browser.');
                }
                
                console.log('Requesting microphone access...');
                this.log('Requesting microphone permission...', 'info');
                
                try {
                    // First try with ideal constraints
                    stream = await navigator.mediaDevices.getUserMedia({
                        audio: {
                            channelCount: { ideal: 1 },
                            sampleRate: { ideal: 16000 },
                            echoCancellation: { ideal: true },
                            noiseSuppression: { ideal: true },
                            autoGainControl: { ideal: true }
                        }
                    });
                } catch (err) {
                    console.warn('Failed with ideal constraints, trying basic constraints:', err);
                    // Fallback to basic constraints
                    stream = await navigator.mediaDevices.getUserMedia({
                        audio: true
                    });
                }
                
                console.log('Microphone access granted');
                this.log('Microphone access granted', 'success');
            }
            
            // Get actual audio track settings
            const audioTracks = stream.getAudioTracks();
            if (audioTracks.length === 0) {
                throw new Error('No audio track available. Please check your input source.');
            }
            
            const audioTrack = audioTracks[0];
            const settings = audioTrack.getSettings();
            console.log('Audio track settings:', settings);
            
            const sourceType = this.inputMode === 'system' ? 'System Audio' : 'Microphone';
            this.log(`${sourceType}: ${settings.sampleRate || 'unknown'} Hz, ${settings.channelCount || 'unknown'} channels`, 'info');
            
            this.mediaStream = stream;
            this.isRecording = true;
            this.isStopping = false;  // Reset stopping flag when starting
            this.audioChunks = [];
            this.transcriptionBuffer = [];
            
            // Clear transcription area
            const area = document.getElementById('transcriptionArea');
            area.innerHTML = '';
            
            // Setup audio processing using AudioWorkletNode (modern) or fallback to ScriptProcessor
            const source = this.audioContext.createMediaStreamSource(stream);
            
            // Use ScriptProcessorNode (deprecated but widely supported)
            // Buffer size must be a power of 2 between 256 and 16384
            // 512 samples = 32ms at 16kHz
            const bufferSize = 512;
            
            // Create processor node
            this.processor = this.audioContext.createScriptProcessor(bufferSize, 1, 1);
            
            this.processor.onaudioprocess = (e) => {
                if (!this.isRecording) return;
                
                const inputData = e.inputBuffer.getChannelData(0);
                
                // Check if we actually have audio data
                const maxLevel = Math.max(...Array.from(inputData).map(Math.abs));
                const rms = Math.sqrt(Array.from(inputData).reduce((sum, val) => sum + val * val, 0) / inputData.length);
                
                // Log first few chunks for debugging
                if (!this.processor._debugCount) {
                    this.processor._debugCount = 0;
                }
                this.processor._debugCount++;
                if (this.processor._debugCount <= 5) {
                    console.log(`[Audio Debug ${this.processor._debugCount}] Samples: ${inputData.length}, SampleRate: ${this.audioContext.sampleRate}Hz, MaxLevel: ${maxLevel.toFixed(6)}, RMS: ${rms.toFixed(6)}`);
                }
                
                if (maxLevel < 0.0001) {
                    // Very quiet or silent - might be muted or no input
                    if (this.processor._debugCount <= 10) {
                        console.warn(`[Audio Debug ${this.processor._debugCount}] Audio level very low: ${maxLevel.toFixed(6)} - check microphone`);
                    }
                }
                
                // Send the raw audio data (will be resampled if needed)
                this.processAudioChunk(inputData, this.audioContext.sampleRate);
            };
            
            // Connect: source -> processor -> destination (to avoid audio feedback, connect to destination)
            source.connect(this.processor);
            this.processor.connect(this.audioContext.destination);
            
            // Notify backend that recording has started
            if (this.socket && this.socket.connected) {
                this.socket.emit('start_recording', {
                    input_mode: this.inputMode,
                    server_capture: false
                });
            }
            
            // Update UI
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            document.getElementById('saveBtn').disabled = false;
            
            this.updateSystemStatus('listening', 'Listening...');
            this.log('Recording started successfully', 'success');
            
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
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
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
        
        // Disconnect processor immediately to stop audio capture
        if (this.processor) {
            try {
                this.processor.disconnect();
            } catch (e) {
                console.warn('Error disconnecting processor:', e);
            }
            this.processor = null;
        }
        
        // Stop media stream tracks
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => {
                try {
                    track.stop();
                } catch (e) {
                    console.warn('Error stopping track:', e);
                }
            });
            this.mediaStream = null;
        }
        
        // Close audio context
        if (this.audioContext) {
            try {
                this.audioContext.close();
            } catch (e) {
                console.warn('Error closing audio context:', e);
            }
            this.audioContext = null;
        }
        
        document.getElementById('startBtn').disabled = false;
        document.getElementById('stopBtn').disabled = true;
        
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
        // Only process if recording and not stopping, and not in system audio mode
        if (!this.isRecording || this.isStopping || this.inputMode === 'system') {
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
        
        // Send to server (only if connected AND still recording AND not stopping)
        // Triple-check to prevent race conditions
        if (this.isRecording && !this.isStopping && this.socket && this.socket.connected) {
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
        
        // Update VAD indicator (green dot in metrics)
        const vadDot = document.getElementById('vadDot');
        const vadStatus = document.getElementById('vadStatus');
        
        if (data.has_speech !== undefined) {
            if (data.has_speech) {
                // Speech detected - show green dot
                if (vadDot) vadDot.classList.add('active');
                if (vadStatus) {
                    vadStatus.textContent = 'Speech Detected';
                    vadStatus.style.color = '#4CAF50';
                }
                
                // Also update header status dot to show speech is detected
                if (this.isRecording) {
                    this.updateSystemStatus('speech_detected', 'Speech Detected');
                }
            } else {
                // No speech - hide green dot
                if (vadDot) vadDot.classList.remove('active');
                if (vadStatus) {
                    if (data.speech_state === 'buffering') {
                        vadStatus.textContent = 'Buffering...';
                        vadStatus.style.color = '#FF9800';
                    } else {
                        vadStatus.textContent = 'No Speech';
                        vadStatus.style.color = '#666';
                    }
                }
                
                // Update header status back to listening if recording and not buffering
                if (this.isRecording && data.speech_state !== 'buffering' && data.speech_state !== 'speech') {
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
        
        if (data.vad_probability !== undefined) {
            const vadProbEl = document.getElementById('vadProbability');
            if (vadProbEl) {
                vadProbEl.textContent = data.vad_probability.toFixed(3);
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
            this.updateSystemStatus('listening', 'Listening...');
            this.log(`Speech ended (duration: ${eventData.duration?.toFixed(2)}s)`, 'info');
        }
    }
    
    handleTranscription(data) {
        const text = data.text || '';
        if (!text) return;
        
        const area = document.getElementById('transcriptionArea');
        const item = document.createElement('div');
        item.className = 'transcription-item final';
        item.textContent = text;
        area.appendChild(item);
        area.scrollTop = area.scrollHeight;
        
        this.log(`Final transcription: ${text.substring(0, 50)}...`, 'success');
    }
    
    handleInterimTranscription(data) {
        const text = data.text || '';
        if (!text) return;
        
        const area = document.getElementById('transcriptionArea');
        
        // Remove previous interim item if exists
        const existingInterim = area.querySelector('.transcription-item.interim');
        if (existingInterim) {
            existingInterim.remove();
        }
        
        // Add new interim item
        const item = document.createElement('div');
        item.className = 'transcription-item interim';
        item.textContent = text;
        area.appendChild(item);
        area.scrollTop = area.scrollHeight;
    }
    
    updateSystemStatus(state, text) {
        const statusBadge = document.getElementById('systemStatus');
        if (statusBadge) {
            statusBadge.className = `status-badge ${state}`;
            const textSpan = statusBadge.querySelector('span:last-child');
            if (textSpan) {
                textSpan.textContent = text;
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
