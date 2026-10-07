/**
 * AEGIS Web App Frontend — minimalist redesign
 * Model pill selector · Full-screen compare page · Custom video player
 */

document.addEventListener('DOMContentLoaded', () => {
    // ── State ──────────────────────────────────────────────────────────────────
    let currentModel = 'vesper.pt';
    let selectedVideoFile = null;
    let selectedSamplePath = null;
    let isAnalyzing = false;

    // ── Core Element Refs ──────────────────────────────────────────────────────
    const chatboxModelContainer = document.getElementById('chatboxModelContainer');
    const chatboxModelBtn = document.getElementById('chatboxModelBtn');
    const chatboxModelName = document.getElementById('chatboxModelName');
    const chatboxModelDropdown = document.getElementById('chatboxModelDropdown');
    const chatboxModelList = document.getElementById('chatboxModelList');
    const chatboxUploadBtn = document.getElementById('chatboxUploadBtn');
    const customModelInput = document.getElementById('customModelInput');
    const resetBtn = document.getElementById('resetBtn');

    // ── Chat History Drawer Refs ──────────────────────────────────────────────
    const sidebarToggleBtn = document.getElementById('sidebarToggleBtn');
    const chatHistoryDrawer = document.getElementById('chatHistoryDrawer');
    const drawerCloseBtn = document.getElementById('drawerCloseBtn');
    const drawerNewChatBtn = document.getElementById('drawerNewChatBtn');
    const drawerBackdrop = document.getElementById('drawerBackdrop');
    const drawerChatList = document.getElementById('drawerChatList');
    const drawerClearBtn = document.getElementById('drawerClearBtn');

    const heroContainer = document.getElementById('heroContainer');
    const samplePillsList = document.getElementById('samplePillsList');
    const messagesContainer = document.getElementById('messagesContainer');
    const chatMain = document.getElementById('chatMain');

    const chatboxCard = document.getElementById('chatboxCard');
    const attachmentTray = document.getElementById('attachmentTray');
    const attachmentName = document.getElementById('attachmentName');
    const attachmentSize = document.getElementById('attachmentSize');
    const removeAttachmentBtn = document.getElementById('removeAttachmentBtn');
    const videoFileInput = document.getElementById('videoFileInput');
    const inputPlaceholder = document.getElementById('inputPlaceholder');
    const sendBtn = document.getElementById('sendBtn');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const dragOverlay = document.getElementById('dragOverlay');

    // ── 1. Upside Dropdown Model Selector (inside chatbox) ─────────────────────
    function openModelDropdown() {
        if (!chatboxModelContainer) return;
        chatboxModelContainer.classList.add('open');
        if (chatboxModelBtn) chatboxModelBtn.setAttribute('aria-expanded', 'true');
    }

    function closeModelDropdown() {
        if (!chatboxModelContainer) return;
        chatboxModelContainer.classList.remove('open');
        if (chatboxModelBtn) chatboxModelBtn.setAttribute('aria-expanded', 'false');
    }

    if (chatboxModelBtn) {
        chatboxModelBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (chatboxModelContainer.classList.contains('open')) {
                closeModelDropdown();
            } else {
                openModelDropdown();
            }
        });
    }

    document.addEventListener('click', (e) => {
        if (chatboxModelContainer && !chatboxModelContainer.contains(e.target)) {
            closeModelDropdown();
        }
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && chatboxModelContainer && chatboxModelContainer.classList.contains('open')) {
            closeModelDropdown();
        }
    });

    async function fetchModels() {
        try {
            const res = await fetch('/api/models');
            const data = await res.json();
            renderModelDropdown(data.models || []);
        } catch (err) {
            console.error('Failed to load models:', err);
        }
    }

    function renderModelDropdown(models) {
        if (!chatboxModelList) return;
        chatboxModelList.innerHTML = '';
        if (!models.length) return;

        // Auto-select first model if none is set
        const found = models.find(m => m.id === currentModel);
        if (!found) {
            currentModel = models[0].id;
            if (chatboxModelName) chatboxModelName.textContent = models[0].name;
        } else {
            if (chatboxModelName) chatboxModelName.textContent = found.name;
        }

        models.forEach(m => {
            const item = document.createElement('button');
            item.type = 'button';
            item.className = `cmd-item${m.id === currentModel ? ' active' : ''}`;
            item.dataset.modelId = m.id;
            item.innerHTML = `
                <span>${escapeHtml(m.name)}</span>
                <span class="cmd-item-check">✓</span>
            `;
            item.addEventListener('click', (e) => {
                e.stopPropagation();
                selectModel(m.id, m.name);
                closeModelDropdown();
            });
            chatboxModelList.appendChild(item);
        });
    }

    function selectModel(modelId, displayName) {
        currentModel = modelId;
        if (chatboxModelName) chatboxModelName.textContent = displayName || modelId;
        // sync dropdown active item
        document.querySelectorAll('.cmd-item').forEach(el => {
            el.classList.toggle('active', el.dataset.modelId === modelId);
        });
    }

    // Upload custom model button
    if (chatboxUploadBtn && customModelInput) {
        chatboxUploadBtn.addEventListener('click', () => {
            closeModelDropdown();
            customModelInput.click();
        });
    }

    if (customModelInput) {
        customModelInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            try {
                selectModel(file.name, file.name.replace('.pt','').replace('.pth',''));
                fetchModels();
            } catch (err) {
                console.error('Custom model error:', err);
            }
        });
    }

    // -------------------------------------------------------------------------
    // 2. Sample Videos
    // -------------------------------------------------------------------------
    async function fetchSamples() {
        try {
            const res = await fetch('/api/sample_videos');
            const data = await res.json();
            renderSamples(data.samples || []);
        } catch (err) {
            console.error('Failed to load samples:', err);
        }
    }

    function renderSamples(samples) {
        samplePillsList.innerHTML = '';
        if (samples.length === 0) {
            document.getElementById('samplePillsContainer').style.display = 'none';
            return;
        }

        samples.forEach(s => {
            const pill = document.createElement('button');
            pill.type = 'button';
            pill.className = 'sample-pill';
            pill.textContent = `${s.class}: ${s.name}`;
            pill.addEventListener('click', () => {
                attachSampleVideo(s.name, s.path);
            });
            samplePillsList.appendChild(pill);
        });
    }

    function attachSampleVideo(name, relPath) {
        selectedVideoFile = null;
        selectedSamplePath = relPath;

        attachmentName.textContent = name;
        attachmentSize.textContent = 'Sample Video';
        chatboxCard.classList.add('has-attachment');
        attachmentTray.classList.add('visible');
        attachmentTray.style.display = '';
        inputPlaceholder.textContent = 'Ready to analyze';

        updateSendState();
    }

    // -------------------------------------------------------------------------
    // 3. Video Attachment Handlers
    // -------------------------------------------------------------------------
    function formatBytes(bytes, decimals = 1) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }

    function attachFile(file) {
        if (!file) return;
        selectedVideoFile = file;
        selectedSamplePath = null;

        attachmentName.textContent = file.name;
        attachmentSize.textContent = formatBytes(file.size);
        chatboxCard.classList.add('has-attachment');
        attachmentTray.classList.add('visible');
        attachmentTray.style.display = '';
        inputPlaceholder.textContent = 'Ready to analyze';

        updateSendState();
    }

    function clearAttachment() {
        selectedVideoFile = null;
        selectedSamplePath = null;
        videoFileInput.value = '';
        chatboxCard.classList.remove('has-attachment');
        attachmentTray.classList.remove('visible');
        attachmentTray.style.display = '';
        inputPlaceholder.textContent = 'Attach a video or drag and drop here...';
        updateSendState();
    }

    removeAttachmentBtn.addEventListener('click', clearAttachment);

    videoFileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            attachFile(e.target.files[0]);
        }
    });

    function updateSendState() {
        const canSend = (selectedVideoFile !== null || selectedSamplePath !== null) && !isAnalyzing;
        sendBtn.disabled = !canSend;
        sendBtn.classList.toggle('active', canSend);
    }

    // -------------------------------------------------------------------------
    // 4. Drag & Drop Support
    // -------------------------------------------------------------------------
    let dragCounter = 0;

    window.addEventListener('dragenter', (e) => {
        e.preventDefault();
        dragCounter++;
        if (e.dataTransfer.types && e.dataTransfer.types.includes('Files')) {
            dragOverlay.classList.add('active');
        }
    });

    window.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dragCounter--;
        if (dragCounter <= 0) {
            dragCounter = 0;
            dragOverlay.classList.remove('active');
        }
    });

    window.addEventListener('dragover', (e) => {
        e.preventDefault();
    });

    window.addEventListener('drop', (e) => {
        e.preventDefault();
        dragCounter = 0;
        dragOverlay.classList.remove('active');

        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            const file = e.dataTransfer.files[0];
            if (file.type.startsWith('video/') || file.name.match(/\.(mp4|avi|mov|mkv)$/i)) {
                attachFile(file);
            }
        }
    });

    // -------------------------------------------------------------------------
    // 5. Submit / Analysis Execution
    // -------------------------------------------------------------------------
    sendBtn.addEventListener('click', () => {
        if (!selectedVideoFile && !selectedSamplePath) return;
        if (isAnalyzing) return;
        executeAnalysis();
    });

    async function executeAnalysis() {
        isAnalyzing = true;
        updateSendState();

        // Snapshot names BEFORE clearAttachment() wipes state
        const videoName = selectedVideoFile ? selectedVideoFile.name : attachmentName.textContent;
        const videoSize = selectedVideoFile ? formatBytes(selectedVideoFile.size) : 'Sample';
        const modelDisplayName = (chatboxModelName && chatboxModelName.textContent)
            ? chatboxModelName.textContent
            : currentModel;

        // 1. Hide Hero Greeting smoothly
        if (heroContainer) {
            heroContainer.style.display = 'none';
        }

        // 2. Append User Message
        appendUserMessage(videoName, videoSize, modelDisplayName);

        // 3. Append Assistant Message (Loading state)
        const { assistantRow, typingIndicator, statusText } = createAssistantLoadingMessage();
        messagesContainer.appendChild(assistantRow);
        scrollToBottom();

        // Animated status ticker
        const statuses = [
            'Reading video frames & timestamps...',
            'Detecting objects & spatial context...',
            'Encoding visual representations with ResNet18...',
            'Evaluating temporal dynamics with AEGIS MIL model...',
            'Finalizing anomaly probabilities...'
        ];
        let statusIdx = 0;
        const statusTimer = setInterval(() => {
            statusIdx = (statusIdx + 1) % statuses.length;
            if (statusText) statusText.textContent = statuses[statusIdx];
        }, 2200);

        // Prepare FormData
        const formData = new FormData();
        formData.append('model', currentModel);
        if (selectedVideoFile) {
            formData.append('video', selectedVideoFile);
        } else if (selectedSamplePath) {
            formData.append('sample_path', selectedSamplePath);
        }

        // Clear input attachment for next video
        clearAttachment();

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData,
            });

            clearInterval(statusTimer);
            const data = await response.json();

            if (!response.ok || data.status === 'error') {
                throw new Error(data.message || 'Analysis failed.');
            }

            // Replace loading with final result
            renderAnalysisResult(assistantRow, data);
            addChatToHistory(videoName, modelDisplayName, data);
        } catch (error) {
            clearInterval(statusTimer);
            renderAnalysisError(assistantRow, error.message);
        } finally {
            isAnalyzing = false;
            updateSendState();
            scrollToBottom();
        }
    }


    function appendUserMessage(filename, size, model) {
        const row = document.createElement('div');
        row.className = 'message-row user';
        row.innerHTML = `
            <div class="user-bubble">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polygon points="23 7 16 12 23 17 23 7"></polygon>
                    <rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect>
                </svg>
                <div class="user-bubble-info">
                    <div class="user-bubble-filename">${escapeHtml(filename)}</div>
                    <div class="user-bubble-model">${size} · Model: ${escapeHtml(model)}</div>
                </div>
            </div>
        `;
        messagesContainer.appendChild(row);
    }

    function createAssistantLoadingMessage() {
        const row = document.createElement('div');
        row.className = 'message-row assistant';
        row.innerHTML = `
            <div class="assistant-avatar">
                <svg width="18" height="18" viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="16" cy="16" r="14"></circle>
                    <circle cx="16" cy="16" r="7"></circle>
                </svg>
            </div>
            <div class="assistant-content">
                <div class="typing-indicator">
                    <div class="typing-dots">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                    <span class="typing-status-text">Reading video frames & timestamps...</span>
                </div>
            </div>
        `;
        return {
            assistantRow: row,
            typingIndicator: row.querySelector('.typing-indicator'),
            statusText: row.querySelector('.typing-status-text'),
        };
    }

    function renderAnalysisResult(assistantRow, data) {
        const content = assistantRow.querySelector('.assistant-content');
        if (!content) return;

        const isAnomaly = data.is_anomaly;
        const predClass = data.prediction;
        const conf = data.confidence;
        const anomScore = data.anomaly_score;
        const peakWindow = `${data.top_window.start_formatted} - ${data.top_window.end_formatted}`;
        const duration = `${data.video_info.duration}s (${data.video_info.total_frames} frames)`;

        // Format probabilities bars
        let probBarsHtml = '';
        const sortedProbs = Object.entries(data.probabilities || {}).sort((a, b) => b[1] - a[1]);

        sortedProbs.forEach(([clsName, pct]) => {
            const isTop = clsName === predClass;
            let fillClass = 'prob-fill';
            if (isTop) {
                fillClass += isAnomaly ? ' highlight danger' : ' highlight success';
            }
            probBarsHtml += `
                <div class="prob-row">
                    <span class="prob-class-name">${escapeHtml(clsName)}</span>
                    <div class="prob-track">
                        <div class="${fillClass}" style="width: ${pct}%"></div>
                    </div>
                    <span class="prob-pct">${pct}%</span>
                </div>
            `;
        });

        // ---------------------------------------------------------------------
        // Custom Video Player with Full Window, Jump to Event, Play/Pause, Mute
        // ---------------------------------------------------------------------
        let videoPlayerHtml = '';
        if (data.video_url) {
            const vidId = `vid_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
            const eventStart = data.top_window ? data.top_window.start : 0;
            const eventStartFmt = data.top_window ? data.top_window.start_formatted : '00:00';
            const totalDur = data.video_info ? data.video_info.duration : 1;
            const markerPct = Math.min(100, Math.max(0, (eventStart / (totalDur || 1)) * 100));

            videoPlayerHtml = `
                <div class="custom-video-box" id="${vidId}">
                    <div class="video-media-stage">
                        <video class="custom-video-el" preload="metadata" playsinline src="${escapeHtml(data.video_url)}"></video>
                        <div class="video-center-play-overlay">
                            <button class="vcenter-play-btn" type="button" aria-label="Play">
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                                    <polygon points="6 3 20 12 6 21 6 3"></polygon>
                                </svg>
                            </button>
                        </div>
                    </div>
                    <div class="custom-video-controls">
                        <!-- Scrubber Bar -->
                        <div class="vscrubber-container">
                            <div class="vscrubber-preview-tooltip">00:00</div>
                            <div class="vscrubber-track">
                                <div class="vscrubber-buffered"></div>
                                <div class="vscrubber-progress"></div>
                                <div class="vscrubber-thumb"></div>
                                <div class="vscrubber-marker" style="left: ${markerPct}%;" title="Anomaly Event Window: ${eventStartFmt}">
                                    <div class="vscrubber-marker-tooltip">Event ${eventStartFmt}</div>
                                </div>
                            </div>
                        </div>
                        <!-- Controls Row -->
                        <div class="vctrl-row">
                            <div class="vctrl-left">
                                <button class="vctrl-btn vctrl-play-pause" type="button" title="Play / Pause">
                                    <svg class="icon-play" width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                                        <polygon points="5 3 19 12 5 21 5 3"></polygon>
                                    </svg>
                                    <svg class="icon-pause" width="16" height="16" viewBox="0 0 24 24" fill="currentColor" style="display:none;">
                                        <rect x="6" y="4" width="4" height="16"></rect>
                                        <rect x="14" y="4" width="4" height="16"></rect>
                                    </svg>
                                </button>
                                <span class="vctrl-time">
                                    <span class="vtime-current">00:00</span>
                                    <span class="vtime-sep">/</span>
                                    <span class="vtime-total">${data.video_info.duration_formatted}</span>
                                </span>
                                <button class="vctrl-btn vctrl-jump-event" type="button" title="Jump video directly to detected event (${eventStartFmt})">
                                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                                        <polygon points="5 3 19 12 5 21 5 3"></polygon>
                                    </svg>
                                    <span>Jump to event (${eventStartFmt})</span>
                                </button>
                            </div>
                            <div class="vctrl-right">
                                <button class="vctrl-btn vctrl-mute" type="button" title="Mute / Unmute">
                                    <svg class="icon-unmuted" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                        <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
                                        <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>
                                    </svg>
                                    <svg class="icon-muted" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="display:none;">
                                        <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
                                        <line x1="23" y1="9" x2="17" y2="15"></line>
                                        <line x1="17" y1="9" x2="23" y2="15"></line>
                                    </svg>
                                </button>
                                <button class="vctrl-btn vctrl-fullscreen" type="button" title="Full Window / Exit Full Window">
                                    <svg class="icon-expand" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                        <polyline points="15 3 21 3 21 9"></polyline>
                                        <polyline points="9 21 3 21 3 15"></polyline>
                                        <line x1="21" y1="3" x2="14" y2="10"></line>
                                        <line x1="3" y1="21" x2="10" y2="14"></line>
                                    </svg>
                                    <svg class="icon-compress" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="display:none;">
                                        <polyline points="4 14 10 14 10 20"></polyline>
                                        <polyline points="20 10 14 10 14 4"></polyline>
                                        <line x1="14" y1="10" x2="21" y2="3"></line>
                                        <line x1="3" y1="21" x2="10" y2="14"></line>
                                    </svg>
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        content.innerHTML = `
            <div class="result-card">
                <!-- Summary Text -->
                <div class="result-summary">
                    ${formatMarkdown(data.summary)}
                </div>

                <!-- Stats Metric Grid -->
                <div class="stats-grid">
                    <div class="stat-tile">
                        <div class="stat-tile-label">Classification</div>
                        <div class="stat-tile-value ${isAnomaly ? 'anomaly' : 'normal'}">${escapeHtml(predClass)}</div>
                    </div>
                    <div class="stat-tile">
                        <div class="stat-tile-label">Confidence</div>
                        <div class="stat-tile-value">${conf}%</div>
                    </div>
                    <div class="stat-tile">
                        <div class="stat-tile-label">Anomaly Score</div>
                        <div class="stat-tile-value ${anomScore > 50 ? 'anomaly' : ''}">${anomScore}%</div>
                    </div>
                    <div class="stat-tile">
                        <div class="stat-tile-label">Peak Window</div>
                        <div class="stat-tile-value" style="font-size:15px;">${peakWindow}</div>
                    </div>
                </div>

                <!-- Embedded Custom Video Player -->
                ${videoPlayerHtml}

                <!-- Probability Distribution -->
                <div class="prob-section">
                    <div class="prob-header">Class Probability Breakdown</div>
                    <div class="prob-bars">
                        ${probBarsHtml}
                    </div>
                </div>
            </div>
        `;

        // Attach custom video listeners if video present
        if (data.video_url) {
            setupCustomVideoPlayer(assistantRow, data);
        }
    }

    function setupCustomVideoPlayer(assistantRow, data) {
        const box = assistantRow.querySelector('.custom-video-box');
        if (!box) return;

        const video = box.querySelector('.custom-video-el');
        const stage = box.querySelector('.video-media-stage');
        const playPauseBtn = box.querySelector('.vctrl-play-pause');
        const iconPlay = box.querySelector('.icon-play');
        const iconPause = box.querySelector('.icon-pause');
        const centerPlayOverlay = box.querySelector('.video-center-play-overlay');
        const centerPlayBtn = box.querySelector('.vcenter-play-btn');
        const timeCurrent = box.querySelector('.vtime-current');
        const timeTotal = box.querySelector('.vtime-total');
        const jumpEventBtn = box.querySelector('.vctrl-jump-event');
        const muteBtn = box.querySelector('.vctrl-mute');
        const iconUnmuted = box.querySelector('.icon-unmuted');
        const iconMuted = box.querySelector('.icon-muted');
        const fullWindowBtn = box.querySelector('.vctrl-fullscreen');
        const iconExpand = box.querySelector('.icon-expand');
        const iconCompress = box.querySelector('.icon-compress');
        const scrubberContainer = box.querySelector('.vscrubber-container');
        const scrubberTrack = box.querySelector('.vscrubber-track');
        const scrubberProgress = box.querySelector('.vscrubber-progress');
        const scrubberBuffered = box.querySelector('.vscrubber-buffered');
        const scrubberThumb = box.querySelector('.vscrubber-thumb');
        const scrubberMarker = box.querySelector('.vscrubber-marker');
        const scrubberPreview = box.querySelector('.vscrubber-preview-tooltip');

        const eventStart = data.top_window ? data.top_window.start : 0;
        let isDragging = false;
        let wasPlayingBeforeDrag = false;

        function formatSec(s) {
            if (isNaN(s) || s < 0) s = 0;
            const m = Math.floor(s / 60);
            const sec = Math.floor(s % 60);
            return `${m.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`;
        }

        function togglePlay() {
            if (video.paused || video.ended) {
                video.play();
            } else {
                video.pause();
            }
        }

        // Play / Pause State Sync
        video.addEventListener('play', () => {
            iconPlay.style.display = 'none';
            iconPause.style.display = 'inline-block';
            centerPlayOverlay.classList.add('hidden');
            resetControlsIdleTimer();
        });

        video.addEventListener('pause', () => {
            iconPlay.style.display = 'inline-block';
            iconPause.style.display = 'none';
            centerPlayOverlay.classList.remove('hidden');
            box.classList.remove('controls-idle');
        });

        playPauseBtn.addEventListener('click', togglePlay);
        centerPlayBtn.addEventListener('click', togglePlay);
        stage.addEventListener('click', (e) => {
            if (e.target.closest('.vcenter-play-btn')) return;
            togglePlay();
        });

        // Time Update & Progress (avoid overwriting while dragging)
        video.addEventListener('timeupdate', () => {
            if (isDragging) return;
            timeCurrent.textContent = formatSec(video.currentTime);
            const pct = (video.currentTime / (video.duration || 1)) * 100;
            const clampedPct = Math.min(100, Math.max(0, pct));
            scrubberProgress.style.width = `${clampedPct}%`;
            if (scrubberThumb) scrubberThumb.style.left = `${clampedPct}%`;
        });

        video.addEventListener('loadedmetadata', () => {
            timeTotal.textContent = formatSec(video.duration);
        });

        // Buffer Progress
        video.addEventListener('progress', () => {
            if (video.buffered.length > 0) {
                const buffEnd = video.buffered.end(video.buffered.length - 1);
                const buffPct = (buffEnd / (video.duration || 1)) * 100;
                scrubberBuffered.style.width = `${Math.min(100, Math.max(0, buffPct))}%`;
            }
        });

        // ── Draggable Timeline Scrubber (Seek Anywhere & Drag by that Red Marker) ─
        function getSeekRatio(clientX) {
            const rect = scrubberTrack.getBoundingClientRect();
            if (rect.width <= 0) return 0;
            const clampedX = Math.max(0, Math.min(clientX - rect.left, rect.width));
            return clampedX / rect.width;
        }

        function updateSeekPosition(clientX, commitVideoTime = true) {
            const ratio = getSeekRatio(clientX);
            const pct = Math.min(100, Math.max(0, ratio * 100));
            const duration = video.duration || 0;
            const seekTime = ratio * duration;

            scrubberProgress.style.width = `${pct}%`;
            if (scrubberThumb) scrubberThumb.style.left = `${pct}%`;
            timeCurrent.textContent = formatSec(seekTime);

            if (commitVideoTime && duration > 0) {
                video.currentTime = seekTime;
            }
            return { ratio, seekTime };
        }

        // Pointer / Drag Event Listeners on Container, Marker, and Thumb
        function startScrubbing(e) {
            if (e.button !== 0 && e.pointerType === 'mouse') return;
            e.preventDefault();
            e.stopPropagation();

            isDragging = true;
            scrubberContainer.classList.add('is-dragging');
            box.classList.add('is-dragging-scrubber');
            document.body.classList.add('is-scrubbing-active');

            wasPlayingBeforeDrag = !video.paused && !video.ended;
            if (wasPlayingBeforeDrag) {
                video.pause();
            }

            updateSeekPosition(e.clientX, true);

            function onWindowPointerMove(moveEvt) {
                if (!isDragging) return;
                moveEvt.preventDefault();
                const { seekTime } = updateSeekPosition(moveEvt.clientX, true);

                if (scrubberPreview) {
                    const rect = scrubberTrack.getBoundingClientRect();
                    const clampedX = Math.max(0, Math.min(moveEvt.clientX - rect.left, rect.width));
                    scrubberPreview.style.left = `${clampedX}px`;
                    scrubberPreview.textContent = formatSec(seekTime);
                }
            }

            function onWindowPointerUp(upEvt) {
                if (!isDragging) return;
                isDragging = false;
                scrubberContainer.classList.remove('is-dragging');
                box.classList.remove('is-dragging-scrubber');
                document.body.classList.remove('is-scrubbing-active');

                window.removeEventListener('pointermove', onWindowPointerMove, { capture: true });
                window.removeEventListener('pointerup', onWindowPointerUp, { capture: true });
                window.removeEventListener('pointercancel', onWindowPointerUp, { capture: true });

                updateSeekPosition(upEvt.clientX, true);

                if (wasPlayingBeforeDrag) {
                    video.play();
                }
            }

            window.addEventListener('pointermove', onWindowPointerMove, { capture: true, passive: false });
            window.addEventListener('pointerup', onWindowPointerUp, { capture: true, passive: false });
            window.addEventListener('pointercancel', onWindowPointerUp, { capture: true, passive: false });
        }

        if (scrubberContainer) {
            scrubberContainer.addEventListener('pointerdown', startScrubbing);
        }
        if (scrubberMarker) {
            scrubberMarker.addEventListener('pointerdown', startScrubbing);
        }
        if (scrubberThumb) {
            scrubberThumb.addEventListener('pointerdown', startScrubbing);
        }

        // Hover tooltip preview when not actively dragging
        if (scrubberContainer) {
            scrubberContainer.addEventListener('pointermove', (e) => {
                if (isDragging || !scrubberPreview) return;
                const ratio = getSeekRatio(e.clientX);
                const rect = scrubberTrack.getBoundingClientRect();
                const clampedX = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
                scrubberPreview.style.left = `${clampedX}px`;
                scrubberPreview.textContent = formatSec(ratio * (video.duration || 0));
            });
        }

        // Jump to Event Button
        jumpEventBtn.addEventListener('click', () => {
            video.currentTime = Math.max(0, eventStart);
            video.play();
        });

        // Mute / Unmute
        muteBtn.addEventListener('click', () => {
            video.muted = !video.muted;
            if (video.muted) {
                iconUnmuted.style.display = 'none';
                iconMuted.style.display = 'inline-block';
            } else {
                iconUnmuted.style.display = 'inline-block';
                iconMuted.style.display = 'none';
            }
        });

        // ── Full-Screen / Full-Window Mode with Custom UI Controls ─────────────
        function isNativeFullscreen() {
            return !!(document.fullscreenElement === box || document.webkitFullscreenElement === box);
        }

        function isFullActive() {
            return isNativeFullscreen() || box.classList.contains('is-full-window');
        }

        function updateFullscreenUIState(active) {
            if (active) {
                box.classList.add('is-full-window');
                iconExpand.style.display = 'none';
                iconCompress.style.display = 'inline-block';
                fullWindowBtn.title = 'Exit Fullscreen (Esc)';
                document.body.style.overflow = 'hidden';
            } else {
                box.classList.remove('is-full-window');
                box.classList.remove('controls-idle');
                iconExpand.style.display = 'inline-block';
                iconCompress.style.display = 'none';
                fullWindowBtn.title = 'Full Screen (Full Window)';
                document.body.style.overflow = '';
            }
        }

        async function enterFullscreen() {
            try {
                if (box.requestFullscreen) {
                    await box.requestFullscreen();
                } else if (box.webkitRequestFullscreen) {
                    await box.webkitRequestFullscreen();
                }
            } catch (err) {
                console.warn('Native requestFullscreen denied/unavailable, fallback to full-window CSS:', err);
            }
            // Always set full-window styling class so UI controls look consistent
            updateFullscreenUIState(true);
        }

        async function exitFullscreen() {
            try {
                if (document.fullscreenElement || document.webkitFullscreenElement) {
                    if (document.exitFullscreen) {
                        await document.exitFullscreen();
                    } else if (document.webkitExitFullscreen) {
                        await document.webkitExitFullscreen();
                    }
                }
            } catch (err) {
                console.warn('exitFullscreen error:', err);
            }
            updateFullscreenUIState(false);
        }

        function toggleFullscreen() {
            if (isFullActive()) {
                exitFullscreen();
            } else {
                enterFullscreen();
            }
        }

        fullWindowBtn.addEventListener('click', toggleFullscreen);

        // Sync with browser native fullscreen changes (e.g. Esc pressed)
        const handleFullscreenChange = () => {
            const isFs = isNativeFullscreen();
            updateFullscreenUIState(isFs);
        };
        document.addEventListener('fullscreenchange', handleFullscreenChange);
        document.addEventListener('webkitfullscreenchange', handleFullscreenChange);

        // Double-click stage to toggle fullscreen
        stage.addEventListener('dblclick', (e) => {
            if (e.target.closest('.vcenter-play-btn')) return;
            toggleFullscreen();
        });

        // Fullscreen Controls Idle Auto-Hide (fade out when inactive during playback)
        let controlsIdleTimer = null;
        function resetControlsIdleTimer() {
            box.classList.remove('controls-idle');
            if (controlsIdleTimer) clearTimeout(controlsIdleTimer);
            if (isFullActive() && !video.paused) {
                controlsIdleTimer = setTimeout(() => {
                    if (isFullActive() && !video.paused && !isDragging) {
                        box.classList.add('controls-idle');
                    }
                }, 2600);
            }
        }

        box.addEventListener('mousemove', resetControlsIdleTimer);
        box.addEventListener('touchstart', resetControlsIdleTimer, { passive: true });

        // Keyboard navigation when fullscreen or video hovered
        box.setAttribute('tabindex', '0');
        box.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            if (e.key === ' ' || e.key === 'k') {
                e.preventDefault();
                togglePlay();
            } else if (e.key === 'f' || e.key === 'F') {
                e.preventDefault();
                toggleFullscreen();
            } else if (e.key === 'm' || e.key === 'M') {
                e.preventDefault();
                muteBtn.click();
            } else if (e.key === 'ArrowLeft') {
                e.preventDefault();
                video.currentTime = Math.max(0, video.currentTime - 5);
            } else if (e.key === 'ArrowRight') {
                e.preventDefault();
                video.currentTime = Math.min(video.duration || 0, video.currentTime + 5);
            } else if (e.key === 'Escape' && isFullActive()) {
                exitFullscreen();
            }
        });
    }

    function renderAnalysisError(assistantRow, message) {
        const content = assistantRow.querySelector('.assistant-content');
        if (!content) return;
        content.innerHTML = `
            <div class="result-card" style="border-color: #ef4444;">
                <div style="color: #f87171; font-weight: 500; margin-bottom: 4px;">Analysis Failed</div>
                <div style="font-size: 13.5px; color: #b4b4b4;">${escapeHtml(message)}</div>
            </div>
        `;
    }

    // ── 6. Full-Screen Compare Page ────────────────────────────────────────────
    const compareBtn = document.getElementById('compareBtn');
    const comparePage = document.getElementById('comparePage');
    const compareCloseBtn = document.getElementById('compareCloseBtn');
    const compareModelCheckboxes = document.getElementById('compareModelCheckboxes');
    const compareSelectionCount = document.getElementById('compareSelectionCount');
    const compareCardsGrid = document.getElementById('compareCardsGrid');
    const compareChartBox = document.getElementById('compareChartBox');
    const compareChartLegend = document.getElementById('compareChartLegend');
    const singleModelDeepdive = document.getElementById('singleModelDeepdive');

    let comparisonData = null;
    let selectedComparisonKeys = ['orion', 'vesper', 'baseline'];

    const MODEL_THEME_COLORS = {
        orion:    '#10a37f',
        vesper:   '#3b82f6',
        baseline: '#71717a',
    };

    async function loadComparisonData() {
        if (comparisonData) return comparisonData;
        try {
            const res = await fetch('/api/models/comparison');
            const data = await res.json();
            comparisonData = data.comparison || {};
            return comparisonData;
        } catch (err) {
            console.error('Failed to load comparison data:', err);
            return {};
        }
    }

    function openComparePage() {
        comparePage.classList.add('open');
        document.body.style.overflow = 'hidden';
        loadComparisonData().then(() => renderComparisonUI());
    }

    function closeComparePage() {
        comparePage.classList.remove('open');
        document.body.style.overflow = '';
    }

    if (compareBtn) compareBtn.addEventListener('click', openComparePage);
    if (compareCloseBtn) compareCloseBtn.addEventListener('click', closeComparePage);

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && comparePage.classList.contains('open') && !document.querySelector('.custom-video-box.is-full-window')) {
            closeComparePage();

        }
    });

    function renderComparisonUI() {
        if (!comparisonData) return;
        const allKeys = Object.keys(comparisonData);

        // Render Checkboxes
        compareModelCheckboxes.innerHTML = '';
        allKeys.forEach(k => {
            const m = comparisonData[k];
            const isChecked = selectedComparisonKeys.includes(k);
            const label = document.createElement('label');
            label.className = `compare-checkbox-pill ${isChecked ? 'active' : ''}`;
            label.innerHTML = `
                <input type="checkbox" value="${k}" ${isChecked ? 'checked' : ''} style="display:none;">
                <span class="checkbox-dot" style="background-color: ${MODEL_THEME_COLORS[k] || '#888'};"></span>
                <span class="checkbox-name">${escapeHtml(m.name)}</span>
            `;

            label.querySelector('input').addEventListener('change', (e) => {
                const checked = e.target.checked;
                if (checked) {
                    if (selectedComparisonKeys.length >= 4) {
                        e.target.checked = false;
                        alert('You can compare a maximum of 4 models simultaneously.');
                        return;
                    }
                    selectedComparisonKeys.push(k);
                } else {
                    if (selectedComparisonKeys.length <= 1) {
                        e.target.checked = true;
                        return; // Keep at least 1 model
                    }
                    selectedComparisonKeys = selectedComparisonKeys.filter(x => x !== k);
                }
                renderComparisonUI();
            });
            compareModelCheckboxes.appendChild(label);
        });

        // Update count label
        compareSelectionCount.textContent = `${selectedComparisonKeys.length} model${selectedComparisonKeys.length > 1 ? 's' : ''} selected`;

        const activeModels = selectedComparisonKeys.map(k => ({ key: k, ...comparisonData[k] }));

        // Render Comparative Chart
        renderBenchmarkChart(activeModels);

        // Render OpenRouter Cards Grid
        renderModelCards(activeModels);

        // Render Single Model Deepdive if 1 model selected
        if (activeModels.length === 1) {
            renderSingleModelDeepdive(activeModels[0]);
            singleModelDeepdive.style.display = 'block';
        } else {
            singleModelDeepdive.style.display = 'none';
        }
    }

    function renderModelCards(models) {
        compareCardsGrid.innerHTML = '';
        // Set dynamic grid columns based on count
        compareCardsGrid.className = `compare-cards-grid cols-${Math.min(models.length, 4)}`;

        models.forEach(m => {
            const card = document.createElement('div');
            card.className = 'openrouter-card';
            const color = MODEL_THEME_COLORS[m.key] || '#888';

            let classesBadges = (m.classes || []).map(c => `<span class="class-pill">${escapeHtml(c)}</span>`).join('');
            let prosList = (m.pros || []).map(p => `<li>${escapeHtml(p)}</li>`).join('');

            card.innerHTML = `
                <!-- Top Brand Header -->
                <div class="or-card-header">
                    <div class="or-card-title-group">
                        <div class="or-name-row">
                            <span class="or-color-strip" style="background-color: ${color};"></span>
                            <h3 class="or-model-name">${escapeHtml(m.name)}</h3>
                            <span class="or-badge">${escapeHtml(m.badge)}</span>
                        </div>
                        <div class="or-checkpoint-id">${escapeHtml(m.id)}</div>
                    </div>
                    <button class="or-select-btn ${currentModel === m.id ? 'active' : ''}" type="button">
                        ${currentModel === m.id ? 'Active Model' : 'Use Model'}
                    </button>
                </div>

                <p class="or-description">${escapeHtml(m.description)}</p>

                <!-- Key Benchmark Badges -->
                <div class="or-benchmark-tiles">
                    <div class="or-bench-tile">
                        <div class="or-bench-label">Video Accuracy</div>
                        <div class="or-bench-val">${m.benchmarks.accuracy}%</div>
                    </div>
                    <div class="or-bench-tile">
                        <div class="or-bench-label">Macro F1</div>
                        <div class="or-bench-val">${m.benchmarks.macro_f1}%</div>
                    </div>
                    <div class="or-bench-tile">
                        <div class="or-bench-label">ROC-AUC</div>
                        <div class="or-bench-val">${m.benchmarks.roc_auc}%</div>
                    </div>
                    <div class="or-bench-tile">
                        <div class="or-bench-label">Latency (CPU)</div>
                        <div class="or-bench-val">${m.benchmarks.latency_cpu_ms} ms</div>
                    </div>
                </div>

                <!-- Parameters Block -->
                <div class="or-block">
                    <div class="or-block-header">Parameters & Dimensions</div>
                    <div class="or-spec-list">
                        <div class="or-spec-item">
                            <span class="or-spec-k">Total Parameters</span>
                            <span class="or-spec-v font-mono">${m.parameters.total}</span>
                        </div>
                        <div class="or-spec-item">
                            <span class="or-spec-k">Trainable Parameters</span>
                            <span class="or-spec-v font-mono">${m.parameters.trainable}</span>
                        </div>
                        <div class="or-spec-item">
                            <span class="or-spec-k">Feature Vector Dim</span>
                            <span class="or-spec-v">${m.parameters.feature_dim}</span>
                        </div>
                        <div class="or-spec-item">
                            <span class="or-spec-k">Temporal Context</span>
                            <span class="or-spec-v">${m.parameters.clip_length} (${m.parameters.stride} stride)</span>
                        </div>
                        <div class="or-spec-item">
                            <span class="or-spec-k">Transformer Architecture</span>
                            <span class="or-spec-v">${m.parameters.transformer_layers} Layers, ${m.parameters.attention_heads} Heads (dim ${m.parameters.hidden_dim})</span>
                        </div>
                    </div>
                </div>

                <!-- Architecture & Objective Block -->
                <div class="or-block">
                    <div class="or-block-header">Architecture Components</div>
                    <div class="or-spec-list">
                        <div class="or-spec-item">
                            <span class="or-spec-k">Visual Encoder</span>
                            <span class="or-spec-v">${m.architecture.visual_encoder}</span>
                        </div>
                        <div class="or-spec-item">
                            <span class="or-spec-k">Pose Dynamics</span>
                            <span class="or-spec-v">${m.architecture.pose_encoder}</span>
                        </div>
                        <div class="or-spec-item">
                            <span class="or-spec-k">Loss Formulation</span>
                            <span class="or-spec-v">${m.architecture.loss}</span>
                        </div>
                    </div>
                </div>

                <!-- Training & Dataset Block -->
                <div class="or-block">
                    <div class="or-block-header">Training & Hardware</div>
                    <div class="or-spec-list">
                        <div class="or-spec-item">
                            <span class="or-spec-k">Datasets</span>
                            <span class="or-spec-v">${m.training.dataset}</span>
                        </div>
                        <div class="or-spec-item">
                            <span class="or-spec-k">Hardware</span>
                            <span class="or-spec-v">${m.training.hardware}</span>
                        </div>
                        <div class="or-spec-item">
                            <span class="or-spec-k">Training Runtime</span>
                            <span class="or-spec-v">${m.training.training_time}</span>
                        </div>
                    </div>
                </div>

                <!-- Event Classes -->
                <div class="or-block">
                    <div class="or-block-header">Recognized Event Classes (${(m.classes || []).length})</div>
                    <div class="class-pill-cloud">
                        ${classesBadges}
                    </div>
                </div>

                <!-- Architectural Advantages -->
                <div class="or-block">
                    <div class="or-block-header">Key Architectural Highlights</div>
                    <ul class="or-pros-list">
                        ${prosList}
                    </ul>
                </div>
            `;

            // Wire "Use Model" button
            const selectBtn = card.querySelector('.or-select-btn');
            selectBtn.addEventListener('click', () => {
                selectModel(m.id, m.name);
                closeComparePage();
            });

            compareCardsGrid.appendChild(card);
        });
    }

    function renderBenchmarkChart(models) {
        // Build crisp, modern SVG Bar Chart
        const metrics = [
            { label: 'Video Accuracy (%)', key: 'accuracy', max: 100 },
            { label: 'Macro F1 (%)', key: 'macro_f1', max: 100 },
            { label: 'Anomaly ROC-AUC (%)', key: 'roc_auc', max: 100 },
            { label: 'Latency (ms/frame)', key: 'latency_cpu_ms', max: 30, lowerIsBetter: true },
        ];

        // Legend
        compareChartLegend.innerHTML = models.map(m => `
            <div class="chart-legend-item">
                <span class="legend-color-box" style="background-color: ${MODEL_THEME_COLORS[m.key] || '#888'};"></span>
                <span class="legend-name">${escapeHtml(m.name)}</span>
            </div>
        `).join('');

        const containerWidth = (compareChartBox && compareChartBox.clientWidth > 100) ? compareChartBox.clientWidth : 860;
        const width = Math.max(760, containerWidth);
        const height = 185;
        const paddingLeft = 148;
        const paddingRight = 65;
        const paddingTop = 16;
        const paddingBottom = 18;
        const plotWidth = width - paddingLeft - paddingRight;
        const plotHeight = height - paddingTop - paddingBottom;

        const groupHeight = plotHeight / metrics.length;
        const barSpacing = 3;
        const barHeight = Math.max(5, Math.min(14, (groupHeight - 12) / models.length));

        let svgContent = `
            <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMinYMid meet" class="clean-benchmark-svg" xmlns="http://www.w3.org/2000/svg">
                <g class="grid-lines">
                    <line x1="${paddingLeft}" y1="${paddingTop}" x2="${paddingLeft}" y2="${height - paddingBottom}" stroke="#2d2d2d" stroke-width="1"></line>
                    <line x1="${paddingLeft + plotWidth * 0.25}" y1="${paddingTop}" x2="${paddingLeft + plotWidth * 0.25}" y2="${height - paddingBottom}" stroke="#252525" stroke-width="1" stroke-dasharray="3,4"></line>
                    <line x1="${paddingLeft + plotWidth * 0.5}" y1="${paddingTop}" x2="${paddingLeft + plotWidth * 0.5}" y2="${height - paddingBottom}" stroke="#252525" stroke-width="1" stroke-dasharray="3,4"></line>
                    <line x1="${paddingLeft + plotWidth * 0.75}" y1="${paddingTop}" x2="${paddingLeft + plotWidth * 0.75}" y2="${height - paddingBottom}" stroke="#252525" stroke-width="1" stroke-dasharray="3,4"></line>
                    <line x1="${paddingLeft + plotWidth}" y1="${paddingTop}" x2="${paddingLeft + plotWidth}" y2="${height - paddingBottom}" stroke="#2d2d2d" stroke-width="1"></line>
                </g>
        `;

        metrics.forEach((metric, mIdx) => {
            const groupY = paddingTop + mIdx * groupHeight;
            const midY = groupY + groupHeight / 2 + 3.5;

            svgContent += `
                <text x="${paddingLeft - 12}" y="${midY}" class="svg-axis-label" text-anchor="end">${metric.label}</text>
            `;

            models.forEach((m, modelIdx) => {
                const val = m.benchmarks[metric.key] || 0;
                const ratio = Math.min(1, Math.max(0, val / metric.max));
                const barWidth = Math.max(2, ratio * plotWidth);
                const barY = groupY + (groupHeight - models.length * (barHeight + barSpacing) + barSpacing) / 2 + modelIdx * (barHeight + barSpacing);
                const color = MODEL_THEME_COLORS[m.key] || '#888';

                svgContent += `
                    <g class="svg-bar-group">
                        <rect x="${paddingLeft}" y="${barY}" width="${barWidth}" height="${barHeight}" rx="2.5" fill="${color}" opacity="0.88">
                            <title>${m.name}: ${val}</title>
                        </rect>
                        <text x="${paddingLeft + barWidth + 7}" y="${barY + barHeight - 2}" class="svg-bar-value font-mono">
                            ${val}${metric.key === 'latency_cpu_ms' ? 'ms' : '%'}
                        </text>
                    </g>
                `;
            });

            if (mIdx < metrics.length - 1) {
                svgContent += `
                    <line x1="${paddingLeft}" y1="${groupY + groupHeight}" x2="${paddingLeft + plotWidth}" y2="${groupY + groupHeight}" stroke="#222" stroke-width="1"></line>
                `;
            }
        });

        svgContent += `</svg>`;
        compareChartBox.innerHTML = svgContent;
    }

    let compareResizeTimer = null;
    window.addEventListener('resize', () => {
        if (comparePage && comparePage.classList.contains('open') && comparisonData) {
            clearTimeout(compareResizeTimer);
            compareResizeTimer = setTimeout(() => {
                renderComparisonUI();
            }, 80);
        }
    });

    function renderSingleModelDeepdive(model) {
        let perClassRows = '';
        const perClass = model.per_class_f1 || {};

        Object.entries(perClass).forEach(([cls, f1]) => {
            perClassRows += `
                <div class="deepdive-f1-row">
                    <span class="deepdive-cls">${escapeHtml(cls)}</span>
                    <div class="deepdive-track">
                        <div class="deepdive-fill" style="width: ${f1}%; background-color: ${MODEL_THEME_COLORS[model.key] || '#10a37f'};"></div>
                    </div>
                    <span class="deepdive-val font-mono">${f1}%</span>
                </div>
            `;
        });

        singleModelDeepdive.innerHTML = `
            <div class="deepdive-container">
                <div class="deepdive-header">
                    <h4 class="deepdive-title">Single Model Deep-Dive: ${escapeHtml(model.name)} Specification Sheet</h4>
                    <span class="deepdive-badge">${escapeHtml(model.badge)} · ${escapeHtml(model.id)}</span>
                </div>
                
                <div class="deepdive-grid">
                    <!-- Per Class F1 -->
                    <div class="deepdive-box">
                        <div class="deepdive-box-title">Per-Class F1 Score Breakdown (Test Set)</div>
                        <div class="deepdive-f1-list">
                            ${perClassRows}
                        </div>
                    </div>

                    <!-- Detailed Neural Architecture Matrix -->
                    <div class="deepdive-box">
                        <div class="deepdive-box-title">Neural Parameter & Tensor Specifications</div>
                        <table class="deepdive-table">
                            <tbody>
                                <tr>
                                    <td>Total Parameter Count</td>
                                    <td class="font-mono text-emerald">${model.parameters.total}</td>
                                </tr>
                                <tr>
                                    <td>Trainable Head / Aggregator</td>
                                    <td class="font-mono">${model.parameters.trainable}</td>
                                </tr>
                                <tr>
                                    <td>Backbone Parameters (Frozen)</td>
                                    <td class="font-mono">${model.parameters.backbone_frozen}</td>
                                </tr>
                                <tr>
                                    <td>Input Frame Window</td>
                                    <td class="font-mono">${model.parameters.clip_length}</td>
                                </tr>
                                <tr>
                                    <td>Sliding Stride</td>
                                    <td class="font-mono">${model.parameters.stride}</td>
                                </tr>
                                <tr>
                                    <td>Transformer Self-Attention</td>
                                    <td class="font-mono">${model.parameters.transformer_layers} Layers, ${model.parameters.attention_heads} Heads (dim ${model.parameters.hidden_dim})</td>
                                </tr>
                                <tr>
                                    <td>Feature Embedding Vector</td>
                                    <td class="font-mono">${model.parameters.feature_dim}</td>
                                </tr>
                                <tr>
                                    <td>Throughput (GPU L40S)</td>
                                    <td class="font-mono">${model.benchmarks.fps_gpu} FPS (~7ms/frame)</td>
                                </tr>
                                <tr>
                                    <td>Throughput (CPU)</td>
                                    <td class="font-mono">${model.benchmarks.fps_cpu} FPS (~${model.benchmarks.latency_cpu_ms}ms/frame)</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        `;
    }

    // ── 7. ChatGPT-Style Chat History Drawer ──────────────────────────────────
    let activeChatId = null;

    function openDrawer() {
        if (!chatHistoryDrawer) return;
        chatHistoryDrawer.classList.add('open');
        if (drawerBackdrop) drawerBackdrop.classList.add('open');
    }

    function closeDrawer() {
        if (!chatHistoryDrawer) return;
        chatHistoryDrawer.classList.remove('open');
        if (drawerBackdrop) drawerBackdrop.classList.remove('open');
    }

    function toggleDrawer() {
        if (!chatHistoryDrawer) return;
        if (chatHistoryDrawer.classList.contains('open')) {
            closeDrawer();
        } else {
            openDrawer();
        }
    }

    if (sidebarToggleBtn) sidebarToggleBtn.addEventListener('click', toggleDrawer);
    if (drawerCloseBtn) drawerCloseBtn.addEventListener('click', closeDrawer);
    if (drawerBackdrop) drawerBackdrop.addEventListener('click', closeDrawer);

    // Ctrl+B shortcut to toggle drawer, Escape to close
    document.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && (e.key === 'b' || e.key === 'B')) {
            e.preventDefault();
            toggleDrawer();
        } else if (e.key === 'Escape' && chatHistoryDrawer && chatHistoryDrawer.classList.contains('open')) {
            closeDrawer();
        }
    });

    const DEFAULT_HISTORY = [
        {
            id: 'mock_1',
            title: 'Normal Walking Baseline',
            model: 'Orion',
            predClass: 'Normal',
            isAnomaly: false,
            conf: 93.4,
            timestamp: Date.now() - 3600000,
            timeAgo: '1h ago',
            data: {
                video_url: '/static/samples/Normal_Videos_003_x264.mp4',
                is_anomaly: false,
                prediction: 'Normal',
                confidence: 93.4,
                anomaly_score: 6.6,
                top_window: { start: 1.2, end: 4.8, start_formatted: '00:01', end_formatted: '00:04' },
                video_info: { duration: 8.5, duration_formatted: '00:08', total_frames: 255 },
                probabilities: { Normal: 93.4, Shoplifting: 3.2, Burglary: 2.1, Fighting: 1.3 },
                summary: '**Normal Activity Identified** · Verified non-anomalous standard pedestrian movement with 93.4% model confidence.'
            }
        },
        {
            id: 'mock_2',
            title: 'Perimeter Intrusion Feed',
            model: 'Orion',
            predClass: 'Burglary',
            isAnomaly: true,
            conf: 96.2,
            timestamp: Date.now() - 7200000,
            timeAgo: '2h ago',
            data: {
                video_url: '/static/samples/Burglary001_x264.mp4',
                is_anomaly: true,
                prediction: 'Burglary',
                confidence: 96.2,
                anomaly_score: 96.2,
                top_window: { start: 2.0, end: 6.5, start_formatted: '00:02', end_formatted: '00:06' },
                video_info: { duration: 11.2, duration_formatted: '00:11', total_frames: 336 },
                probabilities: { Burglary: 96.2, Shoplifting: 2.4, Vandalism: 1.1, Normal: 0.3 },
                summary: '**Security Anomaly Detected** · High-probability burglary event flagged in perimeter zone with 96.2% confidence.'
            }
        },
        {
            id: 'mock_3',
            title: 'Facility Sweep Verification',
            model: 'Vesper',
            predClass: 'Normal',
            isAnomaly: false,
            conf: 91.8,
            timestamp: Date.now() - 86400000,
            timeAgo: 'Yesterday',
            data: {
                video_url: '/static/samples/Normal_Videos_006_x264.mp4',
                is_anomaly: false,
                prediction: 'Normal',
                confidence: 91.8,
                anomaly_score: 8.2,
                top_window: { start: 0.5, end: 3.5, start_formatted: '00:00', end_formatted: '00:03' },
                video_info: { duration: 6.8, duration_formatted: '00:06', total_frames: 204 },
                probabilities: { Normal: 91.8, Fighting: 4.1, Theft: 2.7, Burglary: 1.4 },
                summary: '**Normal Operation Verified** · All observed intervals within standard baseline parameters (91.8%).'
            }
        }
    ];

    function getChatHistory() {
        try {
            const raw = localStorage.getItem('aegis_chat_history');
            if (raw) {
                const parsed = JSON.parse(raw);
                if (Array.isArray(parsed) && parsed.length > 0) return parsed;
            }
        } catch (_) {}
        return DEFAULT_HISTORY;
    }

    function saveChatHistory(history) {
        try {
            localStorage.setItem('aegis_chat_history', JSON.stringify(history));
        } catch (_) {}
    }

    function addChatToHistory(title, modelName, data) {
        const history = getChatHistory();
        const newChat = {
            id: `chat_${Date.now()}`,
            title: title || 'Video Analysis',
            model: modelName || currentModel,
            predClass: data.prediction || 'Normal',
            isAnomaly: !!data.is_anomaly,
            conf: data.confidence || 0,
            timestamp: Date.now(),
            timeAgo: 'Just now',
            data: data
        };
        history.unshift(newChat);
        saveChatHistory(history.slice(0, 30));
        activeChatId = newChat.id;
        renderChatHistory();
    }

    function renderChatHistory() {
        if (!drawerChatList) return;
        const history = getChatHistory();
        drawerChatList.innerHTML = '';

        if (history.length === 0) {
            drawerChatList.innerHTML = `
                <div style="padding: 20px 10px; font-size: 12px; color: #666; text-align: center;">
                    No past analyses found.<br>Attach a video to begin.
                </div>
            `;
            return;
        }

        history.forEach((chat) => {
            const item = document.createElement('div');
            item.className = `drawer-chat-item ${chat.id === activeChatId ? 'active' : ''}`;
            item.setAttribute('data-id', chat.id);

            const dotClass = chat.isAnomaly ? 'dci-status-dot danger' : 'dci-status-dot';
            const statusLabel = chat.isAnomaly ? 'Anomaly' : 'Normal';

            item.innerHTML = `
                <svg class="dci-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polygon points="23 7 16 12 23 17 23 7"></polygon>
                    <rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect>
                </svg>
                <div class="dci-info">
                    <div class="dci-title">${escapeHtml(chat.title)}</div>
                    <div class="dci-meta">
                        <span class="${dotClass}"></span>
                        <span>${escapeHtml(statusLabel)} ${chat.conf}% · ${escapeHtml(chat.timeAgo || 'Recent')}</span>
                    </div>
                </div>
                <button class="dci-delete" type="button" title="Delete chat">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="6" x2="6" y2="18"></line>
                        <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                </button>
            `;

            // Click to load chat session
            item.addEventListener('click', (e) => {
                if (e.target.closest('.dci-delete')) return;
                loadChatSession(chat);
                closeDrawer();
            });

            // Delete item
            const deleteBtn = item.querySelector('.dci-delete');
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                deleteChatSession(chat.id);
            });

            drawerChatList.appendChild(item);
        });
    }

    function deleteChatSession(id) {
        let history = getChatHistory();
        history = history.filter(c => c.id !== id);
        saveChatHistory(history);
        if (activeChatId === id) {
            startNewAnalysis();
        }
        renderChatHistory();
    }

    function clearAllChatHistory() {
        if (confirm('Clear all analysis history from the drawer?')) {
            saveChatHistory([]);
            startNewAnalysis();
            renderChatHistory();
        }
    }

    function loadChatSession(chat) {
        activeChatId = chat.id;
        renderChatHistory();

        if (heroContainer) heroContainer.style.display = 'none';
        messagesContainer.innerHTML = '';

        // Add user bubble
        appendUserMessage(chat.title, 'Historical Record', chat.model);

        // Add assistant result card
        const assistantRow = document.createElement('div');
        assistantRow.className = 'message-row assistant';
        assistantRow.innerHTML = `
            <div class="assistant-avatar">
                <svg width="18" height="18" viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="16" cy="16" r="13"></circle>
                    <circle cx="16" cy="16" r="6"></circle>
                    <path d="M16 3v5M16 24v5M3 16h5M24 16h5"></path>
                </svg>
            </div>
            <div class="assistant-content"></div>
        `;
        messagesContainer.appendChild(assistantRow);

        renderAnalysisResult(assistantRow, chat.data);
        scrollToBottom();
    }

    function startNewAnalysis() {
        activeChatId = null;
        messagesContainer.innerHTML = '';
        if (heroContainer) heroContainer.style.display = 'flex';
        clearAttachment();
        renderChatHistory();
    }

    if (drawerNewChatBtn) drawerNewChatBtn.addEventListener('click', () => {
        startNewAnalysis();
        closeDrawer();
    });

    if (resetBtn) resetBtn.addEventListener('click', startNewAnalysis);
    if (drawerClearBtn) drawerClearBtn.addEventListener('click', clearAllChatHistory);

    function scrollToBottom() {
        chatMain.scrollTop = chatMain.scrollHeight;
    }

    function escapeHtml(str) {
        if (!str) return '';
        return str.replace(/&/g, '&amp;')
                  .replace(/</g, '&lt;')
                  .replace(/>/g, '&gt;')
                  .replace(/"/g, '&quot;')
                  .replace(/'/g, '&#039;');
    }

    function formatMarkdown(str) {
        if (!str) return '';
        let formatted = escapeHtml(str);
        // Bold **text**
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        return formatted;
    }

    // ==============================================================================
    // 7. LIVE CAMERA SURVEILLANCE & CHATGPT-STYLE BREATHING MORPH CONTROLLER
    // ==============================================================================
    const liveBtn = document.getElementById('liveBtn');
    const liveOverlay = document.getElementById('liveOverlay');
    const liveBackdrop = document.getElementById('liveBackdrop');
    const liveMorphContainer = document.getElementById('liveMorphContainer');
    const liveBreathingStatus = document.getElementById('liveBreathingStatus');
    const liveGrantPermBtn = document.getElementById('liveGrantPermBtn');
    const livePermCloseBtn = document.getElementById('livePermCloseBtn');
    const liveErrorCloseBtn = document.getElementById('liveErrorCloseBtn');
    const liveRetryBtn = document.getElementById('liveRetryBtn');
    const liveErrorMessage = document.getElementById('liveErrorMessage');
    const liveActiveCloseBtn = document.getElementById('liveActiveCloseBtn');
    const liveVideo = document.getElementById('liveVideo');
    const liveCaptureCanvas = document.getElementById('liveCaptureCanvas');
    const liveHudModel = document.getElementById('liveHudModel');
    const liveHudFps = document.getElementById('liveHudFps');
    const liveHudClass = document.getElementById('liveHudClass');
    const liveHudAnomaly = document.getElementById('liveHudAnomaly');
    const liveHudConfidence = document.getElementById('liveHudConfidence');
    const liveHudLatency = document.getElementById('liveHudLatency');
    const liveCamSwitchBtn = document.getElementById('liveCamSwitchBtn');
    const liveCamLabel = document.getElementById('liveCamLabel');
    const livePermDesc = document.getElementById('livePermDesc');
    const liveCamDropdownWrap = document.getElementById('liveCamDropdownWrap');
    const liveCamMenu = document.getElementById('liveCamMenu');
    const liveCamMenuList = document.getElementById('liveCamMenuList');

    let liveStream = null;
    let liveIntervalId = null;
    let liveSessionId = null;
    let liveFrameCount = 0;
    let liveLastFpsTime = performance.now();
    let liveCurrentFps = 0;
    let liveVideoDevices = [];
    let liveSelectedDeviceIndex = 0;
    let isLiveActive = false;
    let isProcessingFrame = false;

    function setMorphState(stateName) {
        if (!liveMorphContainer) return;
        liveMorphContainer.classList.remove('state-breathing', 'state-permission', 'state-error', 'state-active');
        liveMorphContainer.classList.add(`state-${stateName}`);
    }

    async function openLiveOverlay() {
        if (!liveOverlay) return;
        liveSessionId = 'live_' + Math.random().toString(36).substring(2, 9);
        if (livePermDesc) {
            livePermDesc.textContent = 'AEGIS requires camera access to process real-time surveillance video.';
        }
        liveOverlay.classList.add('active');
        liveOverlay.setAttribute('aria-hidden', 'false');
        setMorphState('breathing');
        if (liveBreathingStatus) {
            liveBreathingStatus.textContent = `Warming up ${chatboxModelName ? chatboxModelName.textContent : 'model'}...`;
        }

        // 1. Warm up the model in background
        try {
            await fetch('/api/live_init', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: currentModel })
            });
        } catch (e) {
            console.warn('Model warm-up error (proceeding anyway):', e);
        }

        // 2. Check camera permission
        checkCameraPermissionAndStart();
    }

    async function checkCameraPermissionAndStart() {
        setMorphState('breathing');
        if (liveBreathingStatus) liveBreathingStatus.textContent = 'Verifying camera access...';

        try {
            // Check permissions API if supported
            if (navigator.permissions && navigator.permissions.query) {
                try {
                    const status = await navigator.permissions.query({ name: 'camera' });
                    if (status.state === 'granted') {
                        await startCameraStream();
                        return;
                    } else if (status.state === 'prompt') {
                        setTimeout(() => {
                            setMorphState('permission');
                        }, 500);
                        return;
                    }
                } catch (permErr) {
                    // Fall back to device enumeration check
                }
            }

            // Fallback: enumerate devices
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoInputs = devices.filter(d => d.kind === 'videoinput');
            liveVideoDevices = videoInputs;

            // If devices have descriptive labels, permission was already granted!
            const hasLabels = videoInputs.some(d => d.label && d.label.length > 0);
            if (hasLabels) {
                await startCameraStream();
            } else {
                setTimeout(() => {
                    setMorphState('permission');
                }, 500);
            }
        } catch (err) {
            setTimeout(() => {
                setMorphState('permission');
            }, 500);
        }
    }

    async function requestCameraAccess() {
        // Morphs to breathing circle while requesting permission
        setMorphState('breathing');
        if (liveBreathingStatus) liveBreathingStatus.textContent = 'Connecting camera...';

        try {
            await startCameraStream();
        } catch (err) {
            console.warn('Camera access error:', err);
            // Return to permission card and guide user without getting stuck
            setTimeout(() => {
                setMorphState('permission');
                if (livePermDesc) {
                    if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
                        livePermDesc.textContent = 'Camera permission was denied. If blocked, click the camera icon in your browser address bar to allow, then click Allow Camera again.';
                    } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
                        livePermDesc.textContent = 'No camera found. Connect a webcam or phone camera, then click Allow Camera.';
                    } else {
                        livePermDesc.textContent = err.message || 'Camera permission was not granted. Click Allow Camera to try again.';
                    }
                }
            }, 350);
        }
    }

    async function updateAvailableCameras() {
        try {
            const allDevices = await navigator.mediaDevices.enumerateDevices();
            liveVideoDevices = allDevices.filter(d => d.kind === 'videoinput');
            renderLiveCamMenu();
        } catch (e) {
            console.warn('Failed to enumerate cameras:', e);
        }
    }

    function renderLiveCamMenu() {
        if (!liveCamMenuList) return;
        liveCamMenuList.innerHTML = '';
        if (!liveVideoDevices.length) {
            liveCamMenuList.innerHTML = '<div style="font-size:11px;color:#71717a;padding:6px 8px;">No other cameras found</div>';
            return;
        }

        const currentTrack = liveStream ? liveStream.getVideoTracks()[0] : null;
        const currentDeviceId = currentTrack && currentTrack.getSettings ? currentTrack.getSettings().deviceId : null;

        liveVideoDevices.forEach((dev, idx) => {
            const label = dev.label || `Camera ${idx + 1}`;
            const isActive = (currentDeviceId && dev.deviceId === currentDeviceId) || (!currentDeviceId && idx === 0);

            const itemBtn = document.createElement('button');
            itemBtn.type = 'button';
            itemBtn.className = `live-cam-menu-item ${isActive ? 'active' : ''}`;
            itemBtn.innerHTML = `
                <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:180px;">${label}</span>
                ${isActive ? '<svg class="live-cam-menu-item-check" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>' : ''}
            `;
            itemBtn.addEventListener('click', async (e) => {
                e.stopPropagation();
                closeCamMenu();
                if (liveCamLabel) liveCamLabel.textContent = label;
                try {
                    await startCameraStream(dev.deviceId);
                } catch (err) {
                    console.error('Failed to switch camera:', err);
                }
            });
            liveCamMenuList.appendChild(itemBtn);
        });
    }

    function toggleCamMenu(e) {
        if (e) e.stopPropagation();
        if (!liveCamDropdownWrap) return;
        if (liveCamDropdownWrap.classList.contains('open')) {
            closeCamMenu();
        } else {
            openCamMenu();
        }
    }

    function openCamMenu() {
        if (!liveCamDropdownWrap) return;
        renderLiveCamMenu();
        liveCamDropdownWrap.classList.add('open');
        if (liveCamSwitchBtn) liveCamSwitchBtn.setAttribute('aria-expanded', 'true');
    }

    function closeCamMenu() {
        if (!liveCamDropdownWrap) return;
        liveCamDropdownWrap.classList.remove('open');
        if (liveCamSwitchBtn) liveCamSwitchBtn.setAttribute('aria-expanded', 'false');
    }

    async function startCameraStream(deviceId = null) {
        if (liveStream) {
            liveStream.getTracks().forEach(track => track.stop());
            liveStream = null;
        }

        const constraints = {
            video: deviceId 
                ? { deviceId: { exact: deviceId }, width: { ideal: 640 }, height: { ideal: 480 } }
                : { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
            audio: false
        };

        liveStream = await navigator.mediaDevices.getUserMedia(constraints);
        if (liveVideo) {
            liveVideo.srcObject = liveStream;
            await liveVideo.play();
        }

        const track = liveStream.getVideoTracks()[0];
        if (liveCamLabel && track) {
            liveCamLabel.textContent = track.label || 'Live Camera';
        }

        await updateAvailableCameras();

        // System verified ready -> Circle expands smoothly to fullscreen active mode!
        setTimeout(() => {
            isLiveActive = true;
            setMorphState('active');
            updateHudModelName();
            startLiveDetectionLoop();
        }, 320);
    }

    function showLiveError(msg) {
        if (liveErrorMessage) liveErrorMessage.textContent = msg;
        setMorphState('error');
    }

    function updateHudModelName() {
        if (!liveHudModel) return;
        const activeName = chatboxModelName ? chatboxModelName.textContent : 'Orion';
        liveHudModel.textContent = activeName;
    }

    function startLiveDetectionLoop() {
        if (liveIntervalId) {
            clearTimeout(liveIntervalId);
            liveIntervalId = null;
        }
        liveFrameCount = 0;
        liveLastFpsTime = performance.now();

        async function liveLoop() {
            if (!isLiveActive || document.hidden) return;
            await captureAndDetectFrame();
            if (isLiveActive && !document.hidden) {
                liveIntervalId = setTimeout(liveLoop, 280);
            }
        }
        liveLoop();
    }

    async function captureAndDetectFrame() {
        if (!isLiveActive || isProcessingFrame || !liveVideo || !liveCaptureCanvas || document.hidden) return;
        if (liveVideo.readyState < 2 || liveVideo.videoWidth === 0) return;

        isProcessingFrame = true;
        try {
            const width = 320;
            const height = 240;
            liveCaptureCanvas.width = width;
            liveCaptureCanvas.height = height;
            const ctx = liveCaptureCanvas.getContext('2d');
            ctx.drawImage(liveVideo, 0, 0, width, height);

            const base64Data = liveCaptureCanvas.toDataURL('image/jpeg', 0.45);

            const res = await fetch('/api/live_detect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: liveSessionId,
                    model: currentModel,
                    image: base64Data
                })
            });

            if (res.ok) {
                const result = await res.json();
                if (result.status === 'success') {
                    updateLiveHud(result);
                }
            }

            // Compute FPS
            liveFrameCount++;
            const now = performance.now();
            const elapsed = now - liveLastFpsTime;
            if (elapsed >= 1000) {
                liveCurrentFps = Math.round((liveFrameCount / (elapsed / 1000)) * 10) / 10;
                if (liveHudFps) liveHudFps.textContent = `${liveCurrentFps.toFixed(1)} FPS`;
                liveFrameCount = 0;
                liveLastFpsTime = now;
            }
        } catch (err) {
            console.warn('Live detect error:', err);
        } finally {
            isProcessingFrame = false;
        }
    }

    function updateLiveHud(data) {
        if (liveHudClass) {
            liveHudClass.textContent = data.prediction || 'NORMAL';
            if (data.is_anomaly) {
                liveHudClass.classList.add('is-anomaly');
            } else {
                liveHudClass.classList.remove('is-anomaly');
            }
        }
        if (liveHudAnomaly) {
            liveHudAnomaly.textContent = `${data.anomaly_score !== undefined ? data.anomaly_score : 0}%`;
            if (data.anomaly_score >= 45) {
                liveHudAnomaly.style.color = '#f87171';
            } else {
                liveHudAnomaly.style.color = '#e4e4e7';
            }
        }
        if (liveHudConfidence) {
            liveHudConfidence.textContent = `${data.confidence !== undefined ? data.confidence : '--'}%`;
        }
        if (liveHudLatency) {
            liveHudLatency.textContent = `${data.latency_ms || '--'} ms`;
        }
    }

    function closeLiveOverlay() {
        isLiveActive = false;
        if (liveIntervalId) {
            clearTimeout(liveIntervalId);
            liveIntervalId = null;
        }
        if (liveStream) {
            liveStream.getTracks().forEach(t => t.stop());
            liveStream = null;
        }
        if (liveVideo) {
            liveVideo.srcObject = null;
        }
        // Smooth morph back to breathing circle then fade out overlay
        setMorphState('breathing');
        setTimeout(() => {
            if (liveOverlay) {
                liveOverlay.classList.remove('active');
                liveOverlay.setAttribute('aria-hidden', 'true');
            }
        }, 280);
    }

    // Switch camera / open camera selection dropdown
    if (liveCamSwitchBtn) {
        liveCamSwitchBtn.addEventListener('click', toggleCamMenu);
    }

    document.addEventListener('click', (e) => {
        if (liveCamDropdownWrap && !liveCamDropdownWrap.contains(e.target)) {
            closeCamMenu();
        }
    });

    document.addEventListener('visibilitychange', () => {
        if (!document.hidden && isLiveActive && !liveIntervalId) {
            startLiveDetectionLoop();
        }
    });

    if (liveBtn) liveBtn.addEventListener('click', openLiveOverlay);
    if (liveBackdrop) liveBackdrop.addEventListener('click', closeLiveOverlay);
    if (liveActiveCloseBtn) liveActiveCloseBtn.addEventListener('click', closeLiveOverlay);
    if (livePermCloseBtn) livePermCloseBtn.addEventListener('click', closeLiveOverlay);
    if (liveErrorCloseBtn) liveErrorCloseBtn.addEventListener('click', closeLiveOverlay);
    if (liveGrantPermBtn) liveGrantPermBtn.addEventListener('click', requestCameraAccess);
    if (liveRetryBtn) liveRetryBtn.addEventListener('click', checkCameraPermissionAndStart);

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && liveOverlay && liveOverlay.classList.contains('active')) {
            closeLiveOverlay();
        }
    });

    // Init
    fetchModels();
    fetchSamples();
    renderChatHistory();
});

