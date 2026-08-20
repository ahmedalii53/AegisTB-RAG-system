/**
 * AegisTB — Frontend Application Logic
 * Integrates Claude-style Centered Hero view, dynamic Split-Screen Workspace,
 * Saved Consultations (Chat History Management with Rename/Delete),
 * Info / About Modal, Dark/Light Mode, and Sentence-Level PDF Text Highlighting.
 */

// Configure PDF.js Worker
if (window.pdfjsLib) {
  pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
}

let currentPdfDoc = null;
let currentPdfFile = null;
let currentPageNum = 1;
let currentScale = 1.25;
let currentCitationHighlight = null;
let isPdfPanelOpen = false;

// Session Management State
let activeSessionId = null;
let currentSessionMessages = [];

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  checkSystemHealth();
  initHistorySessions();
});

// === Theme Toggle Logic ===
function initTheme() {
  const savedTheme = localStorage.getItem('aegistb_theme') || 'dark';
  document.documentElement.setAttribute('data-theme', savedTheme);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme') || 'dark';
  const newTheme = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', newTheme);
  localStorage.setItem('aegistb_theme', newTheme);
}

// === System Health Check ===
async function checkSystemHealth() {
  const statusElem = document.getElementById('systemStatus');
  const dotElem = statusElem.querySelector('.status-dot');
  const textElem = statusElem.querySelector('.status-text');

  try {
    const res = await fetch('/health');
    if (res.ok) {
      const data = await res.json();
      dotElem.classList.add('online');
      textElem.textContent = `Online (${data.total_indexed_chunks || 1339} chunks)`;
    } else {
      textElem.textContent = 'API Error';
    }
  } catch (err) {
    textElem.textContent = 'Offline';
    console.error('Health check failed:', err);
  }
}

// === Info / About Modal Logic ===

function openInfoModal() {
  const modal = document.getElementById('infoModalOverlay');
  if (modal) modal.classList.add('show');
}

function closeInfoModal(event) {
  const modal = document.getElementById('infoModalOverlay');
  if (modal) modal.classList.remove('show');
}

// === Chat Sessions / History Management ===

function getStoredSessions() {
  try {
    const raw = localStorage.getItem('aegistb_sessions');
    return raw ? JSON.parse(raw) : [];
  } catch (e) {
    console.error('Failed to parse sessions from localStorage', e);
    return [];
  }
}

function setStoredSessions(sessions) {
  try {
    localStorage.setItem('aegistb_sessions', JSON.stringify(sessions));
    updateHistoryBadge();
  } catch (e) {
    console.error('Failed to save sessions to localStorage', e);
  }
}

function initHistorySessions() {
  renderSessionList();
  updateHistoryBadge();
}

function updateHistoryBadge() {
  const sessions = getStoredSessions();
  const badge = document.getElementById('historyBadgeCount');
  if (badge) {
    badge.textContent = sessions.length;
    badge.style.display = sessions.length > 0 ? 'inline-block' : 'none';
  }
}

function toggleHistorySidebar(forceState) {
  const sidebar = document.getElementById('historySidebar');
  const backdrop = document.getElementById('sidebarBackdrop');
  if (!sidebar) return;

  const isOpen = sidebar.classList.contains('open');
  const target = typeof forceState === 'boolean' ? forceState : !isOpen;

  if (target) {
    sidebar.classList.add('open');
    if (backdrop) backdrop.classList.add('show');
    renderSessionList();
  } else {
    sidebar.classList.remove('open');
    if (backdrop) backdrop.classList.remove('show');
  }
}

function renderSessionList() {
  const listElem = document.getElementById('historyList');
  if (!listElem) return;

  const sessions = getStoredSessions();
  if (sessions.length === 0) {
    listElem.innerHTML = '<div class="history-empty">No saved consultations yet.<br>Ask a clinical question to save!</div>';
    return;
  }

  listElem.innerHTML = sessions.map(s => {
    const isActive = s.id === activeSessionId ? 'active' : '';
    const dateStr = formatRelativeDate(s.timestamp || Date.now());
    return `
      <div class="history-item-card ${isActive}" onclick="loadSession('${escapeAttr(s.id)}')">
        <div class="history-item-content">
          <span class="history-item-title" title="${escapeAttr(s.title)}">${escapeHtml(s.title)}</span>
          <span class="history-item-date">${escapeHtml(dateStr)} • ${s.messages ? s.messages.length : 0} msgs</span>
        </div>
        <div class="history-item-actions">
          <button class="history-action-btn" onclick="renameSession('${escapeAttr(s.id)}', event)" title="Rename Consultation">✏️</button>
          <button class="history-action-btn history-delete-btn" onclick="deleteSession('${escapeAttr(s.id)}', event)" title="Delete Consultation">🗑️</button>
        </div>
      </div>
    `;
  }).join('');
}

function saveCurrentQueryToSession(queryText, assistantData) {
  let sessions = getStoredSessions();
  let session = sessions.find(s => s.id === activeSessionId);

  if (!session) {
    // Generate clean concise title from queryText
    let title = queryText.length > 40 ? queryText.substring(0, 37) + '...' : queryText;
    activeSessionId = 'session_' + Date.now();
    session = {
      id: activeSessionId,
      title: title,
      timestamp: Date.now(),
      messages: [],
      activeCitation: assistantData.citations && assistantData.citations[0] ? assistantData.citations[0] : null
    };
    sessions.unshift(session);
  }

  session.messages.push({ role: 'user', content: queryText });
  session.messages.push({ role: 'assistant', data: assistantData });
  session.timestamp = Date.now();

  setStoredSessions(sessions);
  renderSessionList();
}

function loadSession(sessionId) {
  const sessions = getStoredSessions();
  const session = sessions.find(s => s.id === sessionId);
  if (!session) return;

  activeSessionId = session.id;
  currentSessionMessages = session.messages || [];

  // Switch UI to chat view
  switchToActiveChat();

  const chatHistory = document.getElementById('chatHistory');
  chatHistory.innerHTML = '';

  // Render all messages
  for (const msg of currentSessionMessages) {
    if (msg.role === 'user') {
      appendUserMessage(msg.content);
    } else if (msg.role === 'assistant' && msg.data) {
      appendAssistantMessage(msg.data);
    }
  }

  // Load citation in PDF viewer if present
  if (session.activeCitation) {
    const cit = session.activeCitation;
    loadPdfCitation(
      cit.file_name || `${cit.document}.pdf`,
      cit.page,
      cit.section,
      cit.exact_quote || '',
      cit.bbox
    );
  }

  toggleHistorySidebar(false);
  renderSessionList();
}

function renameSession(sessionId, event) {
  if (event) event.stopPropagation();
  const sessions = getStoredSessions();
  const session = sessions.find(s => s.id === sessionId);
  if (!session) return;

  const newTitle = prompt('Enter a new title for this consultation:', session.title);
  if (newTitle && newTitle.trim()) {
    session.title = newTitle.trim();
    setStoredSessions(sessions);
    renderSessionList();
  }
}

function deleteSession(sessionId, event) {
  if (event) event.stopPropagation();
  if (!confirm('Are you sure you want to delete this saved consultation?')) return;

  let sessions = getStoredSessions();
  sessions = sessions.filter(s => s.id !== sessionId);
  setStoredSessions(sessions);

  if (activeSessionId === sessionId) {
    resetToHero();
  } else {
    renderSessionList();
  }
}

function formatRelativeDate(timestamp) {
  const diff = Date.now() - timestamp;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'Just now';
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

// === Workspace State Transition (Hero <-> Active Split) ===

function switchToActiveChat() {
  const heroView = document.getElementById('heroCenterView');
  const chatHistory = document.getElementById('chatHistory');
  const stickyInputWrapper = document.getElementById('stickyInputWrapper');
  const workspace = document.getElementById('workspace');

  if (heroView) heroView.style.display = 'none';
  if (chatHistory) chatHistory.style.display = 'flex';
  if (stickyInputWrapper) stickyInputWrapper.style.display = 'block';
  
  workspace.classList.add('split-active');
  isPdfPanelOpen = true;
  updatePdfToggleBtn();
}

function resetToHero() {
  activeSessionId = null;
  currentSessionMessages = [];

  const heroView = document.getElementById('heroCenterView');
  const chatHistory = document.getElementById('chatHistory');
  const stickyInputWrapper = document.getElementById('stickyInputWrapper');
  const workspace = document.getElementById('workspace');
  const heroInput = document.getElementById('heroInput');

  if (heroView) heroView.style.display = 'flex';
  if (chatHistory) {
    chatHistory.style.display = 'none';
    chatHistory.innerHTML = '';
  }
  if (stickyInputWrapper) stickyInputWrapper.style.display = 'none';
  if (heroInput) heroInput.value = '';

  workspace.classList.remove('split-active');
  isPdfPanelOpen = false;
  updatePdfToggleBtn();

  const placeholder = document.getElementById('pdfPlaceholder');
  const pdfWrapper = document.getElementById('pdfWrapper');
  const banner = document.getElementById('citationBanner');
  if (placeholder) placeholder.style.display = 'flex';
  if (pdfWrapper) pdfWrapper.style.display = 'none';
  if (banner) banner.style.display = 'none';
  document.getElementById('currentDocTitle').textContent = 'Select a citation to inspect guideline';
  
  renderSessionList();
}

function togglePdfPanel(forceState) {
  const workspace = document.getElementById('workspace');
  if (typeof forceState === 'boolean') {
    isPdfPanelOpen = forceState;
  } else {
    isPdfPanelOpen = !isPdfPanelOpen;
  }

  if (isPdfPanelOpen) {
    workspace.classList.add('split-active');
  } else {
    workspace.classList.remove('split-active');
  }
  updatePdfToggleBtn();
}

function updatePdfToggleBtn() {
  const btn = document.getElementById('togglePdfBtn');
  if (btn) {
    if (isPdfPanelOpen) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  }
}

// === Query Submission Handlers ===

function handleHeroKeyDown(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    event.target.form.requestSubmit();
  }
}

function handleHeroSubmit(event) {
  event.preventDefault();
  const heroInput = document.getElementById('heroInput');
  const query = heroInput.value.trim();
  if (!query) return;

  switchToActiveChat();
  executeClinicalQuery(query);
}

function handleKeyDown(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    document.getElementById('chatForm').requestSubmit();
  }
}

function handleQuerySubmit(event) {
  event.preventDefault();
  const input = document.getElementById('queryInput');
  const query = input.value.trim();
  if (!query) return;

  input.value = '';
  executeClinicalQuery(query);
}

function sendPreset(queryText) {
  switchToActiveChat();
  executeClinicalQuery(queryText);
}

// === Core Pipeline API Execution ===

async function executeClinicalQuery(query) {
  const chatHistory = document.getElementById('chatHistory');
  const sendBtn = document.getElementById('sendBtn');

  // Append user message card
  appendUserMessage(query);

  // Append loading state card
  const loadingCard = appendLoadingCard();
  if (sendBtn) sendBtn.disabled = true;

  try {
    const response = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: query })
    });

    if (!response.ok) {
      throw new Error(`Server returned HTTP ${response.status}`);
    }

    const data = await response.json();
    loadingCard.remove();
    appendAssistantMessage(data);

    // Save into history sessions
    saveCurrentQueryToSession(query, data);

    // Auto-load first citation into PDF viewer
    if (data.citations && data.citations.length > 0) {
      const cit = data.citations[0];
      loadPdfCitation(
        cit.file_name || `${cit.document}.pdf`,
        cit.page,
        cit.section,
        cit.exact_quote || data.evidence,
        cit.bbox
      );
    }
  } catch (err) {
    loadingCard.remove();
    appendErrorMessage(`Failed to query evidence pipeline: ${err.message}`);
    console.error(err);
  } finally {
    if (sendBtn) sendBtn.disabled = false;
  }
}

function appendUserMessage(text) {
  const chatHistory = document.getElementById('chatHistory');
  const card = document.createElement('div');
  card.className = 'message-card user-msg';
  card.innerHTML = `<div class="user-question">${escapeHtml(text)}</div>`;
  chatHistory.appendChild(card);
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

function appendLoadingCard() {
  const chatHistory = document.getElementById('chatHistory');
  const card = document.createElement('div');
  card.className = 'message-card clinical-card';
  card.innerHTML = `
    <div style="display: flex; align-items: center; gap: 12px; padding: 6px 0;">
      <span class="loading-spinner"></span>
      <span style="color: var(--text-muted); font-size: 0.88rem; font-weight: 500;">
        Synthesizing WHO TB evidence &amp; verifying citations...
      </span>
    </div>
  `;
  chatHistory.appendChild(card);
  chatHistory.scrollTop = chatHistory.scrollHeight;
  return card;
}

function appendErrorMessage(msg) {
  const chatHistory = document.getElementById('chatHistory');
  const card = document.createElement('div');
  card.className = 'message-card clinical-card';
  card.style.borderColor = 'var(--rose-500)';
  card.innerHTML = `
    <div class="response-block">
      <div class="block-title" style="color: var(--rose-500);">System Notice</div>
      <p style="color: var(--rose-500); font-size: 0.88rem;">${escapeHtml(msg)}</p>
    </div>
  `;
  chatHistory.appendChild(card);
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

function appendAssistantMessage(data) {
  const chatHistory = document.getElementById('chatHistory');
  const card = document.createElement('div');
  card.className = 'message-card clinical-card';

  const rawConf = (data.confidence || 'medium').toLowerCase();
  let confClass = 'confidence-medium';
  let confDisplay = 'High';
  if (rawConf === 'high') {
    confClass = 'confidence-high';
    confDisplay = 'High';
  } else if (rawConf === 'low') {
    confClass = 'confidence-low';
    confDisplay = 'Low';
  } else if (rawConf === 'insufficient') {
    confClass = 'confidence-insufficient';
    confDisplay = 'Insufficient';
  }

  let citationsHtml = '';
  if (data.citations && data.citations.length > 0) {
    const citItems = data.citations.map((c, idx) => {
      const fileName = c.file_name || `${c.document}.pdf`;
      const bboxStr = JSON.stringify(c.bbox || [0,0,0,0]);
      const quoteEscaped = escapeAttr(c.exact_quote || data.evidence || '');
      const secEscaped = escapeAttr(c.section || '');
      return `
        <div class="citation-entry-card" onclick='loadPdfCitation("${escapeAttr(fileName)}", ${c.page}, "${secEscaped}", "${quoteEscaped}", ${bboxStr})' title="Click to view and highlight in PDF">
          <div class="citation-header-line">
            <span class="citation-index">[${idx + 1}]</span>
            <span class="citation-doc">${escapeHtml(c.document || 'WHO Tuberculosis Guideline')}</span>
            <span class="citation-jump-badge">View in PDF ↗</span>
          </div>
          <div class="citation-details-line">
            <span class="citation-field"><strong>Section:</strong> ${escapeHtml(c.section || 'General Guidance')}</span>
            <span class="citation-field"><strong>Page:</strong> ${c.page}</span>
          </div>
        </div>
      `;
    }).join('');

    citationsHtml = `
      <div class="response-block">
        <div class="block-title">CITATIONS</div>
        <div class="citations-container">
          ${citItems}
        </div>
      </div>
    `;
  }

  let evidenceHtml = '';
  if (data.evidence && data.evidence.trim()) {
    evidenceHtml = `
      <div class="response-block">
        <div class="block-title">EVIDENCE</div>
        <div class="evidence-text">${escapeHtml(data.evidence)}</div>
      </div>
    `;
  }

  card.innerHTML = `
    <div class="clinical-response-container">
      <div class="response-block">
        <div class="block-title">RECOMMENDATION</div>
        <div class="recommendation-text">${escapeHtml(data.recommendation)}</div>
      </div>

      ${evidenceHtml}
      ${citationsHtml}

      <div class="response-block">
        <div class="block-title">CONFIDENCE</div>
        <div class="confidence-val ${confClass}">${confDisplay}</div>
      </div>
    </div>
  `;

  chatHistory.appendChild(card);
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

// === PDF Viewer & Text-Matching Highlight Integration ===

async function loadPdfCitation(fileName, pageNumber, sectionTitle, exactQuote, bbox) {
  if (!isPdfPanelOpen) {
    togglePdfPanel(true);
  }

  const currentDocTitle = document.getElementById('currentDocTitle');
  const banner = document.getElementById('citationBanner');
  const bannerSection = document.getElementById('bannerSection');
  const bannerQuote = document.getElementById('bannerQuote');
  const placeholder = document.getElementById('pdfPlaceholder');
  const pdfWrapper = document.getElementById('pdfWrapper');

  placeholder.style.display = 'none';
  pdfWrapper.style.display = 'block';

  currentDocTitle.textContent = `${fileName} — Page ${pageNumber}`;
  bannerSection.textContent = sectionTitle || 'WHO Clinical Recommendation';
  bannerQuote.textContent = exactQuote ? `"${exactQuote}"` : 'Evidence excerpt location';
  banner.style.display = 'block';

  currentCitationHighlight = {
    bbox: bbox,
    exactQuote: exactQuote,
    sectionTitle: sectionTitle
  };

  currentPageNum = pageNumber;

  if (currentPdfDoc && currentPdfFile === fileName) {
    renderPdfPage(currentPageNum);
  } else {
    currentPdfFile = fileName;
    try {
      const pdfUrl = `/pdf/${encodeURIComponent(fileName)}`;
      const loadingTask = pdfjsLib.getDocument(pdfUrl);
      currentPdfDoc = await loadingTask.promise;
      document.getElementById('totalPagesNum').textContent = currentPdfDoc.numPages;
      renderPdfPage(currentPageNum);
    } catch (err) {
      console.error('Error loading PDF:', err);
      currentDocTitle.textContent = `Error loading ${fileName}`;
    }
  }
}

async function renderPdfPage(pageNum) {
  if (!currentPdfDoc) return;

  const canvas = document.getElementById('pdfCanvas');
  const ctx = canvas.getContext('2d');
  const highlightLayer = document.getElementById('pdfHighlightLayer');
  document.getElementById('currentPageNum').textContent = pageNum;

  try {
    const page = await currentPdfDoc.getPage(pageNum);
    const viewport = page.getViewport({ scale: currentScale });

    canvas.height = viewport.height;
    canvas.width = viewport.width;

    const renderContext = {
      canvasContext: ctx,
      viewport: viewport
    };

    await page.render(renderContext).promise;

    // Render Highlights
    highlightLayer.innerHTML = '';
    let highlightCreated = false;

    // Strategy 1: Exact Sentence Text-Matching via PDF.js Text Content
    if (currentCitationHighlight && currentCitationHighlight.exactQuote) {
      try {
        const textContent = await page.getTextContent();
        const quoteLower = currentCitationHighlight.exactQuote.toLowerCase().replace(/[^a-z0-9 ]/g, ' ');
        const quoteTokens = quoteLower.split(/\s+/).filter(w => w.length > 3);

        if (quoteTokens.length >= 3) {
          const matchingItems = [];
          for (const item of textContent.items) {
            const strLower = item.str.toLowerCase();
            const hitCount = quoteTokens.filter(t => strLower.includes(t)).length;
            if (hitCount >= 2 || (quoteTokens.length <= 4 && hitCount >= 1)) {
              matchingItems.push(item);
            }
          }

          if (matchingItems.length > 0) {
            for (const item of matchingItems) {
              const tx = pdfjsLib.Util.transform(viewport.transform, item.transform);
              const itemWidth = item.width * currentScale;
              const itemHeight = Math.max(16, (item.height || 12) * currentScale);

              const hl = document.createElement('div');
              hl.className = 'pdf-highlight';
              hl.style.left = `${Math.max(0, tx[4])}px`;
              hl.style.top = `${Math.max(0, tx[5] - itemHeight)}px`;
              hl.style.width = `${Math.max(60, itemWidth + 8)}px`;
              hl.style.height = `${itemHeight + 6}px`;
              highlightLayer.appendChild(hl);
            }
            highlightCreated = true;
          }
        }
      } catch (err) {
        console.warn('Text-matching highlight fallback:', err);
      }
    }

    // Strategy 2: Bounding Box fallback
    if (!highlightCreated && currentCitationHighlight && currentCitationHighlight.bbox && currentCitationHighlight.bbox[2] > 0) {
      const [x0, y0, x1, y1] = currentCitationHighlight.bbox;
      const scaleX = viewport.width / (page.view[2] || 595.3);
      const scaleY = viewport.height / (page.view[3] || 841.9);

      const highlightEl = document.createElement('div');
      highlightEl.className = 'pdf-highlight';
      highlightEl.style.left = `${x0 * scaleX}px`;
      highlightEl.style.top = `${y0 * scaleY}px`;
      highlightEl.style.width = `${Math.max(40, (x1 - x0) * scaleX)}px`;
      highlightEl.style.height = `${Math.max(24, (y1 - y0) * scaleY)}px`;
      highlightLayer.appendChild(highlightEl);
      highlightCreated = true;
    }

    // Strategy 3: Focused paragraph indicator (compact focused box, never covering whole page)
    if (!highlightCreated) {
      const highlightEl = document.createElement('div');
      highlightEl.className = 'pdf-highlight';
      highlightEl.style.left = '8%';
      highlightEl.style.top = '28%';
      highlightEl.style.width = '84%';
      highlightEl.style.height = '140px';
      highlightLayer.appendChild(highlightEl);
    }

    // Scroll PDF container smoothly
    const firstHl = highlightLayer.querySelector('.pdf-highlight');
    if (firstHl) {
      const topPos = parseInt(firstHl.style.top || '0', 10);
      const container = document.getElementById('pdfContainer');
      container.scrollTo({ top: Math.max(0, topPos - 40), behavior: 'smooth' });
    }

  } catch (err) {
    console.error('Error rendering page:', err);
  }
}

function prevPdfPage() {
  if (currentPageNum > 1) {
    currentPageNum--;
    currentCitationHighlight = null;
    renderPdfPage(currentPageNum);
  }
}

function nextPdfPage() {
  if (currentPdfDoc && currentPageNum < currentPdfDoc.numPages) {
    currentPageNum++;
    currentCitationHighlight = null;
    renderPdfPage(currentPageNum);
  }
}

function zoomPdf(delta) {
  currentScale = Math.max(0.6, Math.min(2.5, currentScale + delta));
  if (currentPdfDoc) {
    renderPdfPage(currentPageNum);
  }
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function escapeAttr(str) {
  if (!str) return '';
  return str.replace(/"/g, '&quot;').replace(/'/g, '&#39;').replace(/\n/g, ' ');
}
