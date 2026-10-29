/**
 * AI TrustLens — Modern Dashboard Controller
 * Handles user interactions, preset queries, API integration,
 * reactive highlight annotations, and dynamic scorecard rendering.
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const questionInput = document.getElementById('clinicalInput');
  const btnVerify = document.getElementById('btnVerify');
  const btnClear = document.getElementById('btnClear');
  const presetChipsContainer = document.getElementById('presetChips');
  const modeTabs = document.querySelectorAll('.tab-btn');
  const pipelineStepper = document.getElementById('pipelineStepper');
  const resultsSection = document.getElementById('resultsSection');
  const filterBtns = document.querySelectorAll('.filter-btn');
  
  // Scorecard Elements
  const gaugeVal = document.getElementById('gaugeVal');
  const gaugeCircle = document.getElementById('gaugeCircle');
  const statTotal = document.getElementById('statTotal');
  const statSupported = document.getElementById('statSupported');
  const statRefuted = document.getElementById('statRefuted');
  const statReview = document.getElementById('statReview');
  const statInsufficient = document.getElementById('statInsufficient');
  const annotatedAnswer = document.getElementById('annotatedAnswer');
  const claimsGrid = document.getElementById('claimsGrid');


  // State
  let currentMode = 'question';
  let currentResult = null;
  let activeFilter = 'all';

  // 1. Fetch & Initialize Presets & Health
  initHealth();
  initPresets();

  // Mode Tabs Switching
  modeTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      modeTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      currentMode = tab.dataset.mode;

      if (currentMode === 'question') {
        questionInput.placeholder = 'Type a medical question (e.g. What are the first-line treatments for hypertension?)...';
        btnVerify.innerHTML = `
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"></path>
          </svg>
          Ask & Verify Claims
        `;
      } else {
        questionInput.placeholder = 'Paste or type a clinical statement / AI answer to verify each factual claim directly...';
        btnVerify.innerHTML = `
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="20 6 9 17 4 12"></polyline>
          </svg>
          Verify Text Directly
        `;
      }
    });
  });

  // Clear Input
  btnClear.addEventListener('click', () => {
    questionInput.value = '';
    questionInput.focus();
  });

  // Verify Action
  btnVerify.addEventListener('click', () => {
    executeVerification();
  });

  // Filter Buttons
  filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      filterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeFilter = btn.dataset.filter;
      applyClaimFilter(activeFilter);
    });
  });



  // ---------------------------------------------------------------------------
  // Core Functions
  // ---------------------------------------------------------------------------

  async function initHealth() {
    try {
      const res = await fetch('/api/health');
      const data = await res.json();
      const dot = document.getElementById('statusDot');
      const text = document.getElementById('statusText');
      if (data.rag_ok) {
        dot.classList.remove('error');
        text.textContent = 'RAG Microservice Online (Port 8000)';
      } else {
        dot.classList.add('error');
        text.textContent = 'RAG Offline (Check port 8000)';
      }
    } catch (e) {
      console.warn('Health check failed:', e);
    }
  }

  async function initPresets() {
    try {
      const res = await fetch('/api/presets');
      const presets = await res.json();
      presetChipsContainer.innerHTML = '';

      presets.forEach(p => {
        const chip = document.createElement('button');
        chip.className = 'preset-chip';
        chip.type = 'button';
        chip.innerHTML = `<span>⚡</span> ${p.title}`;
        chip.addEventListener('click', () => {
          if (p.mode === 'answer') {
            document.querySelector('[data-mode="answer"]').click();
            questionInput.value = p.text;
          } else {
            document.querySelector('[data-mode="question"]').click();
            questionInput.value = p.question;
          }
          questionInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
        });
        presetChipsContainer.appendChild(chip);
      });
    } catch (e) {
      console.warn('Could not load presets:', e);
    }
  }

  async function executeVerification() {
    const text = questionInput.value.trim();
    if (!text) {
      questionInput.focus();
      return;
    }

    btnVerify.disabled = true;
    pipelineStepper.style.display = 'block';
    resultsSection.style.display = 'none';
    pipelineStepper.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    setStep(1, 'active');

    try {
      const response = await fetch('/api/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: currentMode,
          text: text,
          top_k: 5,
        })
      });

      // Safely parse initial launch response
      const rawText = await response.text();
      let initData = {};
      try {
        initData = JSON.parse(rawText);
      } catch (e) {
        throw new Error(
          response.status === 504 || response.status === 524
            ? "Cloudflare proxy timeout (request exceeded limit)."
            : `Server returned non-JSON response (HTTP ${response.status}): ${rawText.substring(0, 100)}`
        );
      }

      if (!response.ok) {
        throw new Error(initData.detail || initData.error || `HTTP ${response.status}`);
      }

      const taskId = initData.task_id;
      if (!taskId) {
        throw new Error("No task ID returned by verification server.");
      }

      // Poll task status every 1.5 seconds (each request takes <10ms, eliminating Cloudflare timeouts)
      const pollTimer = setInterval(async () => {
        try {
          const pollRes = await fetch(`/api/task/${taskId}`);
          if (!pollRes.ok) return;

          const pollData = await pollRes.json();
          const elapsed = pollData.elapsed_seconds || 0;

          // Dynamically advance stepper stages based on model progress
          if (currentMode === 'question') {
            if (elapsed > 18 && elapsed < 85) {
              setStep(1, 'completed');
              setStep(2, 'active');
            } else if (elapsed >= 85) {
              setStep(2, 'completed');
              setStep(3, 'active');
            }
          } else {
            if (elapsed > 2) {
              setStep(1, 'completed');
              setStep(2, 'active');
            }
          }

          if (pollData.status === 'completed') {
            clearInterval(pollTimer);
            setStep(1, 'completed');
            setStep(2, 'completed');
            setStep(3, 'completed');
            setStep(4, 'completed');

            currentResult = pollData.result;
            setTimeout(() => {
              try {
                pipelineStepper.style.display = 'none';
                renderResults(pollData.result);
                resultsSection.style.display = 'block';
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
              } catch (err) {
                console.error("Render error:", err);
                alert("Render error: " + err.message);
              } finally {
                btnVerify.disabled = false;
              }
            }, 300);

          } else if (pollData.status === 'failed') {
            clearInterval(pollTimer);
            pipelineStepper.style.display = 'none';
            btnVerify.disabled = false;
            alert(`Verification Error: ${pollData.error || 'Pipeline execution failed.'}`);
          }
        } catch (pollErr) {
          console.warn("Polling warning:", pollErr);
        }
      }, 1500);

    } catch (error) {
      alert(`Pipeline Error: ${error.message}`);
      pipelineStepper.style.display = 'none';
      btnVerify.disabled = false;
    }
  }

  function setStep(stepIndex, state) {
    const steps = document.querySelectorAll('.step-item');
    if (steps[stepIndex - 1]) {
      steps[stepIndex - 1].className = `step-item ${state}`;
    }
  }

  // ---------------------------------------------------------------------------
  // Result Rendering
  // ---------------------------------------------------------------------------

  function renderResults(data) {
    const claims = data.claims || [];
    const summary = (data.verification && data.verification.summary) || {};
    const answerText = data.answer || '';

    const total = summary.total_claims !== undefined ? summary.total_claims : claims.length;
    const supported = summary.supported || 0;
    const refuted = summary.refuted || 0;
    const review = summary.review_required || 0;
    const insufficient = summary.not_enough_evidence || 0;

    // Metrics Scorecard (with null checks)
    if (statTotal) statTotal.textContent = total;
    if (statSupported) statSupported.textContent = supported;
    if (statRefuted) statRefuted.textContent = refuted;
    if (statReview) statReview.textContent = review;
    if (statInsufficient) statInsufficient.textContent = insufficient;

    // Overall Factual Reliability Gauge
    const evaluated = total - review;
    let accuracyPct = 0;
    if (evaluated > 0) {
      accuracyPct = Math.round((supported / evaluated) * 100);
    } else if (total > 0 && supported > 0) {
      accuracyPct = Math.round((supported / total) * 100);
    }

    if (gaugeVal) gaugeVal.textContent = `${accuracyPct}%`;
    if (gaugeCircle) {
      const circumference = 2 * Math.PI * 45; // r=45
      const offset = circumference - (accuracyPct / 100) * circumference;
      gaugeCircle.style.strokeDashoffset = offset;

      if (accuracyPct >= 75) {
        gaugeCircle.style.stroke = 'var(--status-supported)';
      } else if (accuracyPct >= 50) {
        gaugeCircle.style.stroke = 'var(--brand-cyan)';
      } else {
        gaugeCircle.style.stroke = 'var(--status-review)';
      }
    }

    // Render Annotated Answer Box
    renderAnnotatedAnswer(answerText, claims);

    // Render Claim Breakdown Cards Grid
    renderClaimsGrid(claims);
  }

  function renderAnnotatedAnswer(answerText, claims) {
    if (!annotatedAnswer) return;

    if (!answerText) {
      annotatedAnswer.innerHTML = '<em>No baseline text generated.</em>';
      return;
    }

    // If no claims extracted
    if (!claims || !claims.length) {
      annotatedAnswer.textContent = answerText;
      return;
    }

    // Collect valid numeric span offsets only
    const validSpans = [];
    claims.forEach((c, idx) => {
      const claimMeta = c.claim_meta || {};
      const start = claimMeta.source_start;
      const end = claimMeta.source_end;
      if (typeof start === 'number' && typeof end === 'number' && start >= 0 && end <= answerText.length && start < end) {
        validSpans.push({ claim: c, idx: idx, start: start, end: end });
      }
    });

    if (validSpans.length === 0) {
      annotatedAnswer.textContent = answerText;
      return;
    }

    // Sort by start position to prevent index collision
    validSpans.sort((a, b) => a.start - b.start);

    let html = '';
    let lastIndex = 0;

    validSpans.forEach(item => {
      if (item.start >= lastIndex) {
        html += escapeHtml(answerText.substring(lastIndex, item.start));
        const status = (item.claim.decision_status || item.claim.verification_status || 'UNKNOWN').toLowerCase();
        let statusClass = 'supported';
        if (status.includes('refut')) statusClass = 'refuted';
        else if (status.includes('review')) statusClass = 'review';
        else if (status.includes('insufficient') || status.includes('not_enough')) statusClass = 'insufficient';

        html += `<span class="claim-highlight ${statusClass}" data-claim-idx="${item.idx}" title="Claim #${item.idx + 1}: ${escapeHtml(item.claim.claim || '')} [${status.toUpperCase()}]">${escapeHtml(answerText.substring(item.start, item.end))}</span>`;
        lastIndex = item.end;
      }
    });

    if (lastIndex < answerText.length) {
      html += escapeHtml(answerText.substring(lastIndex));
    }

    annotatedAnswer.innerHTML = html;

    // Attach click events on highlights to scroll to claim card
    document.querySelectorAll('.claim-highlight').forEach(hl => {
      hl.addEventListener('click', () => {
        const idx = hl.dataset.claimIdx;
        const targetCard = document.getElementById(`claim-card-${idx}`);
        if (targetCard) {
          targetCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
          targetCard.style.borderColor = 'var(--brand-cyan)';
          targetCard.style.boxShadow = '0 0 25px var(--brand-cyan-glow)';
          setTimeout(() => {
            targetCard.style.borderColor = '';
            targetCard.style.boxShadow = '';
          }, 1800);
        }
      });
    });
  }

  function renderClaimsGrid(claims) {
    if (!claimsGrid) return;
    claimsGrid.innerHTML = '';

    if (!claims.length) {
      claimsGrid.innerHTML = '<div style="color:var(--text-muted); text-align:center; padding:2rem;">No atomic claims were extracted from this answer.</div>';
      return;
    }

    claims.forEach((c, idx) => {
      const status = c.decision_status || c.verification_status || 'NOT ENOUGH EVIDENCE';
      const category = c.category || 'medical';
      const confidence = c.confidence_score !== undefined ? c.confidence_score : (c.confidence || 0);
      const confPct = Math.round(confidence * 100);
      const explanation = c.explanation || c.verification_explanation || '';

      const card = document.createElement('div');
      card.className = 'claim-card';
      card.id = `claim-card-${idx}`;
      card.dataset.status = status.toUpperCase().replace(/\s+/g, '_');

      // Status badge styling
      let badgeClass = 'badge-supported';
      let barColor = 'var(--status-supported)';
      const st = status.toUpperCase();

      if (st.includes('REFUT')) {
        badgeClass = 'badge-refuted';
        barColor = 'var(--status-refuted)';
      } else if (st.includes('REVIEW')) {
        badgeClass = 'badge-review';
        barColor = 'var(--status-review)';
      } else if (st.includes('INSUFFICIENT') || st.includes('NOT_ENOUGH')) {
        badgeClass = 'badge-insufficient';
        barColor = 'var(--status-insufficient)';
      } else if (st.includes('PARTIALLY') || st.includes('CONFLICTING')) {
        badgeClass = 'badge-review';
        barColor = 'var(--status-conflicting)';
      }

      // Structural validation issues
      let issuesHtml = '';
      const structVal = c.structural_validation || {};
      if (structVal.is_structurally_valid === false && structVal.issues) {
        issuesHtml = `
          <div class="issues-alert">
            <span>⚠</span>
            <div><strong>Structural Integrity Warning:</strong> ${structVal.issues.join(' • ')}</div>
          </div>
        `;
      }

      // Supporting Evidence Box
      let evidenceHtml = '';
      const evidenceList = c.evidence || [];
      const supporting = evidenceList.filter(e => e.stance === 'supports');
      if (supporting.length > 0) {
        const topEv = supporting[0];
        const srcTitle = escapeHtml(topEv.source || 'Medical Knowledge Base');
        const srcUrl = topEv.source_url ? `<a href="${topEv.source_url}" target="_blank" rel="noopener" class="source-link">View Source ↗</a>` : '';
        const excerpt = escapeHtml(topEv.text || '').replace(/\n/g, ' ');

        evidenceHtml = `
          <div class="evidence-box">
            <div class="evidence-source-line">
              <span class="source-title">📄 ${srcTitle}</span>
              ${srcUrl}
            </div>
            <div class="evidence-excerpt">"${excerpt}"</div>
          </div>
        `;
      }

      // Knowledge Graph Triples
      let kgHtml = '';
      const kgList = c.knowledge_graph_evidence || [];
      if (kgList.length > 0) {
        const chips = kgList.slice(0, 2).map(t => {
          return `<span class="kg-chip">🕸️ (${escapeHtml(t.subject)}) —[${escapeHtml(t.predicate)}]→ (${escapeHtml(t.object)})</span>`;
        }).join('');
        kgHtml = `<div class="kg-triples-row">${chips}</div>`;
      }

      card.innerHTML = `
        <div class="claim-card-top">
          <div class="claim-meta-pills">
            <span class="claim-num">CLAIM #${idx + 1}</span>
            <span class="category-pill">${escapeHtml(category)}</span>
          </div>
          <span class="badge ${badgeClass}">${status.replace(/_/g, ' ')}</span>
        </div>

        <div class="claim-statement">"${escapeHtml(c.claim || '')}"</div>

        <div class="confidence-row">
          <span style="font-size:0.75rem; color:var(--text-muted); font-weight:600;">CONFIDENCE:</span>
          <div class="confidence-bar-bg">
            <div class="confidence-bar-fill" style="width: ${confPct}%; background: ${barColor};"></div>
          </div>
          <span class="confidence-label">${confPct}%</span>
        </div>

        ${issuesHtml}

        <div class="rationale-box">
          <strong>Clinical Verification Rationale:</strong> ${escapeHtml(explanation)}
        </div>

        ${evidenceHtml}
        ${kgHtml}
      `;

      claimsGrid.appendChild(card);
    });
  }

  function applyClaimFilter(filter) {
    const cards = document.querySelectorAll('.claim-card');
    cards.forEach(card => {
      const status = card.dataset.status;
      if (filter === 'all') {
        card.style.display = 'block';
      } else if (filter === 'supported' && status.includes('SUPPORTED') && !status.includes('PARTIALLY')) {
        card.style.display = 'block';
      } else if (filter === 'refuted' && status.includes('REFUTED')) {
        card.style.display = 'block';
      } else if (filter === 'review' && (status.includes('REVIEW') || status.includes('PARTIALLY') || status.includes('CONFLICTING'))) {
        card.style.display = 'block';
      } else if (filter === 'insufficient' && (status.includes('INSUFFICIENT') || status.includes('NOT_ENOUGH'))) {
        card.style.display = 'block';
      } else {
        card.style.display = 'none';
      }
    });
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }
});
