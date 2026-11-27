/*!
 * gemm-ladder-figs.js — Reusable inline animated CUDA GEMM figures for blog posts.
 * Zero external dependencies. Auto-mounts onto <div class="gemm-fig" data-scene="..."></div>.
 * Syncs automatically with Chirpy's light/dark theme toggle and pauses off-screen canvases.
 */
(function () {
  'use strict';

  /* =========================================================================
   * 1. Scoped Styles (injected once so posts need only a single <script> tag)
   * ========================================================================= */
  const STYLE_ID = 'gemm-fig-styles';
  if (!document.getElementById(STYLE_ID)) {
    const style = document.createElement('style');
    style.id = STYLE_ID;
    style.textContent = `
      .gemm-fig {
        margin: 1.5rem 0 1.75rem;
        border: 1px solid var(--main-border-color, #2a3446);
        border-radius: 0.8rem;
        overflow: hidden;
        background: var(--card-bg, #121824);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.16);
        font-family: 'Inter', 'Source Sans Pro', system-ui, -apple-system, sans-serif;
      }
      .gemm-fig-head {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 0.5rem 0.75rem;
        padding: 0.62rem 0.9rem;
        border-bottom: 1px solid var(--main-border-color, #2a3446);
        background: rgba(127, 142, 166, 0.08);
      }
      .gemm-fig-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: var(--heading-color, inherit);
        letter-spacing: -0.01em;
      }
      .gemm-fig-controls {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.35rem;
      }
      .gemm-fig-btn {
        appearance: none;
        border: 1px solid var(--main-border-color, #334361);
        background: var(--main-bg, #0d121c);
        color: var(--text-color, #cbd5e1);
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.25rem 0.62rem;
        border-radius: 999px;
        cursor: pointer;
        line-height: 1.25;
        transition: border-color 0.15s, background 0.15s, color 0.15s;
      }
      .gemm-fig-btn:hover { border-color: #22d3ee; }
      .gemm-fig-btn.active {
        background: rgba(34, 211, 238, 0.16);
        border-color: #0891b2;
        color: #0891b2;
      }
      .gemm-fig-btn.warn.active {
        background: rgba(234, 88, 12, 0.15);
        border-color: #ea580c;
        color: #ea580c;
      }
      .gemm-fig[data-theme="dark"] .gemm-fig-btn.active {
        color: #22d3ee;
        border-color: #22d3ee;
      }
      .gemm-fig[data-theme="dark"] .gemm-fig-btn.warn.active {
        color: #fb923c;
        border-color: #f97316;
      }
      .gemm-fig-canvas-wrap {
        position: relative;
        width: 100%;
        aspect-ratio: 16 / 9.6;
        min-height: 350px;
        max-height: 490px;
      }
      .gemm-fig[data-scene="roofline-twist"] .gemm-fig-canvas-wrap {
        aspect-ratio: 16 / 8.5;
        min-height: 310px;
        max-height: 410px;
      }
      .gemm-fig-canvas-wrap canvas {
        display: block;
        width: 100%;
        height: 100%;
      }
      .gemm-fig-foot {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.52rem 0.9rem;
        border-top: 1px solid var(--main-border-color, #2a3446);
        background: rgba(127, 142, 166, 0.06);
        font-size: 0.77rem;
        color: var(--text-muted-color, #8b949e);
        line-height: 1.35;
      }
      .gemm-fig-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #10b981;
        flex-shrink: 0;
      }
      .gemm-fig-dot.paused { background: #64748b; }
      .gemm-fig-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        padding: 0.55rem 0.9rem;
        border-bottom: 1px solid var(--main-border-color, #2a3446);
        background: rgba(127, 142, 166, 0.04);
        align-items: center;
      }
      .gemm-fig-knobs {
        display: grid;
        grid-template-columns: repeat(7, minmax(0, 1fr));
        gap: 0.38rem;
        width: 100%;
      }
      .gemm-fig-knob {
        border: 1px solid var(--main-border-color, #2a3446);
        border-radius: 0.45rem;
        padding: 0.28rem 0.35rem;
        background: var(--main-bg, rgba(0, 0, 0, 0.12));
        display: flex;
        flex-direction: column;
        gap: 0.22rem;
      }
      .gemm-fig-knob-lbl {
        display: flex;
        justify-content: space-between;
        font-size: 0.66rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', ui-monospace, monospace;
        color: var(--text-muted-color, #8b949e);
      }
      .gemm-fig-knob-lbl span:last-child { color: #0891b2; }
      .gemm-fig[data-theme="dark"] .gemm-fig-knob-lbl span:last-child { color: #22d3ee; }
      .gemm-fig-kopts { display: flex; gap: 0.18rem; }
      .gemm-fig-kopt {
        flex: 1;
        border: 1px solid var(--main-border-color, #334361);
        background: transparent;
        color: var(--text-color, #cbd5e1);
        font-size: 0.68rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', ui-monospace, monospace;
        padding: 0.16rem 0;
        border-radius: 0.28rem;
        cursor: pointer;
      }
      .gemm-fig-kopt.active {
        background: #0891b2;
        border-color: #0891b2;
        color: #ffffff;
      }
      .gemm-fig[data-theme="dark"] .gemm-fig-kopt.active {
        background: #22d3ee;
        border-color: #22d3ee;
        color: #06131a;
      }
      @media (max-width: 720px) {
        .gemm-fig-knobs { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .gemm-fig-canvas-wrap { aspect-ratio: 4 / 3.2; min-height: 310px; }
      }
    `;
    document.head.appendChild(style);
  }

  /* =========================================================================
   * 2. Hardware Specs & Autotuner Validator (exact match to my_matmul_kernels.py)
   * ========================================================================= */
  const GPUS = {
    h100: {
      name: 'H100 SXM5', sms: 132, fp32Tflops: 61.8, bf16Tflops: 989.4,
      bwTbs: 3.35, maxSmemKiB: 228, regsPerSm: 65536, maxThreadsPerSm: 2048,
      maxBlocksPerSm: 32, memLabel: 'HBM3', tcLabel: 'BF16 TC', tcShort: 'BF16',
      tcDesc: '4th-gen Tensor Cores (wgmma)'
    },
    a100: {
      name: 'A100 SXM4', sms: 108, fp32Tflops: 19.5, bf16Tflops: 312.0,
      bwTbs: 2.04, maxSmemKiB: 164, regsPerSm: 65536, maxThreadsPerSm: 2048,
      maxBlocksPerSm: 32, memLabel: 'HBM2e', tcLabel: 'BF16 TC', tcShort: 'BF16',
      tcDesc: '3rd-gen Tensor Cores (mma)'
    },
    t4: {
      name: 'T4 (Colab)', sms: 40, fp32Tflops: 8.1, bf16Tflops: 65.0,
      bwTbs: 0.32, maxSmemKiB: 64, regsPerSm: 65536, maxThreadsPerSm: 1024,
      maxBlocksPerSm: 16, memLabel: 'GDDR6', tcLabel: 'FP16 TC', tcShort: 'FP16',
      tcDesc: '2nd-gen Turing Tensor Cores (mma)'
    }
  };

  const SEARCH_SPACE = {
    BM: [32, 64, 128], BN: [32, 64, 128], BK: [8, 16, 32],
    WM: [16, 32, 64], WN: [16, 32, 64], TM: [4, 8], TN: [4, 8]
  };

  const PRESETS = {
    baseline:     { BM: 64,  BN: 64,  BK: 16, WM: 16, WN: 32, TM: 4, TN: 4 },
    high_ai:      { BM: 128, BN: 128, BK: 16, WM: 32, WN: 64, TM: 8, TN: 8 },
    small_block:  { BM: 32,  BN: 32,  BK: 32, WM: 16, WN: 32, TM: 4, TN: 4 },
    invalid_warp: { BM: 64,  BN: 64,  BK: 16, WM: 32, WN: 32, TM: 4, TN: 4 },
    invalid_fit:  { BM: 32,  BN: 64,  BK: 16, WM: 64, WN: 32, TM: 8, TN: 8 }
  };

  const ROOF_STEPS = [
    { id: 'R1', name: 'K1 Naive', ai: 0.061, eff: 0.45, note: '33 sectors/warp — 87.5% wasted HBM bus' },
    { id: 'R2', name: 'K2 Coalesced', ai: 0.40, eff: 0.72, note: '33 → 5 sectors (6.6×), still 0× reuse' },
    { id: 'R3', name: 'K3 SMEM Tile', ai: 8.0, eff: 0.48, note: '32× HBM reuse — MIO queue stalls under the roof!' },
    { id: 'R4', name: 'K4 1D Tiled', ai: 8.0, eff: 0.68, note: 'SAME 8.0 FLOP/B! Climbs vertically by cutting MIO (2.0 → 1.125)' },
    { id: 'R5', name: 'K5 2D Outer', ai: 16.0, eff: 0.76, note: '2× HBM AI + 0.50 rd/FMA, held down by 8×64 warp ribbon' },
    { id: 'R6', name: 'K6 Warp 16×32', ai: 16.0, eff: 0.86, note: 'SAME 16.0 FLOP/B! Climbs vertically via 72 → 48 warp perimeter' },
    { id: 'R7', name: 'K7 LDS.128', ai: 16.0, eff: 0.95, note: 'SAME 16.0 FLOP/B! Climbs vertically: 0 bank conflicts + 4× less MIO' },
    { id: 'R8', name: 'K8 Autotuned', ai: 32.0, eff: 0.94, note: '128×128×16 crosses the FP32 ridge into compute-bound territory!' }
  ];

  function evaluateConfig(cfg, gpu) {
    const { BM, BN, BK, WM, WN, TM, TN } = cfg;
    const lanesPerWarp = (WM / TM) * (WN / TN);
    const NT = (BM / TM) * (BN / TN);
    const smemBytes = (BM * BK + BK * BN) * 4;
    const smemKiB = smemBytes / 1024;
    const estRegs = (TM * TN) + (TM + TN) + 24;

    const checks = [
      { label: 'BM % WM == 0 & BN % WN == 0', ok: (BM % WM === 0) && (BN % WN === 0) },
      { label: 'WM % TM == 0 & WN % TN == 0', ok: (WM % TM === 0) && (WN % TN === 0) },
      { label: '(WM/TM)*(WN/TN) == 32 lanes', ok: lanesPerWarp === 32, detail: `${lanesPerWarp}` },
      { label: '32 <= NT <= 1024 threads/block', ok: Number.isInteger(NT) && NT >= 32 && NT <= 1024, detail: `${Math.round(NT)}` },
      { label: 'BK, BN, TM, TN % 4 == 0 (float4)', ok: BK % 4 === 0 && BN % 4 === 0 && TM % 4 === 0 && TN % 4 === 0 },
      { label: 'SMEM <= 48 KiB default cap', ok: smemBytes <= 48 * 1024, detail: `${smemKiB.toFixed(1)}K` },
      { label: 'Est. Regs <= 255 / thread', ok: estRegs <= 255, detail: `${estRegs}r` }
    ];
    const valid = checks.every(c => c.ok);
    const hbmAI = (BM * BN) / (2 * (BM + BN));
    const safeNT = Math.max(32, Math.round(NT));
    const warpsPerBlock = Math.max(1, Math.ceil(safeNT / 32));
    const regsPerBlock = Math.ceil((estRegs * 32) / 256) * 256 * warpsPerBlock;
    const blocksByRegs = Math.floor(gpu.regsPerSm / Math.max(1, regsPerBlock));
    const blocksBySmem = Math.floor((gpu.maxSmemKiB * 1024) / Math.max(1, smemBytes));
    const blocksByThreads = Math.floor(gpu.maxThreadsPerSm / safeNT);
    const activeBlocksPerSm = valid ? Math.max(0, Math.min(gpu.maxBlocksPerSm, blocksByRegs, blocksBySmem, blocksByThreads)) : 0;
    const occupancyPct = Math.min(100, ((activeBlocksPerSm * safeNT) / gpu.maxThreadsPerSm) * 100);

    let limiter = 'Max Blocks/SM';
    if (!valid) limiter = 'Invalid';
    else if (activeBlocksPerSm === blocksByRegs) limiter = 'Registers/SM';
    else if (activeBlocksPerSm === blocksByThreads) limiter = 'Max Threads/SM';
    else if (activeBlocksPerSm === blocksBySmem) limiter = 'SMEM/SM';

    const gridBlocks = Math.ceil(2048 / BN) * Math.ceil(2048 / BM);
    const waves = valid && activeBlocksPerSm > 0 ? gridBlocks / Math.max(1, activeBlocksPerSm * gpu.sms) : 0;

    return {
      valid, checks, NT: Math.round(NT), warpsPerBlock, smemKiB, estRegs, hbmAI,
      blocksByRegs, blocksBySmem, blocksByThreads, activeBlocksPerSm,
      occupancyPct, limiter, gridBlocks, waves
    };
  }

  /* =========================================================================
   * 3. Theme Detection (Chirpy data-mode + Bootstrap data-bs-theme + OS fallback)
   * ========================================================================= */
  function detectLightMode() {
    const html = document.documentElement;
    const dm = html.getAttribute('data-mode') || html.getAttribute('data-bs-theme');
    if (dm === 'light') return true;
    if (dm === 'dark') return false;
    try {
      const bg = getComputedStyle(document.body).backgroundColor;
      const m = bg && bg.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
      if (m) {
        const lum = 0.299 * (+m[1]) + 0.587 * (+m[2]) + 0.114 * (+m[3]);
        return lum > 140;
      }
    } catch (e) {}
    return !(window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches);
  }

  function getPal(light) {
    if (light) {
      return {
        bg: '#eef2f7', panel: '#ffffff', panelBorder: '#cbd5e1',
        grid: 'rgba(15, 23, 42, 0.08)', text: '#0f172a', textSub: '#334155',
        muted: '#64748b', onSolid: '#ffffff',
        hbm: '#ea580c', hbmBg: 'rgba(234, 88, 12, 0.14)',
        danger: '#e11d48', dangerBg: 'rgba(225, 29, 72, 0.14)',
        smem: '#0891b2', smemBg: 'rgba(8, 145, 178, 0.14)',
        reg: '#059669', regBg: 'rgba(5, 150, 105, 0.15)',
        warp: '#7c3aed', warpBg: 'rgba(124, 58, 237, 0.14)',
        gold: '#b45309', goldBg: 'rgba(180, 83, 9, 0.14)',
        cellEmpty: '#e2e8f0', cellStroke: '#cbd5e1'
      };
    }
    return {
      bg: '#0d121c', panel: '#141c2b', panelBorder: '#283650',
      grid: 'rgba(148, 163, 184, 0.10)', text: '#e6edf3', textSub: '#b4bfd0',
      muted: '#7f8ea6', onSolid: '#06131a',
      hbm: '#f97316', hbmBg: 'rgba(249, 115, 22, 0.18)',
      danger: '#f43f5e', dangerBg: 'rgba(244, 63, 94, 0.18)',
      smem: '#22d3ee', smemBg: 'rgba(34, 211, 238, 0.18)',
      reg: '#34d399', regBg: 'rgba(52, 211, 153, 0.18)',
      warp: '#a78bfa', warpBg: 'rgba(167, 139, 250, 0.20)',
      gold: '#facc15', goldBg: 'rgba(250, 204, 21, 0.18)',
      cellEmpty: '#1b263b', cellStroke: '#2a3954'
    };
  }

  /* =========================================================================
   * 4. Drawing Primitives
   * ========================================================================= */
  function roundRect(ctx, x, y, w, h, r) {
    const rr = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.arcTo(x + w, y, x + w, y + h, rr);
    ctx.arcTo(x + w, y + h, x, y + h, rr);
    ctx.arcTo(x, y + h, x, y, rr);
    ctx.arcTo(x, y, x + w, y, rr);
    ctx.closePath();
  }

  function drawBox(ctx, x, y, w, h, r, fill, stroke, lw = 1.2) {
    roundRect(ctx, x, y, w, h, r);
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke(); }
  }

  function drawArrow(ctx, x1, y1, x2, y2, color, lw = 2, head = 7) {
    const ang = Math.atan2(y2 - y1, x2 - x1);
    ctx.save();
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = lw;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - head * Math.cos(ang - 0.45), y2 - head * Math.sin(ang - 0.45));
    ctx.lineTo(x2 - head * Math.cos(ang + 0.45), y2 - head * Math.sin(ang + 0.45));
    ctx.closePath(); ctx.fill();
    ctx.restore();
  }

  function drawPacket(ctx, x1, y1, x2, y2, t, color, radius = 4.2) {
    const u = ((t % 1) + 1) % 1;
    const x = x1 + (x2 - x1) * u;
    const y = y1 + (y2 - y1) * u;
    ctx.save();
    ctx.shadowColor = color; ctx.shadowBlur = 10; ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  }

  function drawBadge(ctx, x, y, text, bg, fg, align = 'center', opaqueUnder = null) {
    ctx.save();
    ctx.font = '700 10.5px Inter, system-ui, sans-serif';
    const tw = ctx.measureText(text).width;
    const pw = tw + 18, ph = 21;
    const bx = align === 'center' ? x - pw / 2 : align === 'right' ? x - pw : x;
    if (opaqueUnder) drawBox(ctx, bx, y, pw, ph, 10.5, opaqueUnder, null, 0);
    drawBox(ctx, bx, y, pw, ph, 10.5, bg, fg, 1.2);
    ctx.fillStyle = fg; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(text, bx + pw / 2, y + ph / 2 + 0.5);
    ctx.restore();
  }

  function drawFitText(ctx, text, x, y, maxW, fontSpec, color, align = 'left') {
    ctx.save();
    ctx.font = fontSpec; ctx.fillStyle = color; ctx.textAlign = align;
    ctx.fillText(text, x, y, maxW);
    ctx.restore();
  }

  /* =========================================================================
   * 5. Focused Scene Renderers
   * ========================================================================= */
  const SCENES = {
    'roofline-twist': {
      title: 'The Roofline Plot Twist — Moving Right vs. Climbing Vertically Under the Roof',
      hasGpuBar: true,
      hasStepBar: true,
      toggles: [{ key: 'showTC', label: 'Show Tensor Core Roof' }],
      render(ctx, W, H, t, pal, inst) {
        const gpu = GPUS[inst.gpuKey];
        const showTC = inst.toggles.showTC;
        const stepIdx = inst.stepIdx;
        const step = ROOF_STEPS[stepIdx];
        inst.setCaption(`${step.id} (${step.name}): ${step.note}`);

        const ridgeFP32 = gpu.fp32Tflops / gpu.bwTbs;
        const ridgeTC = gpu.bf16Tflops / gpu.bwTbs;
        const pL = 54, pR = W - 24, pT = 30, pB = H - 38;
        const pw = pR - pL, ph = pB - pT;

        const xMin = 0.04, yMin = 0.1;
        const minLogX = Math.log10(xMin);
        const maxLogX = Math.log10(showTC ? 512 : 96);
        const minLogY = Math.log10(yMin);
        const maxLogY = Math.log10(showTC ? Math.max(1700, gpu.bf16Tflops * 2.2) : Math.max(120, gpu.fp32Tflops * 2.1));

        const xOf = ai => pL + ((Math.log10(Math.max(xMin, ai)) - minLogX) / (maxLogX - minLogX)) * pw;
        const yOf = tf => pB - ((Math.log10(Math.max(yMin, tf)) - minLogY) / (maxLogY - minLogY)) * ph;

        const bwStartX = Math.max(xMin, yMin / gpu.bwTbs);
        const bwStartY = bwStartX * gpu.bwTbs;

        // Grid
        ctx.strokeStyle = pal.grid; ctx.lineWidth = 1;
        const xTicks = showTC ? [0.1, 1, 8, 16, 32, 100, 300] : [0.1, 1, 8, 16, 32, 64];
        xTicks.forEach(v => {
          const x = xOf(v);
          if (x > pL && x < pR) {
            ctx.beginPath(); ctx.moveTo(x, pT); ctx.lineTo(x, pB); ctx.stroke();
            ctx.fillStyle = pal.muted; ctx.font = '700 9.5px JetBrains Mono, monospace'; ctx.textAlign = 'center';
            ctx.fillText(v < 1 ? v.toFixed(1) : `${v}`, x, pB + 14);
          }
        });
        const yTicks = showTC ? [0.1, 1, 10, 100, 1000] : [0.1, 1, 10, 50];
        yTicks.forEach(v => {
          const y = yOf(v);
          if (y >= pT && y <= pB) {
            ctx.beginPath(); ctx.moveTo(pL, y); ctx.lineTo(pR, y); ctx.stroke();
            ctx.fillStyle = pal.muted; ctx.font = '700 9.5px JetBrains Mono, monospace'; ctx.textAlign = 'right';
            ctx.fillText(v < 1 ? v.toFixed(1) : `${v}`, pL - 7, y + 3);
          }
        });

        // Shaded regions
        const rxFP32 = xOf(ridgeFP32);
        ctx.fillStyle = inst.light ? 'rgba(234, 88, 12, 0.05)' : 'rgba(249, 115, 22, 0.06)';
        ctx.fillRect(pL, pT, Math.max(0, rxFP32 - pL), ph);
        ctx.fillStyle = inst.light ? 'rgba(5, 150, 105, 0.05)' : 'rgba(52, 211, 153, 0.06)';
        ctx.fillRect(rxFP32, pT, Math.max(0, pR - rxFP32), ph);

        // Axes
        ctx.strokeStyle = pal.panelBorder; ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.moveTo(pL, pT); ctx.lineTo(pL, pB); ctx.lineTo(pR, pB); ctx.stroke();
        ctx.fillStyle = pal.muted; ctx.font = '700 10px Inter, system-ui, sans-serif'; ctx.textAlign = 'center';
        ctx.fillText('Exact HBM Arithmetic Intensity (FLOPs/B, log) · Vertical gap below roof = conceptual stall headroom (not measured)', pL + pw / 2, H - 8);
        ctx.save(); ctx.translate(15, pT + ph / 2); ctx.rotate(-Math.PI / 2);
        ctx.fillText('Roofline Ceiling TFLOP/s (log)', 0, 0); ctx.restore();

        // Highlight the two VERTICAL climbs (at x=8 and x=16) with glowing vertical pillars!
        const climb8X = xOf(8.0);
        const climb16X = xOf(16.0);
        [
          { x: climb8X, label: 'Vertical Climb #1 (MIO)', active: stepIdx === 2 || stepIdx === 3 },
          { x: climb16X, label: 'Vertical Climb #2 (Warp/LDS.128)', active: stepIdx >= 4 && stepIdx <= 6 }
        ].forEach(cl => {
          ctx.fillStyle = cl.active ? pal.warpBg : (inst.light ? 'rgba(124,58,237,0.05)' : 'rgba(167,139,250,0.06)');
          ctx.fillRect(cl.x - 10, pT + 14, 20, ph - 14);
          if (cl.active) {
            drawBadge(ctx, cl.x, pT + 4, cl.label, pal.warpBg, pal.warp, 'center', pal.bg);
          }
        });

        // Optional TC roof
        if (showTC) {
          ctx.save();
          ctx.strokeStyle = pal.warp; ctx.lineWidth = 1.8; ctx.setLineDash([5, 4]);
          ctx.beginPath();
          ctx.moveTo(xOf(bwStartX), yOf(bwStartY));
          ctx.lineTo(xOf(ridgeTC), yOf(gpu.bf16Tflops));
          ctx.lineTo(pR, yOf(gpu.bf16Tflops));
          ctx.stroke(); ctx.setLineDash([]);
          ctx.fillStyle = pal.warp; ctx.font = '700 10px JetBrains Mono, monospace'; ctx.textAlign = 'right';
          ctx.fillText(`${gpu.tcLabel} ${gpu.bf16Tflops.toFixed(0)} TF (ridge ${ridgeTC.toFixed(0)})`, pR - 6, Math.max(pT + 10, yOf(gpu.bf16Tflops) - 6));
          ctx.restore();
        }

        // Main FP32 Roof
        ctx.strokeStyle = pal.smem; ctx.lineWidth = 2.6;
        ctx.beginPath();
        ctx.moveTo(xOf(bwStartX), yOf(bwStartY));
        ctx.lineTo(rxFP32, yOf(gpu.fp32Tflops));
        ctx.lineTo(pR, yOf(gpu.fp32Tflops));
        ctx.stroke();

        // FP32 Ridge line
        ctx.strokeStyle = pal.gold; ctx.lineWidth = 1.1; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(rxFP32, yOf(gpu.fp32Tflops)); ctx.lineTo(rxFP32, pB); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = pal.smem; ctx.font = '700 10px JetBrains Mono, monospace'; ctx.textAlign = 'right';
        ctx.fillText(`${gpu.name} FP32 ${gpu.fp32Tflops} TF`, pR - 6, yOf(gpu.fp32Tflops) - 6);
        ctx.fillStyle = pal.gold; ctx.textAlign = 'left';
        ctx.fillText(`FP32 Ridge ${ridgeFP32.toFixed(1)}`, Math.min(pR - 92, rxFP32 + 5), pB - 7);

        // Connected staircase trajectory R1 -> R8
        const pts = ROOF_STEPS.map(s => {
          const ceil = Math.min(gpu.fp32Tflops, s.ai * gpu.bwTbs);
          return { x: xOf(s.ai), y: yOf(Math.max(yMin * 1.08, ceil * s.eff)), s };
        });

        ctx.strokeStyle = pal.reg; ctx.lineWidth = 2;
        ctx.beginPath();
        pts.forEach((p, i) => { if (i === 0) ctx.moveTo(p.x, p.y); else if (i <= stepIdx) ctx.lineTo(p.x, p.y); });
        ctx.stroke();

        // Future dashed path
        if (stepIdx < pts.length - 1) {
          ctx.strokeStyle = pal.muted; ctx.lineWidth = 1.2; ctx.setLineDash([3, 4]);
          ctx.beginPath();
          for (let i = stepIdx; i < pts.length; i++) {
            if (i === stepIdx) ctx.moveTo(pts[i].x, pts[i].y); else ctx.lineTo(pts[i].x, pts[i].y);
          }
          ctx.stroke(); ctx.setLineDash([]);
        }

        // Animated pulse moving into active node
        if (stepIdx > 0) {
          drawPacket(ctx, pts[stepIdx - 1].x, pts[stepIdx - 1].y, pts[stepIdx].x, pts[stepIdx].y, (t * 0.75) % 1, pal.gold, 4);
        }

        // Nodes
        pts.forEach((p, i) => {
          const isCur = i === stepIdx;
          const isPast = i < stepIdx;
          ctx.save();
          if (isCur) { ctx.shadowColor = pal.gold; ctx.shadowBlur = 14; }
          ctx.fillStyle = isCur ? pal.gold : isPast ? pal.reg : pal.muted;
          ctx.beginPath(); ctx.arc(p.x, p.y, isCur ? 6.5 : 4, 0, Math.PI * 2); ctx.fill();
          if (isCur) { ctx.strokeStyle = pal.bg; ctx.lineWidth = 1.8; ctx.stroke(); }
          ctx.restore();
        });

        // Opaque active callout pill
        const cur = pts[stepIdx];
        const lbl = `${step.id} · ${step.name} (${step.ai < 1 ? step.ai.toFixed(2) : step.ai.toFixed(1)} FLOP/B)`;
        ctx.save();
        ctx.font = '800 10.5px JetBrains Mono, monospace';
        const pillW = ctx.measureText(lbl).width + 16, pillH = 21;
        let px = cur.x > pL + pw * 0.5 ? cur.x - pillW - 12 : cur.x + 12;
        px = Math.max(pL + 6, Math.min(pR - pillW - 6, px));
        let py = cur.y + 10;
        if (py + pillH > pB - 20) py = cur.y - pillH - 10;
        py = Math.max(pT + 26, Math.min(pB - pillH - 6, py));
        drawBox(ctx, px, py, pillW, pillH, 6, pal.panel, pal.gold, 1.3);
        ctx.fillStyle = pal.text; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(lbl, px + pillW / 2, py + pillH / 2 + 0.5);
        ctx.restore();
      }
    },

    coalescing: {
      title: '1. Global Coalescing · 33 vs. 5 HBM Sectors ("Crates") per Warp',
      toggles: [{ key: 'naive', label: 'Show K1 Naive (33 Sectors · 87.5% Waste)', warn: true }],
      render(ctx, W, H, t, pal, inst) {
        const naive = inst.toggles.naive;
        const kStep = Math.floor(t * 0.9) % 8;
        const pulse = (t * 0.9) % 1;
        const pad = 16;

        if (naive) {
          inst.setCaption(`K1 Naive (out_row = tx) · k=${kStep}: Warp 0 jumps stride-K across 32 rows of matA → 32 red crates (1/8 float used!) + 1 B broadcast = 33 sectors.`);
        } else {
          inst.setCaption(`K2 Coalesced (out_col = tx) · k=${kStep}: Warp 0 broadcasts 1 matA crate + reads 32 contiguous matB floats in 4 full crates = 5 sectors (6.6× drop!).`);
        }

        const topY = 38;
        const availMatW = W - pad * 2 - 52;
        const matW = Math.min(188, availMatW / 3);
        const matH = Math.min(168, H * 0.42);
        const gapX = (W - pad * 2 - matW * 3) / 2;
        const ax = pad, bx = ax + matW + gapX, cx = bx + matW + gapX;

        drawFitText(ctx, naive ? 'matA · 32 Strided Rows!' : 'matA · 1 Broadcast Cell', ax, topY - 10, matW, '800 11.5px Inter, system-ui, sans-serif', naive ? pal.danger : pal.warp, 'left');
        drawFitText(ctx, naive ? 'matB · 1 Broadcast Cell' : 'matB · 32 Contiguous Floats', bx, topY - 10, matW, '800 11.5px Inter, system-ui, sans-serif', pal.reg, 'left');
        drawFitText(ctx, naive ? 'matC · Warp 0 Column' : 'matC · Warp 0 Row', cx, topY - 10, matW, '800 11.5px Inter, system-ui, sans-serif', pal.text, 'left');

        const N = 8, cw = matW / N, ch = matH / N;
        for (let r = 0; r < N; r++) {
          for (let c = 0; c < N; c++) {
            const actA = naive ? (c === kStep) : (r === 0 && c === kStep);
            ctx.fillStyle = actA ? (naive ? pal.danger : pal.warp) : pal.cellEmpty;
            ctx.fillRect(ax + c * cw + 1, topY + r * ch + 1, cw - 2, ch - 2);

            const actB = naive ? (r === kStep && c === 0) : (r === kStep);
            ctx.fillStyle = actB ? pal.reg : pal.cellEmpty;
            ctx.fillRect(bx + c * cw + 1, topY + r * ch + 1, cw - 2, ch - 2);

            const actC = naive ? (c === 0) : (r === 0);
            ctx.fillStyle = actC ? (naive ? pal.hbmBg : pal.smemBg) : pal.cellEmpty;
            ctx.fillRect(cx + c * cw + 1, topY + r * ch + 1, cw - 2, ch - 2);
            if (actC) {
              ctx.strokeStyle = naive ? pal.hbm : pal.smem; ctx.lineWidth = 1.1;
              ctx.strokeRect(cx + c * cw + 1.5, topY + r * ch + 1.5, cw - 3, ch - 3);
            }
          }
        }

        // Packets
        if (naive) {
          for (let r = 0; r < N; r += 2) drawPacket(ctx, ax + kStep * cw + cw / 2, topY + r * ch + ch / 2, cx + cw / 2, topY + r * ch + ch / 2, (pulse + r * 0.06) % 1, pal.danger, 3.6);
          drawPacket(ctx, bx + cw / 2, topY + kStep * ch + ch / 2, cx + cw / 2, topY + matH / 2, pulse, pal.reg, 4.2);
        } else {
          drawPacket(ctx, ax + kStep * cw + cw / 2, topY + ch / 2, cx + matW / 2, topY + ch / 2, pulse, pal.warp, 4.6);
          for (let c = 0; c < N; c += 2) drawPacket(ctx, bx + c * cw + cw / 2, topY + kStep * ch + ch / 2, cx + c * cw + cw / 2, topY + ch / 2, (pulse + c * 0.05) % 1, pal.reg, 3.8);
        }

        // Bottom sector crate visualizer
        const secY = topY + matH + 24;
        const secH = H - secY - 14;
        drawBox(ctx, pad, secY, W - pad * 2, secH, 10, pal.panel, naive ? pal.danger : pal.reg, 1.4);

        if (naive) {
          drawFitText(ctx, '32-Byte HBM Crates Triggered per Warp: 32 Strided A + 1 Broadcast B = 33 Sectors (1,056 B)', pad + 14, secY + 21, W - pad * 2 - 28, '800 11.5px Inter, system-ui, sans-serif', pal.danger, 'left');
          const availW = (W - pad * 2) * 0.68;
          const cols = 8, rows = 4;
          const crateW = (availW - (cols - 1) * 5) / cols;
          const crateH = Math.min(22, (secH - 46) / rows - 4);
          for (let i = 0; i < 32; i++) {
            const r = Math.floor(i / cols), c = i % cols;
            const x = pad + 14 + c * (crateW + 5), y = secY + 34 + r * (crateH + 5);
            const lit = i <= Math.floor(pulse * 36);
            drawBox(ctx, x, y, crateW, crateH, 3.5, lit ? pal.dangerBg : pal.cellEmpty, lit ? pal.danger : pal.cellStroke, 1);
            ctx.fillStyle = lit ? pal.danger : pal.muted;
            ctx.fillRect(x + 2, y + 2, Math.max(3.5, (crateW - 4) / 8), crateH - 4);
          }
          const sumX = pad + 14 + availW + 16;
          const sumW = W - pad - 14 - sumX;
          drawBox(ctx, sumX, secY + 34, sumW, secH - 46, 8, pal.bg, pal.danger, 1.3);
          drawFitText(ctx, '33 SECTORS', sumX + 10, secY + 60, sumW - 20, '800 16px JetBrains Mono, monospace', pal.danger, 'left');
          drawFitText(ctx, '87.5% of A wasted', sumX + 10, secY + 80, sumW - 20, '700 11px Inter, system-ui, sans-serif', pal.hbm, 'left');
          drawFitText(ctx, '0.06 FLOP/B', sumX + 10, secY + 100, sumW - 20, '700 11px JetBrains Mono, monospace', pal.textSub, 'left');
        } else {
          drawFitText(ctx, '32-Byte HBM Crates Triggered per Warp: 1 Broadcast A + 4 Full B = 5 Sectors (6.6× Drop!)', pad + 14, secY + 21, W - pad * 2 - 28, '800 11.5px Inter, system-ui, sans-serif', pal.reg, 'left');
          const boxY = secY + 36, boxH = Math.min(54, secH - 72);
          const totalInner = W - pad * 2 - 28;
          const unitW = (totalInner - 40) / 5;
          drawBox(ctx, pad + 14, boxY, unitW, boxH, 7, pal.warpBg, pal.warp, 1.5);
          drawFitText(ctx, '1 × matA', pad + 14 + unitW / 2, boxY + 21, unitW - 10, '800 11px JetBrains Mono, monospace', pal.warp, 'center');
          drawFitText(ctx, '32-lane bcast', pad + 14 + unitW / 2, boxY + 38, unitW - 10, '600 9.5px Inter, system-ui, sans-serif', pal.textSub, 'center');
          for (let s = 0; s < 4; s++) {
            const sx = pad + 14 + (s + 1) * (unitW + 10);
            drawBox(ctx, sx, boxY, unitW, boxH, 7, pal.regBg, pal.reg, 1.5);
            drawFitText(ctx, `matB Sec ${s}`, sx + unitW / 2, boxY + 21, unitW - 10, '800 11px JetBrains Mono, monospace', pal.reg, 'center');
            drawFitText(ctx, `Lanes ${s * 8}..${s * 8 + 7} (100%)`, sx + unitW / 2, boxY + 38, unitW - 10, '600 9.5px Inter, system-ui, sans-serif', pal.textSub, 'center');
          }
          drawBadge(ctx, W / 2, secY + secH - 27, 'Next Wall: Every thread STILL returns to slow HBM on every k-step (0× reuse!)', pal.hbmBg, pal.hbm, 'center');
        }
      }
    },

    smem: {
      title: '2a. Shared Memory Tiling · 32× HBM Reuse vs. MIO Queue Saturation',
      toggles: [],
      render(ctx, W, H, t, pal, inst) {
        const cycle = (t * 0.55) % 4;
        const tileIdx = Math.floor((t * 0.55) / 4) % 4;
        const phase = Math.floor(cycle);
        const frac = cycle - phase;
        const labels = [
          `Tile t=${tileIdx} · Step 1/4: 1,024 threads cooperatively stage 32×32 tiles HBM → SMEM (8 KiB)`,
          `Tile t=${tileIdx} · Step 2/4: cuda.syncthreads() #1 — load barrier before compute starts`,
          `Tile t=${tileIdx} · Step 3/4: 32 FMAs from SMEM (32× HBM reuse!), BUT 2 SMEM reads/FMA saturate MIO!`,
          `Tile t=${tileIdx} · Step 4/4: cuda.syncthreads() #2 — protect tile t before tile t+1 overwrites SMEM`
        ];
        inst.setCaption(labels[phase]);

        const pad = 16, colW = (W - pad * 4) / 3, topY = 26, boxH = H - 78;

        // 1. HBM
        drawBox(ctx, pad, topY, colW, boxH, 10, pal.panel, phase === 0 ? pal.hbm : pal.panelBorder, phase === 0 ? 1.8 : 1.2);
        drawFitText(ctx, '1. Off-Chip HBM (~500 cyc)', pad + 12, topY + 21, colW - 24, '800 11.5px Inter, system-ui, sans-serif', pal.hbm, 'left');
        const sx = pad + 12, stripW = colW - 24, aY = topY + 42, stripH = 52;
        drawFitText(ctx, 'matA sliding K-tile', sx, aY - 5, stripW, '700 10px JetBrains Mono, monospace', pal.textSub, 'left');
        for (let i = 0; i < 4; i++) {
          const tw = stripW / 4, act = i === tileIdx;
          drawBox(ctx, sx + i * tw + 2, aY, tw - 4, stripH, 5, act ? pal.hbmBg : pal.cellEmpty, act ? pal.hbm : pal.cellStroke, act ? 1.8 : 1);
          drawFitText(ctx, `t=${i}`, sx + i * tw + tw / 2, aY + stripH / 2 + 4, tw - 6, '700 10px JetBrains Mono, monospace', act ? pal.hbm : pal.muted, 'center');
        }
        const bY = aY + stripH + 26, bTileH = Math.min(30, (boxH - (bY - topY) - 20) / 4);
        drawFitText(ctx, 'matB sliding K-tile', sx, bY - 5, stripW, '700 10px JetBrains Mono, monospace', pal.textSub, 'left');
        for (let i = 0; i < 4; i++) {
          const act = i === tileIdx;
          drawBox(ctx, sx, bY + i * (bTileH + 4), stripW, bTileH, 5, act ? pal.hbmBg : pal.cellEmpty, act ? pal.hbm : pal.cellStroke, act ? 1.8 : 1);
          drawFitText(ctx, `B tile t=${i} (32×32)`, sx + stripW / 2, bY + i * (bTileH + 4) + bTileH / 2 + 3.5, stripW - 10, '700 9.5px JetBrains Mono, monospace', act ? pal.hbm : pal.muted, 'center');
        }

        // 2. SMEM
        const mx = pad * 2 + colW, syncGlow = (phase === 1 || phase === 3);
        drawBox(ctx, mx, topY, colW, boxH, 10, pal.panel, syncGlow ? pal.gold : pal.smem, syncGlow ? 2.2 : 1.5);
        drawFitText(ctx, '2. On-Chip SMEM (8 KiB)', mx + 12, topY + 21, colW - 24, '800 11.5px Inter, system-ui, sans-serif', pal.smem, 'left');
        const gSize = Math.min(colW - 32, (boxH - 84) / 2);
        const gx = mx + (colW - gSize) / 2, gaY = topY + 40, gbY = gaY + gSize + 24;
        const innerK = phase === 2 ? Math.floor(frac * 8) : -1;
        drawFitText(ctx, 'sh_memA[32,32] (4 KiB)', gx, gaY - 5, gSize, '700 9.5px JetBrains Mono, monospace', pal.textSub, 'left');
        drawFitText(ctx, 'sh_memB[32,32] (4 KiB)', gx, gbY - 5, gSize, '700 9.5px JetBrains Mono, monospace', pal.textSub, 'left');
        const cell = gSize / 8;
        for (let r = 0; r < 8; r++) {
          for (let c = 0; c < 8; c++) {
            const filled = phase > 0 || (r * 8 + c) / 64 <= frac;
            ctx.fillStyle = (phase === 2 && r === 2 && c === innerK) ? pal.gold : filled ? pal.smemBg : pal.cellEmpty;
            ctx.fillRect(gx + c * cell + 1, gaY + r * cell + 1, cell - 2, cell - 2);
            ctx.fillStyle = (phase === 2 && r === innerK && c === 5) ? pal.gold : filled ? pal.smemBg : pal.cellEmpty;
            ctx.fillRect(gx + c * cell + 1, gbY + r * cell + 1, cell - 2, cell - 2);
          }
        }
        if (syncGlow) {
          const waveY = topY + 28 + frac * (boxH - 40);
          ctx.save(); ctx.strokeStyle = pal.gold; ctx.lineWidth = 3; ctx.shadowColor = pal.gold; ctx.shadowBlur = 12;
          ctx.beginPath(); ctx.moveTo(mx + 6, waveY); ctx.lineTo(mx + colW - 6, waveY); ctx.stroke(); ctx.restore();
          drawBadge(ctx, mx + colW / 2, gaY + gSize / 2 - 10, phase === 1 ? 'syncthreads() #1' : 'syncthreads() #2', pal.goldBg, pal.gold, 'center', pal.panel);
        }
        if (phase === 0) {
          for (let p = 0; p < 3; p++) {
            drawPacket(ctx, pad + colW, aY + 12 + p * 14, mx, gaY + 12 + p * 20, (frac * 2 + p * 0.2) % 1, pal.hbm, 3.8);
            drawPacket(ctx, pad + colW, bY + 14 + p * 18, mx, gbY + 12 + p * 20, (frac * 2 + p * 0.25) % 1, pal.hbm, 3.8);
          }
        }

        // 3. Compute + MIO
        const rx = pad * 3 + colW * 2;
        drawBox(ctx, rx, topY, colW, boxH, 10, pal.panel, phase === 2 ? pal.danger : pal.panelBorder, phase === 2 ? 1.8 : 1.2);
        drawFitText(ctx, '3. Compute + MIO Queue', rx + 12, topY + 21, colW - 24, '800 11.5px Inter, system-ui, sans-serif', pal.reg, 'left');
        drawBox(ctx, rx + 12, topY + 34, colW - 24, 54, 7, pal.regBg, pal.reg, 1.2);
        drawFitText(ctx, '32× HBM Reuse!', rx + 20, topY + 56, colW - 40, '800 15px JetBrains Mono, monospace', pal.reg, 'left');
        drawFitText(ctx, '65,536 → 2,048 HBM reads', rx + 20, topY + 75, colW - 40, '600 10px Inter, system-ui, sans-serif', pal.textSub, 'left');

        const mioY = topY + 106;
        drawFitText(ctx, 'NEW WALL: MIO Queue Pressure', rx + 12, mioY, colW - 24, '800 11px Inter, system-ui, sans-serif', pal.danger, 'left');
        const mioLoad = phase === 2 ? 0.88 + 0.12 * Math.sin(frac * Math.PI) : 0.16;
        drawBox(ctx, rx + 12, mioY + 8, colW - 24, 22, 6, pal.bg, pal.panelBorder, 1);
        drawBox(ctx, rx + 14, mioY + 10, Math.max(8, (colW - 28) * mioLoad), 18, 4, mioLoad > 0.65 ? pal.danger : pal.smem, null);
        drawFitText(ctx, phase === 2 ? 'SATURATED: 2 rd / 1 FMA' : 'Idle during load/sync', rx + colW / 2, mioY + 22, colW - 32, '700 9.5px JetBrains Mono, monospace', mioLoad > 0.65 ? pal.onSolid : pal.text, 'center');

        const fmaY = mioY + 44, fmaW = colW - 24;
        drawBox(ctx, rx + 12, fmaY, fmaW, Math.max(76, boxH - (fmaY - topY) - 12), 8, pal.bg, pal.panelBorder, 1);
        drawFitText(ctx, 'sh_memA[ty,i] + sh_memB[i,tx]', rx + colW / 2, fmaY + 23, fmaW - 16, '700 10px JetBrains Mono, monospace', pal.smem, 'center');
        drawFitText(ctx, '↓ 2 SMEM loads ↓', rx + colW / 2, fmaY + 43, fmaW - 16, '700 10px JetBrains Mono, monospace', pal.danger, 'center');
        drawFitText(ctx, 'acc += a * b  (only 1 FMA!)', rx + colW / 2, fmaY + 63, fmaW - 16, '700 10px JetBrains Mono, monospace', pal.reg, 'center');
        if (phase === 2) {
          drawPacket(ctx, mx + colW, gaY + gSize * 0.35, rx, fmaY + 23, (frac * 4) % 1, pal.smem, 3.8);
          drawPacket(ctx, mx + colW, gbY + gSize * 0.5, rx, fmaY + 23, (frac * 4 + 0.5) % 1, pal.smem, 3.8);
        }

        // Bottom 4-step indicator
        const pY = H - 36, stepW = (W - pad * 2 - 24) / 4;
        ['1. HBM → SMEM', '2. syncthreads()', '3. SMEM Compute (MIO!)', '4. syncthreads()'].forEach((nm, i) => {
          const x = pad + i * (stepW + 8), act = i === phase;
          drawBox(ctx, x, pY, stepW, 24, 6, act ? (i === 2 ? pal.dangerBg : pal.smemBg) : pal.panel, act ? (i === 2 ? pal.danger : pal.smem) : pal.panelBorder, act ? 1.5 : 1);
          drawFitText(ctx, nm, x + stepW / 2, pY + 15, stepW - 10, '700 10px Inter, system-ui, sans-serif', act ? pal.text : pal.muted, 'center');
        });
      }
    },

    '1d': {
      title: '2b. 1D Register Tiling (TM=8) · Reusing b_val & The 1.0 SMEM Floor',
      toggles: [],
      render(ctx, W, H, t, pal, inst) {
        const dotK = Math.floor(t * 1.1) % 8;
        const subTm = Math.floor((t * 1.1 - Math.floor(t * 1.1)) * 8);
        inst.setCaption(`dot_k=${dotK} · Loaded b_val ONCE into register → fanned out across tm=0..7 (tm=${subTm}). 9 SMEM reads drive 8 FMAs (1.125 rd/FMA).`);

        const pad = 16, topY = 26, mainH = H - 44;
        const leftW = (W - pad * 4) * 0.30, midW = (W - pad * 4) * 0.38, rightW = (W - pad * 4) * 0.32;
        const lx = pad, mx = lx + leftW + pad, rx = mx + midW + pad;

        // Left: SMEM
        drawBox(ctx, lx, topY, leftW, mainH, 10, pal.panel, pal.smem, 1.3);
        drawFitText(ctx, 'SMEM Tiles (32×32)', lx + 12, topY + 21, leftW - 24, '800 11.5px Inter, system-ui, sans-serif', pal.smem, 'left');
        const aX = lx + 12, aY = topY + 42, aW = leftW - 24, aH = Math.min(136, mainH * 0.44);
        drawFitText(ctx, `shmem_A[0..7, k=${dotK}] (8 rd)`, aX, aY - 5, aW, '700 9.5px JetBrains Mono, monospace', pal.textSub, 'left');
        for (let r = 0; r < 8; r++) {
          for (let c = 0; c < 8; c++) {
            ctx.fillStyle = (c === dotK && r === subTm) ? pal.gold : c === dotK ? pal.smemBg : pal.cellEmpty;
            ctx.fillRect(aX + c * (aW / 8) + 1, aY + r * (aH / 8) + 1, aW / 8 - 2, aH / 8 - 2);
          }
        }
        const bY = aY + aH + 28, bH = Math.min(90, mainH - (bY - topY) - 14);
        drawFitText(ctx, `shmem_B[k=${dotK}, col] (1 rd!)`, aX, bY - 5, aW, '700 9.5px JetBrains Mono, monospace', pal.reg, 'left');
        for (let r = 0; r < 8; r++) {
          for (let c = 0; c < 8; c++) {
            ctx.fillStyle = (r === dotK && c === 2) ? pal.reg : pal.cellEmpty;
            ctx.fillRect(aX + c * (aW / 8) + 1, bY + r * (bH / 8) + 1, aW / 8 - 2, Math.max(3, bH / 8 - 2));
          }
        }

        // Center: Registers
        drawBox(ctx, mx, topY, midW, mainH, 10, pal.panel, pal.reg, 1.5);
        drawFitText(ctx, 'Thread 0 Registers (TM = 8)', mx + 12, topY + 21, midW - 24, '800 11.5px Inter, system-ui, sans-serif', pal.reg, 'left');
        const bvX = mx + 12, bvW = Math.min(82, midW * 0.31), bvH = 42, bvY = topY + mainH * 0.44;
        drawBox(ctx, bvX, bvY, bvW, bvH, 8, pal.regBg, pal.reg, 2);
        drawFitText(ctx, 'b_val', bvX + bvW / 2, bvY + 17, bvW - 8, '800 11.5px JetBrains Mono, monospace', pal.reg, 'center');
        drawFitText(ctx, 'Reused 8×!', bvX + bvW / 2, bvY + 33, bvW - 8, '600 9px Inter, system-ui, sans-serif', pal.textSub, 'center');

        const accW = Math.min(104, midW * 0.42), accX = mx + midW - accW - 12;
        const accTop = topY + 38, accStep = (mainH - 50) / 8;
        for (let tm = 0; tm < 8; tm++) {
          const y = accTop + tm * accStep, act = tm === subTm;
          ctx.strokeStyle = act ? pal.gold : 'rgba(52, 211, 153, 0.35)'; ctx.lineWidth = act ? 2.2 : 1;
          ctx.beginPath(); ctx.moveTo(bvX + bvW, bvY + bvH / 2); ctx.lineTo(accX, y + (accStep - 4) / 2); ctx.stroke();
          drawBox(ctx, accX, y, accW, accStep - 4, 5, act ? pal.regBg : pal.bg, act ? pal.gold : pal.reg, act ? 1.8 : 1);
          drawFitText(ctx, `res[${tm}] += A*b`, accX + accW / 2, y + (accStep - 4) / 2 + 3.5, accW - 8, '700 9.5px JetBrains Mono, monospace', act ? pal.text : pal.textSub, 'center');
        }
        drawPacket(ctx, bvX + bvW, bvY + bvH / 2, accX, accTop + subTm * accStep + (accStep - 4) / 2, (t * 6) % 1, pal.gold, 3.8);

        // Right: Block C + 1D Wall chart
        drawBox(ctx, rx, topY, rightW, mainH, 10, pal.panel, pal.panelBorder, 1.2);
        drawFitText(ctx, 'Block C (4 Warps, 128 thr)', rx + 12, topY + 21, rightW - 24, '800 11.5px Inter, system-ui, sans-serif', pal.text, 'left');
        const cX = rx + 12, cY = topY + 34, cW = rightW - 24, cH = 112, bandH = cH / 4;
        for (let w = 0; w < 4; w++) {
          drawBox(ctx, cX, cY + w * bandH + 2, cW, bandH - 4, 4, w === 0 ? pal.warpBg : pal.cellEmpty, w === 0 ? pal.warp : pal.cellStroke, 1);
          drawFitText(ctx, `Warp ${w} (rows ${w * 8}..${w * 8 + 7})`, cX + 22, cY + w * bandH + bandH / 2 + 3.5, cW - 28, '700 9.5px JetBrains Mono, monospace', w === 0 ? pal.warp : pal.muted, 'left');
        }
        drawBox(ctx, cX + 5, cY + 4, 10, bandH - 8, 2, pal.reg, pal.gold, 1.3);

        const chY = cY + cH + 18, chH = mainH - (chY - topY) - 12;
        drawBox(ctx, cX, chY, cW, chH, 7, pal.bg, pal.hbm, 1.2);
        drawFitText(ctx, 'THE 1D WALL: 1 + 1/TM > 1.0', cX + 10, chY + 17, cW - 20, '800 10px Inter, system-ui, sans-serif', pal.hbm, 'left');
        const pL = cX + 20, pR = cX + cW - 12, pT = chY + 26, pB = chY + chH - 20;
        const yOf = v => pB - ((v - 0.5) / 1.7) * (pB - pT);
        ctx.strokeStyle = pal.danger; ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(pL, yOf(1.0)); ctx.lineTo(pR, yOf(1.0)); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = pal.danger; ctx.font = '700 8.5px JetBrains Mono, monospace'; ctx.textAlign = 'left';
        ctx.fillText('1.0 floor', pL + 4, yOf(1.0) - 4);
        const tms = [1, 2, 4, 8, 16];
        ctx.strokeStyle = pal.smem; ctx.lineWidth = 2; ctx.beginPath();
        tms.forEach((tm, idx) => {
          const px = pL + (idx / (tms.length - 1)) * (pR - pL), py = yOf(1 + 1 / tm);
          if (idx === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        });
        ctx.stroke();
        tms.forEach((tm, idx) => {
          const px = pL + (idx / (tms.length - 1)) * (pR - pL), py = yOf(1 + 1 / tm);
          ctx.fillStyle = tm === 8 ? pal.gold : pal.smem;
          ctx.beginPath(); ctx.arc(px, py, tm === 8 ? 4.5 : 3, 0, Math.PI * 2); ctx.fill();
          ctx.fillStyle = pal.muted; ctx.textAlign = 'center'; ctx.fillText(`${tm}`, px, pB + 11);
        });
      }
    },

    '2d': {
      title: '3a. 2D Register Outer Product (4×4) · Area Beats Perimeter',
      toggles: [{ key: 'warpFootprint', label: 'Reveal Hidden Warp 0 Flaw (8×64 Ribbon)', warn: true }],
      render(ctx, W, H, t, pal, inst) {
        const pad = 18;
        if (inst.toggles.warpFootprint) {
          inst.setCaption('HIDDEN FLAW: In Kernel 5, Warp 0 (tid 0..31) sprawls across 2 rows of 16 patches = skinny 8×64 ribbon (72 unique SMEM floats/k)!');
          const topY = 40, size = Math.min(H - 94, W * 0.44), bx = pad + 68, by = topY + 34;
          drawFitText(ctx, '64×64 Block Tile (16×16 Grid of 4×4 Patches) — Where Does Warp 0 Land?', pad, 24, W - pad * 2, '800 12px Inter, system-ui, sans-serif', pal.hbm, 'left');
          drawBox(ctx, bx, topY, size, 22, 5, pal.dangerBg, pal.danger, 1.4);
          drawFitText(ctx, 'sh_memB: 64 unique floats loaded by Warp 0!', bx + size / 2, topY + 14.5, size - 12, '700 10px JetBrains Mono, monospace', pal.danger, 'center');
          const patchSz = size / 16;
          drawBox(ctx, bx - 54, by, 46, patchSz * 2, 5, pal.dangerBg, pal.danger, 1.4);
          drawFitText(ctx, '8 A', bx - 31, by + patchSz + 4, 40, '700 10px JetBrains Mono, monospace', pal.danger, 'center');

          const pulseLane = Math.floor(t * 14) % 32;
          for (let pr = 0; pr < 16; pr++) {
            for (let pc = 0; pc < 16; pc++) {
              const tid = pr * 16 + pc, isW0 = tid < 32, isW1 = tid >= 32 && tid < 64;
              ctx.fillStyle = tid === pulseLane ? pal.gold : isW0 ? pal.dangerBg : isW1 ? pal.warpBg : pal.cellEmpty;
              ctx.fillRect(bx + pc * patchSz + 0.7, by + pr * patchSz + 0.7, patchSz - 1.4, patchSz - 1.4);
            }
          }
          ctx.strokeStyle = pal.danger; ctx.lineWidth = 2; ctx.strokeRect(bx, by, size, patchSz * 2);

          const rx = bx + size + 22, rw = W - rx - pad;
          drawBox(ctx, rx, by - 6, rw, size + 6, 10, pal.panel, pal.hbm, 1.4);
          drawFitText(ctx, 'Warp 0 = 8 × 64 Ribbon', rx + 14, by + 20, rw - 28, '800 14px JetBrains Mono, monospace', pal.danger, 'left');
          const lines = [
            '• patches_per_row = 64 / 4 = 16',
            '• 32 lanes fill 2 rows of 16 patches',
            '• Output area: 8 × 64 = 512 FMAs/k',
            '• Perimeter: 8 + 64 = 72 floats!',
            '• 2-way bank conflict on sh_memB',
            '• Stride-16 conflicts on sh_memA',
            '',
            '→ Fix: Reshape Warp 0 into 16×32',
            '   (perimeter shrinks 72 → 48!)'
          ];
          lines.forEach((ln, i) => {
            const good = ln.startsWith('→') || ln.startsWith('   ');
            drawFitText(ctx, ln, rx + 14, by + 44 + i * 20, rw - 28, (good ? '800' : '600') + ' 11px Inter, system-ui, sans-serif', good ? pal.reg : pal.textSub, 'left');
          });
          return;
        }

        const cellIdx = Math.floor(t * 5.2) % 16, curI = Math.floor(cellIdx / 4), curJ = cellIdx % 4;
        inst.setCaption(`Outer Product (${curI},${curJ}) · reg_M[${curI}] × reg_N[${curJ}] → dot_products[${curI},${curJ}] (8 SMEM reads drive 16 FMAs = 0.50 rd/FMA!).`);
        const gridCell = Math.min(50, (H - 144) / 4), gridSz = gridCell * 4, gx = pad + 104, gy = 108;

        drawFitText(ctx, 'reg_N[0..3] ← 4 SMEM loads (sh_memB)', gx, 34, W - gx - pad, '800 11.5px Inter, system-ui, sans-serif', pal.smem, 'left');
        for (let j = 0; j < 4; j++) {
          const x = gx + j * gridCell, act = j === curJ;
          drawBox(ctx, x + 3, 44, gridCell - 6, 38, 6, act ? pal.smem : pal.smemBg, pal.smem, act ? 2 : 1.2);
          ctx.fillStyle = act ? pal.onSolid : pal.text; ctx.font = '800 11px JetBrains Mono, monospace'; ctx.textAlign = 'center';
          ctx.fillText(`N[${j}]`, x + gridCell / 2, 67);
        }
        ctx.save(); ctx.translate(24, gy + gridSz / 2); ctx.rotate(-Math.PI / 2);
        ctx.fillStyle = pal.reg; ctx.font = '800 11px Inter, system-ui, sans-serif'; ctx.textAlign = 'center';
        ctx.fillText('reg_M[0..3] ← 4 SMEM loads (sh_memA)', 0, 0); ctx.restore();
        for (let i = 0; i < 4; i++) {
          const y = gy + i * gridCell, act = i === curI;
          drawBox(ctx, 42, y + 3, 46, gridCell - 6, 6, act ? pal.reg : pal.regBg, pal.reg, act ? 2 : 1.2);
          ctx.fillStyle = act ? pal.onSolid : pal.text; ctx.font = '800 11px JetBrains Mono, monospace'; ctx.textAlign = 'center';
          ctx.fillText(`M[${i}]`, 65, y + gridCell / 2 + 4);
        }
        ctx.fillStyle = pal.smemBg; ctx.fillRect(gx + curJ * gridCell + 2, 84, gridCell - 4, gy + gridSz - 84);
        ctx.fillStyle = pal.regBg; ctx.fillRect(90, gy + curI * gridCell + 2, gx + gridSz - 90, gridCell - 4);

        for (let i = 0; i < 4; i++) {
          for (let j = 0; j < 4; j++) {
            const idx = i * 4 + j, x = gx + j * gridCell, y = gy + i * gridCell;
            const isCur = i === curI && j === curJ, done = idx < cellIdx;
            drawBox(ctx, x + 3, y + 3, gridCell - 6, gridCell - 6, 6, isCur ? pal.goldBg : done ? pal.regBg : pal.panel, isCur ? pal.gold : done ? pal.reg : pal.panelBorder, isCur ? 2.1 : 1.1);
            ctx.fillStyle = isCur ? pal.gold : done ? pal.reg : pal.textSub;
            ctx.font = '700 10px JetBrains Mono, monospace'; ctx.textAlign = 'center';
            ctx.fillText(`C${i}${j}`, x + gridCell / 2, y + gridCell / 2 - 2);
            ctx.font = '600 8.5px JetBrains Mono, monospace'; ctx.fillStyle = pal.muted;
            ctx.fillText(isCur ? 'FMA!' : done ? '✓' : '···', x + gridCell / 2, y + gridCell / 2 + 11);
          }
        }

        const rx = gx + gridSz + 26, rw = W - rx - pad;
        drawBox(ctx, rx, 44, rw, gy + gridSz - 44, 10, pal.panel, pal.panelBorder, 1.2);
        drawFitText(ctx, 'Area-to-Perimeter Win', rx + 14, 68, rw - 28, '800 13px Inter, system-ui, sans-serif', pal.reg, 'left');
        const rows = [
          { k: 'SMEM Loads / dot_k', v: '4 + 4 = 8 floats', c: pal.smem },
          { k: 'Register FMAs / dot_k', v: '4 × 4 = 16 FMAs', c: pal.reg },
          { k: 'SMEM Reads / FMA', v: '0.50 (breaks 1.0!)', c: pal.gold },
          { k: 'Block 64×64×16', v: 'Still 8 KiB SMEM!', c: pal.warp },
          { k: 'HBM AI Jump', v: '8.0 → 16.0 FLOP/B', c: pal.smem }
        ];
        const stepY = Math.min(42, (gy + gridSz - 90) / 5);
        rows.forEach((s, idx) => {
          const sy = 84 + idx * stepY;
          drawBox(ctx, rx + 12, sy, rw - 24, stepY - 6, 6, pal.bg, pal.panelBorder, 1);
          drawFitText(ctx, s.k, rx + 20, sy + 12, rw - 40, '700 9px Inter, system-ui, sans-serif', pal.muted, 'left');
          drawFitText(ctx, s.v, rx + 20, sy + stepY - 12, rw - 40, '800 11.5px JetBrains Mono, monospace', s.c, 'left');
        });
      }
    },

    warp: {
      title: '3b. Explicit 16×32 Warp Tiling · 8-Way & 4-Way Hardware Broadcasts',
      toggles: [],
      render(ctx, W, H, t, pal, inst) {
        const bcastRow = Math.floor(t * 1.4) % 4, bcastCol = Math.floor(t * 2.1) % 8, pulse = (t * 2.1) % 1;
        inst.setCaption(`Warp 0 (16×32) · tileA 8-way broadcast across L${bcastRow * 8}..L${bcastRow * 8 + 7} · tileB 4-way down col ${bcastCol} — 48 unique floats (-33% vs K5!).`);

        const pad = 16, leftW = Math.min(192, W * 0.28), topY = 24, boxH = H - 42;
        drawBox(ctx, pad, topY, leftW, boxH, 10, pal.panel, pal.warp, 1.3);
        drawFitText(ctx, 'Block 64×64 → 8 Warps', pad + 12, topY + 20, leftW - 24, '800 11px Inter, system-ui, sans-serif', pal.warp, 'left');

        const blkSz = leftW - 24, bx = pad + 12, by = topY + 32, ww = blkSz / 2, wh = blkSz / 4;
        for (let wr = 0; wr < 4; wr++) {
          for (let wc = 0; wc < 2; wc++) {
            const wid = wr * 2 + wc, isW0 = wid === 0;
            drawBox(ctx, bx + wc * ww + 2, by + wr * wh + 2, ww - 4, wh - 4, 4, isW0 ? pal.warpBg : pal.cellEmpty, isW0 ? pal.warp : pal.cellStroke, isW0 ? 1.8 : 1);
            const ly = wr === 0 ? by + wr * wh + wh * 0.78 : by + wr * wh + wh / 2 + 3.5;
            drawFitText(ctx, `W${wid} (16×32)`, bx + wc * ww + ww / 2, ly, ww - 8, '800 9.5px JetBrains Mono, monospace', isW0 ? pal.warp : pal.muted, 'center');
          }
        }
        ctx.strokeStyle = pal.danger; ctx.lineWidth = 1.4; ctx.setLineDash([4, 3]);
        ctx.strokeRect(bx + 2, by + 2, blkSz - 4, wh * 0.48); ctx.setLineDash([]);
        drawFitText(ctx, 'K5 Warp 0 (8×64)', bx + blkSz / 2, by + wh * 0.30, blkSz - 8, '700 8px JetBrains Mono, monospace', pal.danger, 'center');

        const sumY = by + blkSz + 18;
        drawFitText(ctx, '--- K5: 8×64 (72 flt)', bx, sumY, blkSz, '700 9.5px JetBrains Mono, monospace', pal.danger, 'left');
        drawFitText(ctx, '■  K6: 16×32 (48 flt)', bx, sumY + 17, blkSz, '700 9.5px JetBrains Mono, monospace', pal.warp, 'left');
        drawFitText(ctx, '-33% SMEM · 512 FMAs!', bx, sumY + 36, blkSz, '800 10.5px Inter, system-ui, sans-serif', pal.reg, 'left');

        // Right 4x8 Lane Broadcast Grid
        const rx = pad * 2 + leftW, rw = W - rx - pad;
        drawBox(ctx, rx, topY, rw, boxH, 10, pal.panel, pal.panelBorder, 1.2);
        drawFitText(ctx, 'Inside Warp 0: 32 Lanes (4×8 Grid of 4×4 Patches) — Hardware Multicast', rx + 12, topY + 20, rw - 24, '800 11.5px Inter, system-ui, sans-serif', pal.text, 'left');

        const gridLeft = rx + 86, gridTop = topY + 72, gridW = rw - 102, gridH = boxH - 106;
        const cw = gridW / 8, ch = gridH / 4;
        drawFitText(ctx, 'tileB[dot_k, 0..31] — 4-Way Vertical Broadcast ↓', gridLeft + gridW / 2, topY + 38, gridW, '700 9.5px JetBrains Mono, monospace', pal.smem, 'center');

        for (let c = 0; c < 8; c++) {
          const x = gridLeft + c * cw, act = c === bcastCol;
          drawBox(ctx, x + 2, topY + 46, cw - 4, 19, 4, act ? pal.smem : pal.smemBg, pal.smem, act ? 1.8 : 1);
          drawFitText(ctx, `${c * 4}..${c * 4 + 3}`, x + cw / 2, topY + 59, cw - 6, '700 8.5px JetBrains Mono, monospace', act ? pal.onSolid : pal.smem, 'center');
        }
        for (let r = 0; r < 4; r++) {
          const y = gridTop + r * ch, act = r === bcastRow;
          drawBox(ctx, rx + 12, y + 5, 66, ch - 10, 5, act ? pal.reg : pal.regBg, pal.reg, act ? 1.8 : 1);
          drawFitText(ctx, `A[${r * 4}..${r * 4 + 3}]`, rx + 45, y + ch / 2 - 2, 60, '700 9px JetBrains Mono, monospace', act ? pal.onSolid : pal.reg, 'center');
          drawFitText(ctx, '8-way →', rx + 45, y + ch / 2 + 10, 60, '600 8px Inter, system-ui, sans-serif', act ? pal.onSolid : pal.reg, 'center');
        }

        ctx.fillStyle = pal.regBg; ctx.fillRect(gridLeft, gridTop + bcastRow * ch + 2, gridW, ch - 4);
        ctx.fillStyle = pal.smemBg; ctx.fillRect(gridLeft + bcastCol * cw + 2, gridTop, cw - 4, gridH);

        for (let r = 0; r < 4; r++) {
          for (let c = 0; c < 8; c++) {
            const lane = r * 8 + c, x = gridLeft + c * cw, y = gridTop + r * ch;
            const cross = r === bcastRow && c === bcastCol, onRow = r === bcastRow, onCol = c === bcastCol;
            drawBox(ctx, x + 2.5, y + 3.5, cw - 5, ch - 7, 5, cross ? pal.goldBg : (onRow || onCol) ? pal.warpBg : pal.bg, cross ? pal.gold : onRow ? pal.reg : onCol ? pal.smem : pal.panelBorder, cross ? 2 : 1.1);
            drawFitText(ctx, `L${lane}`, x + cw / 2, y + ch / 2 - 1, cw - 8, '800 10px JetBrains Mono, monospace', cross ? pal.gold : pal.text, 'center');
            drawFitText(ctx, '4×4', x + cw / 2, y + ch / 2 + 11, cw - 8, '600 8px JetBrains Mono, monospace', pal.muted, 'center');
          }
        }
        drawPacket(ctx, gridLeft, gridTop + bcastRow * ch + ch / 2, gridLeft + gridW, gridTop + bcastRow * ch + ch / 2, pulse, pal.reg, 4);
        drawPacket(ctx, gridLeft + bcastCol * cw + cw / 2, gridTop, gridLeft + bcastCol * cw + cw / 2, gridTop + gridH, pulse, pal.smem, 4);
        drawBadge(ctx, rx + rw / 2, topY + boxH - 28, '8-Way tileA + 4-Way tileB Hardware Multicast → 0 Bank Conflicts', pal.regBg, pal.reg, 'center');
      }
    },

    vectorized: {
      title: '3c. Transposed sh_memA + 128-Bit Vector Loads (32 SMEM Banks)',
      toggles: [{ key: 'before', label: 'Before: (BM×BK) Scalar LDS.32', warn: true }],
      render(ctx, W, H, t, pal, inst) {
        const beforeMode = inst.toggles.before;
        const pulse = (t * 0.85) % 1, step4 = Math.floor(pulse * 4), pad = 16;

        inst.setCaption(beforeMode
          ? 'BEFORE (Kernel 5): sh_memA is (64×16) → stride-16 reads collide on Bank 0 & Bank 16 + force 4× scalar LDS.32!'
          : 'AFTER (Kernel 6): sh_memA transposed to (16×64) → 4 contiguous adjacent banks → 1× LDS.128 (0 bank conflicts, 4× less MIO!).');

        const topY = 22;
        drawBox(ctx, pad, topY, W - pad * 2, 108, 10, pal.panel, beforeMode ? pal.danger : pal.reg, 1.5);
        drawFitText(
          ctx,
          beforeMode ? '32 Physical SMEM Banks — Stride-16 Collisions on Bank 0 & Bank 16!' : '32 Physical SMEM Banks — Contiguous 128-Bit (4-Bank) Conflict-Free Reads!',
          pad + 14, topY + 20, W - pad * 2 - 28, '800 11.5px Inter, system-ui, sans-serif', beforeMode ? pal.danger : pal.reg, 'left'
        );

        const bankLeft = pad + 14, bankW = (W - pad * 2 - 28) / 32, bankY = topY + 32, bankH = 48;
        for (let b = 0; b < 32; b++) {
          let fill = pal.cellEmpty, stroke = pal.cellStroke;
          if (beforeMode) {
            if (b === 0 || b === 16) { fill = pal.danger; stroke = pal.gold; }
          } else if (b < 16) {
            const act = Math.floor(b / 4) === step4;
            fill = act ? pal.reg : pal.smemBg; stroke = act ? pal.gold : pal.smem;
          }
          ctx.fillStyle = fill; ctx.fillRect(bankLeft + b * bankW + 1, bankY, bankW - 2, bankH);
          ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.strokeRect(bankLeft + b * bankW + 1, bankY, bankW - 2, bankH);
          if (b % 4 === 0) {
            ctx.fillStyle = pal.muted; ctx.font = '700 8px JetBrains Mono, monospace'; ctx.textAlign = 'center';
            ctx.fillText(`B${b}`, bankLeft + b * bankW + bankW / 2, bankY + bankH + 13);
          }
        }

        const botY = topY + 120, botH = H - botY - 14, halfW = (W - pad * 3) / 2;
        // Left: layout
        drawBox(ctx, pad, botY, halfW, botH, 10, pal.panel, pal.panelBorder, 1.2);
        drawFitText(ctx, beforeMode ? 'sh_memA Shape: (BM=64 × BK=16)' : 'sh_memA Shape: TRANSPOSED (BK=16 × BM=64)', pad + 12, botY + 20, halfW - 24, '800 11.5px Inter, system-ui, sans-serif', pal.text, 'left');
        const gx = pad + 14, gy = botY + 34, gw = halfW - 28, gh = botH - 70;
        const cw = gw / 8, ch = gh / 4;
        for (let r = 0; r < 4; r++) {
          for (let c = 0; c < 8; c++) {
            const active = beforeMode ? (c === 0) : (r === 0 && c < 4);
            const cur = beforeMode ? (active && r === step4) : active;
            drawBox(ctx, gx + c * cw + 2, gy + r * ch + 2, cw - 4, ch - 4, 4, cur ? (beforeMode ? pal.danger : pal.reg) : active ? (beforeMode ? pal.dangerBg : pal.regBg) : pal.cellEmpty, active ? (beforeMode ? pal.danger : pal.reg) : pal.cellStroke, cur ? 1.8 : 1);
            if (active) drawFitText(ctx, beforeMode ? `r${r},k0` : `k0,r${c}`, gx + c * cw + cw / 2, gy + r * ch + ch / 2 + 3.5, cw - 6, '700 9px JetBrains Mono, monospace', cur ? pal.onSolid : pal.text, 'center');
          }
        }
        drawFitText(ctx, beforeMode ? 'Stride 16 (64B) → Bank 0, 16, 0, 16 (CONFLICT!)' : 'Stride 1 (4B) → Banks 0..3 (128-bit CONTIGUOUS!)', gx, botY + botH - 12, halfW - 28, '700 9.5px JetBrains Mono, monospace', beforeMode ? pal.danger : pal.reg, 'left');

        // Right: instructions
        const rx = pad * 2 + halfW, innerMaxW = halfW - 40;
        drawBox(ctx, rx, botY, halfW, botH, 10, pal.panel, beforeMode ? pal.danger : pal.smem, 1.3);
        drawFitText(ctx, beforeMode ? 'MIO Pipeline: 4 × Scalar LDS.32' : 'MIO Pipeline: 1 × Vector LDS.128', rx + 12, botY + 20, halfW - 24, '800 11.5px Inter, system-ui, sans-serif', beforeMode ? pal.danger : pal.smem, 'left');
        if (beforeMode) {
          for (let i = 0; i < 4; i++) {
            const iy = botY + 34 + i * 29, act = i === step4;
            drawBox(ctx, rx + 12, iy, halfW - 24, 23, 5, act ? pal.dangerBg : pal.bg, act ? pal.danger : pal.panelBorder, act ? 1.5 : 1);
            drawFitText(ctx, `LDS.32 reg_M[${i}] ← sh_memA[r+${i},k] ${i >= 2 ? '⚠ BANK CONFLICT' : ''}`, rx + 20, iy + 15, innerMaxW, '700 9.5px JetBrains Mono, monospace', act ? pal.danger : pal.textSub, 'left');
          }
        } else {
          const iy = botY + 38;
          drawBox(ctx, rx + 12, iy, halfW - 24, 44, 7, pal.regBg, pal.reg, 1.8);
          drawFitText(ctx, 'LDS.128 reg_M[0..3] ← sh_memA[k, r+0..3]', rx + 20, iy + 18, innerMaxW, '800 11px JetBrains Mono, monospace', pal.reg, 'left');
          drawFitText(ctx, 'Single 16B load · 0 bank conflicts · 4× less MIO', rx + 20, iy + 35, innerMaxW, '600 10px Inter, system-ui, sans-serif', pal.textSub, 'left');
          const iy2 = iy + 54;
          drawBox(ctx, rx + 12, iy2, halfW - 24, 44, 7, pal.smemBg, pal.smem, 1.6);
          drawFitText(ctx, 'LDS.128 reg_N[0..3] ← sh_memB[k, c+0..3]', rx + 20, iy2 + 18, innerMaxW, '800 11px JetBrains Mono, monospace', pal.smem, 'left');
          drawFitText(ctx, '2 vector loads → 16 FMAs (0.125 inst/FMA!)', rx + 20, iy2 + 35, innerMaxW, '600 10px Inter, system-ui, sans-serif', pal.textSub, 'left');
        }
        drawBadge(ctx, rx + halfW / 2, botY + botH - 25, beforeMode ? '0.50 LDS.32 inst/FMA + Stalls' : '0.125 LDS.128 inst/FMA (4× MIO Win)', beforeMode ? pal.dangerBg : pal.regBg, beforeMode ? pal.danger : pal.reg, 'center');
      }
    },

    autotune: {
      title: '4. 7-Hyperparameter Autotuner · Tile Geometry, Constraints & SM Occupancy',
      hasGpuBar: true,
      hasAutotuneKnobs: true,
      toggles: [],
      render(ctx, W, H, t, pal, inst) {
        const gpu = GPUS[inst.gpuKey];
        const cfg = inst.cfg;
        const ev = evaluateConfig(cfg, gpu);
        const { BM, BN, BK, WM, WN, TM, TN } = cfg;

        inst.setCaption(!ev.valid
          ? `INVALID CONFIG (${BM}×${BN}×${BK}, W=${WM}×${WN}, T=${TM}×${TN}) — Rejected by is_valid_config() before JIT compile!`
          : `VALID · HBM AI = ${ev.hbmAI.toFixed(1)} FLOP/B · ${ev.activeBlocksPerSm} blk/SM (${ev.occupancyPct.toFixed(0)}% occ, capped by ${ev.limiter}) · ${ev.waves.toFixed(2)} waves (2048³)`);

        const pad = 14, leftW = (W - pad * 3) * 0.46, rightW = (W - pad * 3) * 0.54, topY = 16, panelH = H - 30;
        drawBox(ctx, pad, topY, leftW, panelH, 10, pal.panel, ev.valid ? pal.smem : pal.danger, 1.5);
        drawFitText(ctx, `Tile Hierarchy: Block ${BM}×${BN} (BK=${BK})`, pad + 12, topY + 20, leftW - 24, '800 11.5px Inter, system-ui, sans-serif', ev.valid ? pal.smem : pal.danger, 'left');

        const maxDraw = Math.min(leftW - 40, panelH - 102);
        const scale = maxDraw / Math.max(BM, BN), drawW = BN * scale, drawH = BM * scale;
        const bx = pad + (leftW - drawW) / 2, by = topY + 34 + (maxDraw - drawH) / 2;
        drawBox(ctx, bx, by, drawW, drawH, 6, pal.bg, ev.valid ? pal.smem : pal.danger, 1.8);

        if (BM % WM === 0 && BN % WN === 0) {
          const wrC = BM / WM, wcC = BN / WN, ww = drawW / wcC, wh = drawH / wrC;
          const actW = Math.floor(t * 1.3) % (wrC * wcC);
          for (let wr = 0; wr < wrC; wr++) {
            for (let wc = 0; wc < wcC; wc++) {
              const wid = wr * wcC + wc, wx = bx + wc * ww, wy = by + wr * wh, isW0 = wid === 0;
              drawBox(ctx, wx + 1.5, wy + 1.5, ww - 3, wh - 3, 4, isW0 ? pal.warpBg : (wid === actW ? pal.smemBg : pal.cellEmpty), isW0 ? pal.warp : pal.cellStroke, isW0 ? 1.8 : 1);
              if (isW0 && WM % TM === 0 && WN % TN === 0) {
                const prC = WM / TM, pcC = WN / TN, pw = (ww - 4) / pcC, ph = (wh - 4) / prC;
                const pLane = Math.floor(t * 10) % (prC * pcC);
                for (let pr = 0; pr < prC; pr++) {
                  for (let pc = 0; pc < pcC; pc++) {
                    ctx.fillStyle = (pr * pcC + pc) === pLane ? pal.gold : pal.regBg;
                    ctx.fillRect(wx + 2 + pc * pw + 0.5, wy + 2 + pr * ph + 0.5, Math.max(2, pw - 1), Math.max(2, ph - 1));
                  }
                }
              }
            }
          }
        } else {
          drawFitText(ctx, `Warp ${WM}×${WN} does not`, bx + drawW / 2, by + drawH / 2 - 6, drawW - 12, '800 12px JetBrains Mono, monospace', pal.danger, 'center');
          drawFitText(ctx, `tile Block ${BM}×${BN}!`, bx + drawW / 2, by + drawH / 2 + 12, drawW - 12, '800 12px JetBrains Mono, monospace', pal.danger, 'center');
        }

        const warpWord = ev.warpsPerBlock === 1 ? 'warp' : 'warps';
        drawFitText(ctx, `NT = ${ev.NT} threads (${ev.warpsPerBlock} ${warpWord}) · AI = ${ev.hbmAI.toFixed(1)} F/B`, pad + leftW / 2, topY + panelH - 28, leftW - 20, '700 10px JetBrains Mono, monospace', pal.textSub, 'center');
        drawFitText(ctx, `Thread Patch = ${TM}×${TN} (${TM * TN} FMAs/k)`, pad + leftW / 2, topY + panelH - 12, leftW - 20, '700 10px JetBrains Mono, monospace', pal.reg, 'center');

        const rx = pad * 2 + leftW, topH = Math.min(184, panelH * 0.54);
        drawBox(ctx, rx, topY, rightW, topH, 10, pal.panel, ev.valid ? pal.reg : pal.danger, 1.3);
        drawFitText(ctx, ev.valid ? 'is_valid_config() — ALL 7 PASS' : 'is_valid_config() — REJECTED', rx + 12, topY + 18, rightW - 24, '800 11px Inter, system-ui, sans-serif', ev.valid ? pal.reg : pal.danger, 'left');
        ev.checks.forEach((chk, i) => {
          const cy = topY + 35 + i * 20.5;
          ctx.fillStyle = chk.ok ? pal.reg : pal.danger; ctx.font = '800 10.5px JetBrains Mono, monospace'; ctx.textAlign = 'left';
          ctx.fillText(chk.ok ? '✓' : '✕', rx + 12, cy);
          drawFitText(ctx, chk.label + (chk.detail ? ` [${chk.detail}]` : ''), rx + 26, cy, rightW - 36, '600 10px JetBrains Mono, monospace', chk.ok ? pal.textSub : pal.danger, 'left');
        });

        const by2 = topY + topH + 10, bh2 = panelH - topH - 10;
        drawBox(ctx, rx, by2, rightW, bh2, 10, pal.panel, pal.panelBorder, 1.2);
        drawFitText(ctx, `${gpu.name} SM Residency (${ev.activeBlocksPerSm} blk/SM · ${ev.occupancyPct.toFixed(0)}% occ)`, rx + 12, by2 + 18, rightW - 24, '800 11px Inter, system-ui, sans-serif', pal.text, 'left');
        const bars = [
          { label: `Regs (~${ev.estRegs}r/thr)`, val: ev.blocksByRegs, col: pal.reg },
          { label: `SMEM (${ev.smemKiB.toFixed(1)}K/blk)`, val: ev.blocksBySmem, col: pal.smem },
          { label: `Threads (${ev.NT}/blk)`, val: ev.blocksByThreads, col: pal.warp }
        ];
        const bSpace = (bh2 - 32) / 3;
        bars.forEach((b, i) => {
          const y = by2 + 28 + i * bSpace, lim = ev.valid && b.val === ev.activeBlocksPerSm;
          drawFitText(ctx, b.label, rx + 12, y + 9, rightW * 0.56, '700 9.5px JetBrains Mono, monospace', lim ? pal.hbm : pal.textSub, 'left');
          drawFitText(ctx, `${Math.min(b.val, gpu.maxBlocksPerSm)} blk${lim ? ' (CAP!)' : ''}`, rx + rightW - 12, y + 9, rightW * 0.4, '700 9.5px JetBrains Mono, monospace', lim ? pal.hbm : pal.textSub, 'right');
          const trackW = rightW - 24, fillW = Math.min(1, Math.min(b.val, gpu.maxBlocksPerSm) / gpu.maxBlocksPerSm) * trackW;
          drawBox(ctx, rx + 12, y + 14, trackW, 8, 4, pal.bg, null);
          drawBox(ctx, rx + 12, y + 14, Math.max(4, fillW), 8, 4, lim ? pal.hbm : b.col, null);
        });
      }
    },

    fusion: {
      title: '5. Epilogue Fusion · Eliminating 4 Full M×N HBM Round-Trips',
      hasGpuBar: true,
      toggles: [{ key: 'unfused', label: 'Show Unfused (3 Kernels · 5× HBM)', warn: true }],
      render(ctx, W, H, t, pal, inst) {
        const unfused = inst.toggles.unfused;
        const gpu = GPUS[inst.gpuKey];
        const tcRatio = (gpu.bf16Tflops / gpu.fp32Tflops).toFixed(0);
        const pulse = (t * 0.65) % 1, pad = 16;

        inst.setCaption(unfused
          ? 'UNFUSED: GEMM → HBM → Bias → HBM → GELU → HBM pays 5× M×N output HBM traffic + 3 launches!'
          : 'FUSED EPILOGUE: Bias + GELU execute directly on dot_products[TM, TN] in SM registers → 1 single HBM store!');

        const smY = 18, smH = H * 0.42, hbmY = smY + smH + 56, hbmH = H - hbmY - 14;
        drawBox(ctx, pad, smY, W - pad * 2, smH, 11, pal.panel, unfused ? pal.panelBorder : pal.reg, unfused ? 1.3 : 1.8);
        drawFitText(ctx, 'ON-CHIP SM (Registers & Execution Units — ~20+ TB/s effective)', pad + 14, smY + 20, W - pad * 2 - 28, '800 11.5px Inter, system-ui, sans-serif', pal.reg, 'left');
        drawBox(ctx, pad, hbmY, W - pad * 2, hbmH, 11, pal.panel, unfused ? pal.danger : pal.hbm, 1.5);

        const stageW = Math.min(180, (W - pad * 2 - 96) / 3);
        const gap = (W - pad * 2 - 32 - stageW * 3) / 2;
        const x1 = pad + 16, x2 = x1 + stageW + gap, x3 = x2 + stageW + gap;
        const boxY = smY + 34, boxH = smH - 48;

        if (unfused) {
          [
            { x: x1, name: 'Kernel 1: GEMM', sub: 'cuBLAS / Custom', col: pal.warp },
            { x: x2, name: 'Kernel 2: + Bias', sub: 'Elementwise CUDA', col: pal.hbm },
            { x: x3, name: 'Kernel 3: GELU', sub: 'Elementwise CUDA', col: pal.hbm }
          ].forEach(s => {
            drawBox(ctx, s.x, boxY, stageW, boxH, 8, pal.bg, s.col, 1.5);
            drawFitText(ctx, s.name, s.x + stageW / 2, boxY + boxH / 2 - 5, stageW - 14, '800 11.5px Inter, system-ui, sans-serif', s.col, 'center');
            drawFitText(ctx, s.sub, s.x + stageW / 2, boxY + boxH / 2 + 12, stageW - 14, '600 9.5px JetBrains Mono, monospace', pal.muted, 'center');
          });
          const bW = Math.min(130, stageW * 0.85), hBoxY = hbmY + 12, hBoxH = hbmH - 42;
          [
            { x: x1 + stageW * 0.65, label: 'Tmp C1 (M×N)', sub: 'Write + Re-read!' },
            { x: x2 + stageW * 0.65, label: 'Tmp C2 (M×N)', sub: 'Write + Re-read!' },
            { x: x3 + stageW * 0.12, label: 'Final Out (M×N)', sub: 'Write #3' }
          ].forEach((b, idx) => {
            const cx = Math.min(W - pad - bW - 14, b.x);
            drawBox(ctx, cx, hBoxY, bW, hBoxH, 7, pal.dangerBg, pal.danger, 1.3);
            drawFitText(ctx, b.label, cx + bW / 2, hBoxY + hBoxH / 2 - 4, bW - 12, '800 10px JetBrains Mono, monospace', pal.danger, 'center');
            drawFitText(ctx, b.sub, cx + bW / 2, hBoxY + hBoxH / 2 + 12, bW - 12, '600 9px Inter, system-ui, sans-serif', pal.textSub, 'center');
            if (idx === 0) {
              drawArrow(ctx, x1 + stageW * 0.72, boxY + boxH, cx + bW * 0.30, hBoxY, pal.danger, 2);
              drawArrow(ctx, cx + bW * 0.70, hBoxY, x2 + stageW * 0.28, boxY + boxH, pal.danger, 2);
              drawPacket(ctx, x1 + stageW * 0.72, boxY + boxH, cx + bW * 0.30, hBoxY, pulse, pal.danger, 4);
              drawPacket(ctx, cx + bW * 0.70, hBoxY, x2 + stageW * 0.28, boxY + boxH, (pulse + 0.5) % 1, pal.danger, 4);
            } else if (idx === 1) {
              drawArrow(ctx, x2 + stageW * 0.72, boxY + boxH, cx + bW * 0.30, hBoxY, pal.danger, 2);
              drawArrow(ctx, cx + bW * 0.70, hBoxY, x3 + stageW * 0.28, boxY + boxH, pal.danger, 2);
              drawPacket(ctx, x2 + stageW * 0.72, boxY + boxH, cx + bW * 0.30, hBoxY, (pulse + 0.25) % 1, pal.danger, 4);
              drawPacket(ctx, cx + bW * 0.70, hBoxY, x3 + stageW * 0.28, boxY + boxH, (pulse + 0.75) % 1, pal.danger, 4);
            } else {
              drawArrow(ctx, x3 + stageW * 0.5, boxY + boxH, cx + bW * 0.5, hBoxY, pal.danger, 2);
              drawPacket(ctx, x3 + stageW * 0.5, boxY + boxH, cx + bW * 0.5, hBoxY, (pulse + 0.4) % 1, pal.danger, 4);
            }
          });
          drawFitText(ctx, `OFF-CHIP ${gpu.memLabel} (${gpu.bwTbs} TB/s on ${gpu.name} — Every Round-Trip Pays ~500 Cycles)`, pad + 14, hbmY + hbmH - 10, W - pad * 2 - 28, '800 10.5px Inter, system-ui, sans-serif', pal.danger, 'left');
          drawBadge(ctx, W / 2, smY + smH + 17, '5× Output Tensor HBM Traffic (3 Writes + 2 Reads) + 3 Launches', pal.dangerBg, pal.danger, 'center', pal.bg);
        } else {
          drawFitText(ctx, `OFF-CHIP ${gpu.memLabel} (${gpu.bwTbs} TB/s on ${gpu.name} — Single Final Store Only)`, pad + 14, hbmY + 19, W - pad * 2 - 28, '800 11px Inter, system-ui, sans-serif', pal.hbm, 'left');
          drawBox(ctx, x1 - 4, boxY - 4, (x3 + stageW) - x1 + 8, boxH + 8, 10, pal.regBg, pal.reg, 1.8);
          const stages = [
            { x: x1 + 6, name: '1. GEMM Outer Prod', sub: 'dot_products[TM,TN]', col: pal.smem },
            { x: x2 + 3, name: '2. + Bias (In Regs!)', sub: '0 HBM bytes moved', col: pal.reg },
            { x: x3 - 6, name: '3. GELU (In Regs!)', sub: '0 HBM bytes moved', col: pal.reg }
          ];
          stages.forEach((s, idx) => {
            drawBox(ctx, s.x, boxY + 6, stageW - 6, boxH - 12, 7, pal.bg, s.col, 1.4);
            drawFitText(ctx, s.name, s.x + (stageW - 6) / 2, boxY + boxH / 2 - 5, stageW - 14, '800 11px Inter, system-ui, sans-serif', s.col, 'center');
            drawFitText(ctx, s.sub, s.x + (stageW - 6) / 2, boxY + boxH / 2 + 12, stageW - 14, '600 9.5px JetBrains Mono, monospace', pal.textSub, 'center');
            if (idx < 2) {
              drawArrow(ctx, s.x + stageW - 6, boxY + boxH / 2, stages[idx + 1].x, boxY + boxH / 2, pal.reg, 2);
              drawPacket(ctx, s.x + stageW - 6, boxY + boxH / 2, stages[idx + 1].x, boxY + boxH / 2, (pulse * 2 + idx * 0.4) % 1, pal.gold, 4);
            }
          });
          const outW = 198, outX = Math.min(W - pad - outW - 12, Math.max(pad + 12, x3 - 6 + (stageW - 6) / 2 - outW / 2));
          const hBoxY = hbmY + 30, hBoxH = hbmH - 42;
          drawBox(ctx, outX, hBoxY, outW, hBoxH, 7, pal.regBg, pal.reg, 1.7);
          drawFitText(ctx, 'Final matC (M × N)', outX + outW / 2, hBoxY + hBoxH / 2 - 11, outW - 16, '800 11.5px JetBrains Mono, monospace', pal.reg, 'center');
          drawFitText(ctx, 'SINGLE WRITE!', outX + outW / 2, hBoxY + hBoxH / 2 + 4, outW - 16, '800 10px JetBrains Mono, monospace', pal.reg, 'center');
          drawFitText(ctx, '1× HBM Traffic · 1 Launch', outX + outW / 2, hBoxY + hBoxH / 2 + 19, outW - 16, '600 9.5px Inter, system-ui, sans-serif', pal.textSub, 'center');
          const dropX = outX + outW / 2;
          drawArrow(ctx, dropX, boxY + boxH + 4, dropX, hBoxY, pal.reg, 2.2);
          drawPacket(ctx, dropX, boxY + boxH + 4, dropX, hBoxY, pulse, pal.reg, 4.6);

          const noteMaxW = outX - (pad + 16) - 14;
          drawFitText(ctx, 'What cuBLAS / Triton / FlashAttention Add Next:', pad + 16, hBoxY + 16, noteMaxW, '800 10.5px Inter, system-ui, sans-serif', pal.warp, 'left');
          drawFitText(ctx, `• ${gpu.tcDesc}: ${gpu.bf16Tflops.toFixed(0)} TF ${gpu.tcShort} (${tcRatio}× FP32)`, pad + 16, hBoxY + 34, noteMaxW, '600 10px Inter, system-ui, sans-serif', pal.textSub, 'left');
          drawFitText(ctx, '• TMA / cp.async double-buffering + XOR swizzled SMEM', pad + 16, hBoxY + 50, noteMaxW, '600 10px Inter, system-ui, sans-serif', pal.textSub, 'left');
          drawBadge(ctx, W / 2, smY + smH + 17, 'Fused Epilogue: 4 Full M×N HBM Round-Trips Deleted!', pal.regBg, pal.reg, 'center', pal.bg);
        }
      }
    }
  };

  /* =========================================================================
   * 6. Mount & Lifecycle Manager (IntersectionObserver + Single RAF Loop)
   * ========================================================================= */
  const instances = [];

  function mountFigure(el) {
    const sceneKey = el.getAttribute('data-scene');
    const spec = SCENES[sceneKey];
    if (!spec) return;

    const inst = {
      el, sceneKey, spec,
      playing: true,
      visible: true,
      simTime: 0,
      gpuKey: el.getAttribute('data-gpu') || 'h100',
      stepIdx: Number(el.getAttribute('data-step') || 3),
      toggles: {},
      cfg: { ...PRESETS.baseline },
      light: detectLightMode(),
      w: 680, h: 400
    };

    (spec.toggles || []).forEach(tg => {
      inst.toggles[tg.key] = el.getAttribute('data-initial-' + tg.key) === 'true';
    });

    el.innerHTML = '';
    el.setAttribute('data-theme', inst.light ? 'light' : 'dark');

    // Header bar
    const head = document.createElement('div');
    head.className = 'gemm-fig-head';
    const titleEl = document.createElement('span');
    titleEl.className = 'gemm-fig-title';
    titleEl.textContent = el.getAttribute('data-title') || spec.title;
    head.appendChild(titleEl);

    const ctrls = document.createElement('div');
    ctrls.className = 'gemm-fig-controls';

    (spec.toggles || []).forEach(tg => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'gemm-fig-btn' + (tg.warn ? ' warn' : '') + (inst.toggles[tg.key] ? ' active' : '');
      b.textContent = tg.label;
      b.addEventListener('click', () => {
        inst.toggles[tg.key] = !inst.toggles[tg.key];
        b.classList.toggle('active', inst.toggles[tg.key]);
      });
      ctrls.appendChild(b);
    });

    const playBtn = document.createElement('button');
    playBtn.type = 'button';
    playBtn.className = 'gemm-fig-btn';
    playBtn.textContent = 'Pause';
    ctrls.appendChild(playBtn);
    head.appendChild(ctrls);
    el.appendChild(head);

    // Optional Roofline step bar & separate GPU target bar (prevents orphan GPU button wrap!)
    if (spec.hasStepBar) {
      const sBar = document.createElement('div');
      sBar.className = 'gemm-fig-bar';
      ROOF_STEPS.forEach((s, idx) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'gemm-fig-btn' + (idx === inst.stepIdx ? ' active' : '');
        b.textContent = `${s.id} ${s.name.split(' ')[1]}`;
        b.addEventListener('click', () => {
          inst.stepIdx = idx;
          sBar.querySelectorAll('[data-stepbtn]').forEach((x, j) => x.classList.toggle('active', j === idx));
        });
        b.setAttribute('data-stepbtn', '1');
        sBar.appendChild(b);
      });
      el.appendChild(sBar);
    }
    if (spec.hasGpuBar) {
      const gBar = document.createElement('div');
      gBar.className = 'gemm-fig-bar';
      ['h100', 'a100', 't4'].forEach(gk => {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'gemm-fig-btn' + (gk === inst.gpuKey ? ' active' : '');
        b.textContent = GPUS[gk].name;
        b.setAttribute('data-gpubtn', gk);
        b.addEventListener('click', () => {
          inst.gpuKey = gk;
          gBar.querySelectorAll('[data-gpubtn]').forEach(x => x.classList.toggle('active', x.getAttribute('data-gpubtn') === gk));
        });
        gBar.appendChild(b);
      });
      el.appendChild(gBar);
    }

    // Optional 7-Knob Autotuner controls
    if (spec.hasAutotuneKnobs) {
      const pBar = document.createElement('div');
      pBar.className = 'gemm-fig-bar';
      [
        ['baseline', 'Default 64×64'],
        ['high_ai', 'Cross Ridge 128×128'],
        ['small_block', 'Small 32×32'],
        ['invalid_warp', 'Break Warp ≠32', true],
        ['invalid_fit', 'Break Tile Fit', true]
      ].forEach(([pk, lbl, warn]) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'gemm-fig-btn' + (warn ? ' warn' : '');
        b.textContent = lbl;
        b.addEventListener('click', () => {
          inst.cfg = { ...PRESETS[pk] };
          syncKnobs();
        });
        pBar.appendChild(b);
      });
      el.appendChild(pBar);

      const kWrap = document.createElement('div');
      kWrap.className = 'gemm-fig-bar';
      const kGrid = document.createElement('div');
      kGrid.className = 'gemm-fig-knobs';
      Object.keys(SEARCH_SPACE).forEach(k => {
        const box = document.createElement('div');
        box.className = 'gemm-fig-knob';
        box.innerHTML = `<div class="gemm-fig-knob-lbl"><span>${k}</span><span data-kval="${k}">${inst.cfg[k]}</span></div>`;
        const opts = document.createElement('div');
        opts.className = 'gemm-fig-kopts';
        SEARCH_SPACE[k].forEach(v => {
          const ob = document.createElement('button');
          ob.type = 'button';
          ob.className = 'gemm-fig-kopt';
          ob.textContent = String(v);
          ob.setAttribute('data-k', k);
          ob.setAttribute('data-v', String(v));
          ob.addEventListener('click', () => {
            inst.cfg[k] = v;
            syncKnobs();
          });
          opts.appendChild(ob);
        });
        box.appendChild(opts);
        kGrid.appendChild(box);
      });
      kWrap.appendChild(kGrid);
      el.appendChild(kWrap);

      function syncKnobs() {
        Object.keys(SEARCH_SPACE).forEach(k => {
          const ve = kGrid.querySelector(`[data-kval="${k}"]`);
          if (ve) ve.textContent = String(inst.cfg[k]);
        });
        kGrid.querySelectorAll('.gemm-fig-kopt').forEach(b => {
          b.classList.toggle('active', inst.cfg[b.getAttribute('data-k')] === Number(b.getAttribute('data-v')));
        });
      }
      syncKnobs();
    }

    // Canvas shell
    const wrap = document.createElement('div');
    wrap.className = 'gemm-fig-canvas-wrap';
    const canvas = document.createElement('canvas');
    canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label', titleEl.textContent);
    wrap.appendChild(canvas);
    el.appendChild(wrap);
    inst.canvas = canvas;
    inst.ctx = canvas.getContext('2d');
    inst.wrap = wrap;

    // Footer caption
    const foot = document.createElement('div');
    foot.className = 'gemm-fig-foot';
    const dot = document.createElement('span');
    dot.className = 'gemm-fig-dot';
    const cap = document.createElement('span');
    cap.textContent = 'Running simulation...';
    foot.appendChild(dot);
    foot.appendChild(cap);
    el.appendChild(foot);

    inst.setCaption = txt => { if (cap.textContent !== txt) cap.textContent = txt; };
    playBtn.addEventListener('click', () => {
      inst.playing = !inst.playing;
      playBtn.textContent = inst.playing ? 'Pause' : 'Play';
      dot.classList.toggle('paused', !inst.playing);
    });

    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = wrap.getBoundingClientRect();
      if (rect.width < 40) return;
      inst.w = Math.floor(rect.width);
      inst.h = Math.floor(rect.height);
      canvas.width = Math.floor(inst.w * dpr);
      canvas.height = Math.floor(inst.h * dpr);
      inst.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    inst.resize = resize;
    resize();
    if (window.ResizeObserver) new ResizeObserver(resize).observe(wrap);

    instances.push(inst);
  }

  function boot() {
    document.querySelectorAll('.gemm-fig[data-scene]').forEach(mountFigure);
    if (!instances.length) return;

    // Pause off-screen figures to keep scrolling silky smooth
    if (window.IntersectionObserver) {
      const io = new IntersectionObserver(entries => {
        entries.forEach(en => {
          const found = instances.find(i => i.el === en.target);
          if (found) found.visible = en.isIntersecting;
        });
      }, { rootMargin: '120px' });
      instances.forEach(i => io.observe(i.el));
    }

    // Watch theme mutations (Chirpy toggle or Bootstrap theme)
    function syncThemes() {
      const light = detectLightMode();
      instances.forEach(i => {
        i.light = light;
        i.el.setAttribute('data-theme', light ? 'light' : 'dark');
      });
    }
    if (window.MutationObserver) {
      new MutationObserver(syncThemes).observe(document.documentElement, {
        attributes: true, attributeFilter: ['data-mode', 'data-bs-theme', 'class']
      });
    }
    if (window.matchMedia) {
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', syncThemes);
    }

    let lastTs = 0;
    function tick(ts) {
      if (!lastTs) lastTs = ts;
      const dt = Math.min(0.05, (ts - lastTs) / 1000);
      lastTs = ts;
      instances.forEach(inst => {
        if (!inst.visible) return;
        if (inst.playing) inst.simTime += dt;
        const pal = getPal(inst.light);
        inst.ctx.fillStyle = pal.bg;
        inst.ctx.fillRect(0, 0, inst.w, inst.h);
        inst.spec.render(inst.ctx, inst.w, inst.h, inst.simTime, pal, inst);
      });
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
