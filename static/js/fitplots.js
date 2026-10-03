/*
 * fitplots.js — plain, publication-style plots for one fitted spectrum.
 * Shared by the Single-Spectrum Fitter (/) and the Profile Diagnostic Viewer (/v2).
 *
 * Graph 1  Full spectrum: observed flux, power-law continuum, continuum windows, fitted lines.
 * Graph 2  Line fit: data ± 1σ pixel noise, total model, local continuum, Gaussian components.
 * Graph 3  Normalised residuals (data − model)/σ with ±1σ and ±2σ reference lines.
 * Graph 4  Residual distribution compared with the standard normal N(0, 1).
 * Verification table: statistical checks returned by the server (src/refine.py).
 */
(function (global) {
  'use strict';

  let T = {
    paper: '#ffffff', plot: '#ffffff', ink: '#1a1a1a', muted: '#6b6b6b', grid: '#e6e6e6',
    data: '#1a1a1a', model: '#1f4e8c', cont: '#8a8a8a', comp: '#c0504d', shade: 'rgba(0,0,0,0.06)',
    bar: 'rgba(31,78,140,0.25)', pass: '#2e6b2e', fail: '#9b2c2c',
    font: 'Inter, Helvetica, Arial, sans-serif', mono: 'IBM Plex Mono, Menlo, monospace',
  };
  const CFG = { responsive: true, displayModeBar: false };
  const FLUX_UNIT = 'F<sub>λ</sub> (erg s⁻¹ cm⁻² Å⁻¹)';

  function setTheme(t) { T = Object.assign({}, T, t || {}); }

  function axis(title, extra) {
    return Object.assign({
      title: { text: title, font: { family: T.font, size: 11, color: T.ink } },
      showline: true, linecolor: T.ink, linewidth: 1, mirror: true, ticks: 'inside', ticklen: 4,
      tickcolor: T.ink, tickfont: { family: T.mono, size: 10, color: T.ink },
      gridcolor: T.grid, zeroline: false, exponentformat: 'e',
    }, extra || {});
  }

  function layout(xTitle, yTitle, extra) {
    return Object.assign({
      paper_bgcolor: T.paper, plot_bgcolor: T.plot,
      margin: { t: 10, r: 15, l: 70, b: 45 },
      font: { family: T.font, size: 11, color: T.ink },
      xaxis: axis(xTitle), yaxis: axis(yTitle),
      showlegend: true,
      legend: { orientation: 'h', x: 0, y: 1.02, yanchor: 'bottom', font: { size: 10, color: T.ink }, bgcolor: 'rgba(0,0,0,0)' },
      hovermode: 'closest',
    }, extra || {});
  }

  function message(el, text) {
    const node = typeof el === 'string' ? document.getElementById(el) : el;
    Plotly.purge(node);
    node.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:100%;
      font:12px ${T.font};color:${T.muted};text-align:center;padding:0 16px;">${text}</div>`;
  }

  function parseWindows(str) {
    return String(str || '').split(',').map(s => s.split(':').map(Number)).filter(w => w.length === 2 && w.every(isFinite));
  }

  /* Graph 1 */
  function overview(el, data) {
    const wl = data.wavelength || [], flux = data.observed_flux || [], cont = data.continuum_fit || [];
    if (!wl.length) return message(el, 'No spectrum data.');
    const good = wl.map((w, i) => flux[i] !== 0);
    const xs = wl.filter((_, i) => good[i]), ys = flux.filter((_, i) => good[i]);
    const traces = [
      { x: xs, y: ys, mode: 'lines', name: 'Observed', line: { color: T.data, width: 1 } },
      { x: wl, y: cont, mode: 'lines', name: `Power law F = A λ^α (α = ${fmt(data.spectral_index, 3)})`,
        line: { color: T.cont, width: 1.2, dash: 'dash' } },
    ];
    const shapes = parseWindows(data.continuum_windows).map(([a, b]) => ({
      type: 'rect', x0: a, x1: b, y0: 0, y1: 1, yref: 'paper', fillcolor: T.shade, line: { width: 0 }, layer: 'below',
    }));
    const ann = [];
    (data.lines || []).forEach(l => {
      if (!l.detected) return;
      shapes.push({ type: 'line', x0: l.center, x1: l.center, y0: 0, y1: 1, yref: 'paper',
        line: { color: T.model, width: 0.8, dash: 'dot' } });
      ann.push({ x: l.center, y: 1, yref: 'paper', yanchor: 'bottom', text: shortName(l.line_name),
        showarrow: false, textangle: -90, font: { size: 9, color: T.model } });
    });
    const sorted = ys.slice().sort((a, b) => a - b);
    const hi = sorted[Math.floor(sorted.length * 0.995)] || 1;
    Plotly.newPlot(el, traces, layout('Rest wavelength (Å)', FLUX_UNIT, {
      shapes, annotations: ann, margin: { t: 40, r: 15, l: 70, b: 45 },
      yaxis: axis(FLUX_UNIT, { range: [0, hi * 1.15] }),
      legend: { orientation: 'h', x: 0, y: -0.22, font: { size: 10, color: T.ink } },
    }), CFG);
  }

  /* Graph 2 */
  function lineFit(el, line) {
    const p = line && line.plot;
    if (!p) return message(el, 'No least-squares fit available for this line.');
    const err = p.x.map(() => p.yerr);
    const traces = [
      { x: p.x, y: p.y, mode: 'markers', name: 'Data ± 1σ', marker: { color: T.data, size: 4 },
        error_y: { type: 'data', array: err, visible: true, color: T.muted, thickness: 0.8, width: 0 } },
      { x: p.x, y: p.continuum, mode: 'lines', name: 'Local continuum', line: { color: T.cont, width: 1.2, dash: 'dash' } },
    ];
    p.components.forEach((c, k) => {
      traces.push({ x: p.x, y: c.y.map((v, i) => v + p.continuum[i]), mode: 'lines',
        name: k === 0 ? 'Gaussian (this line)' : c.label,
        line: { color: k === 0 ? T.comp : T.cont, width: 1, dash: k === 0 ? 'solid' : 'dot' } });
    });
    traces.push({ x: p.x, y: p.model, mode: 'lines', name: 'Total model', line: { color: T.model, width: 1.8 } });
    const shapes = (p.mask_ranges || []).map(([a, b]) => ({
      type: 'rect', x0: a, x1: b, y0: 0, y1: 1, yref: 'paper', fillcolor: T.shade, line: { width: 0 }, layer: 'below',
    }));
    const hw = (line.fwhm_ang || 0) / 2;
    const peak = (line.amplitude || 0) + interp(p.x, p.continuum, line.center);
    const half = (line.amplitude || 0) / 2 + interp(p.x, p.continuum, line.center);
    shapes.push({ type: 'line', x0: line.center, x1: line.center, y0: 0, y1: 1, yref: 'paper',
      line: { color: T.comp, width: 0.8, dash: 'dot' } });
    shapes.push({ type: 'line', x0: line.center - hw, x1: line.center + hw, y0: half, y1: half,
      line: { color: T.comp, width: 1.2 } });
    const ann = [{ x: line.center + hw, y: half, xanchor: 'left', text: ` FWHM ${fmt(line.fwhm_ang, 2)} Å`,
      showarrow: false, font: { size: 9, color: T.comp } },
      { x: line.center, y: peak, yanchor: 'bottom', text: `μ = ${fmt(line.center, 2)} ± ${fmt(line.center_err, 2)} Å`,
        showarrow: false, font: { size: 9, color: T.comp } }];
    Plotly.newPlot(el, traces, layout('Rest wavelength (Å)', FLUX_UNIT, {
      shapes, annotations: ann, margin: { t: 40, r: 15, l: 70, b: 45 },
    }), CFG);
  }

  /* Graph 3 */
  function residuals(el, line) {
    const p = line && line.plot;
    if (!p) return message(el, 'No residuals: line not refined.');
    const x0 = p.x[0], x1 = p.x[p.x.length - 1];
    const ref = (y, dash, name) => ({ x: [x0, x1], y: [y, y], mode: 'lines', name, showlegend: !!name,
      hoverinfo: 'skip', line: { color: T.muted, width: 0.8, dash } });
    const m = Math.max(3, ...p.norm_resid.map(Math.abs)) * 1.15;
    Plotly.newPlot(el, [
      ref(0, 'solid'), ref(1, 'dash', '±1σ'), ref(-1, 'dash'), ref(2, 'dot', '±2σ'), ref(-2, 'dot'),
      { x: p.x, y: p.norm_resid, mode: 'markers', name: '(data − model)/σ', marker: { color: T.data, size: 4 } },
    ], layout('Rest wavelength (Å)', '(F − F<sub>model</sub>) / σ', { yaxis: axis('(F − F<sub>model</sub>) / σ', { range: [-m, m] }) }), CFG);
  }

  /* Graph 4 */
  function residualDistribution(el, line) {
    const p = line && line.plot;
    if (!p) return message(el, 'No residuals: line not refined.');
    const r = p.norm_resid, n = r.length;
    const lim = Math.max(4, Math.ceil(Math.max(...r.map(Math.abs))));
    const xs = [], ys = [];
    for (let i = 0; i <= 200; i++) {
      const x = -lim + (2 * lim * i) / 200;
      xs.push(x); ys.push(Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI));
    }
    const mean = r.reduce((a, b) => a + b, 0) / n;
    const sd = Math.sqrt(r.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1));
    Plotly.newPlot(el, [
      { x: r, type: 'histogram', histnorm: 'probability density', name: `Residuals (N = ${n}, mean ${fmt(mean, 2)}, s.d. ${fmt(sd, 2)})`,
        xbins: { start: -lim, end: lim, size: 0.5 }, marker: { color: T.bar, line: { color: T.model, width: 1 } } },
      { x: xs, y: ys, mode: 'lines', name: 'N(0, 1)', line: { color: T.ink, width: 1.2 } },
    ], layout('(F − F<sub>model</sub>) / σ', 'Probability density', { bargap: 0.02,
      xaxis: axis('(F − F<sub>model</sub>) / σ', { range: [-lim, lim] }) }), CFG);
  }

  /* Graph 5: same-year light curve of one line, this spectrum highlighted */
  function lightCurve(el, ctxLine, ctx, thisFlux, thisErr) {
    const pts = (ctxLine && ctxLine.year_points) || [];
    if (!pts.length) return message(el, `No detected ${ctxLine ? ctxLine.line : 'line'} fluxes in ${ctx.year}.`);
    const jd0 = pts[0].jd;
    const others = pts.filter(p => p.spectrum !== ctx.spectrum);
    const traces = [{
      x: others.map(p => p.jd - jd0), y: others.map(p => p.flux), mode: 'markers', name: `${ctx.year} detections`,
      marker: { color: T.data, size: 5 },
      error_y: { type: 'data', array: others.map(p => p.flux_err), visible: true, color: T.muted, thickness: 0.8, width: 0 },
      text: others.map(p => p.spectrum), hovertemplate: '%{text}<br>JD−JD₀ %{x:.1f} d<br>F %{y:.3e}<extra></extra>',
    }];
    if (thisFlux != null) {
      traces.push({ x: [ctx.jd - jd0], y: [thisFlux], mode: 'markers', name: 'This spectrum',
        marker: { color: T.comp, size: 10, symbol: 'diamond', line: { color: T.ink, width: 1 } },
        error_y: { type: 'data', array: [thisErr || 0], visible: true, color: T.comp, thickness: 1.2, width: 4 } });
    }
    const fv = ctxLine.fvar;
    const shapes = fv && fv.f_mean != null ? [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: fv.f_mean, y1: fv.f_mean,
      line: { color: T.model, width: 1, dash: 'dash' } }] : [];
    const ann = fv && fv.f_mean != null ? [{ xref: 'paper', x: 1, y: fv.f_mean, xanchor: 'right', yanchor: 'bottom',
      text: `F̄ ${ctx.year}` + (fv.fvar != null ? `, F_var = ${fmt(fv.fvar, 3)} ± ${fmt(fv.fvar_err, 3)}` : ', not variable'),
      showarrow: false, font: { size: 9, color: T.model } }] : [];
    Plotly.newPlot(el, traces, layout(`JD − ${fmt(jd0, 2)} (days)`, 'F (erg s⁻¹ cm⁻²)', { shapes, annotations: ann }), CFG);
  }

  /* Verification table */
  function verification(container, line) {
    const node = typeof container === 'string' ? document.getElementById(container) : container;
    const v = line && line.verification;
    if (!v) { node.innerHTML = `<p style="font-size:12px;color:${T.muted};margin:6px 0;">No verification: line not refined.</p>`; return; }
    const ok = v.status === 'VERIFIED';
    const rows = v.checks.map(c => `<tr>
        <td>${esc(c.name)}${c.required ? '' : ' <span style="color:' + T.muted + '">(info)</span>'}</td>
        <td>${esc(c.value)}</td><td>${esc(c.criterion)}</td>
        <td style="font-weight:600;color:${c.pass ? T.pass : (c.required ? T.fail : T.muted)}">${c.pass ? 'pass' : (c.required ? 'fail' : 'note')}</td>
        <td style="color:${T.muted}">${esc(c.reference)}</td></tr>`).join('');
    node.innerHTML = `
      <div style="font-size:12px;margin:2px 0 8px;">
        <strong style="color:${ok ? T.pass : T.fail}">${ok ? 'VERIFIED' : 'CHECK'}</strong>
        ${ok ? '— all required checks pass.' : '— failed: ' + v.failed.map(esc).join(', ') + '. A single Gaussian may not fully describe this profile; values remain the best-fit estimates with their stated errors.'}
      </div>
      <table class="verif-tbl" style="width:100%;border-collapse:collapse;font-size:11px;">
        <thead><tr><th>Check</th><th>Value</th><th>Criterion</th><th>Result</th><th>Reference</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  }

  function interp(xs, ys, x) {
    if (!xs.length) return 0;
    if (x <= xs[0]) return ys[0];
    for (let i = 1; i < xs.length; i++) if (x <= xs[i]) {
      const t = (x - xs[i - 1]) / (xs[i] - xs[i - 1]); return ys[i - 1] + t * (ys[i] - ys[i - 1]);
    }
    return ys[ys.length - 1];
  }
  function fmt(v, d) { return v == null || !isFinite(v) ? '—' : Number(v).toFixed(d); }
  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
  function shortName(n) { return String(n).replace(/\s*\(.*\)/, ''); }

  global.FitPlots = { setTheme, overview, lineFit, residuals, residualDistribution, lightCurve, verification };
})(window);
