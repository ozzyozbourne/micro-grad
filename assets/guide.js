// tinygrad study guide — shared runtime. Exposes window.Guide.
// Every chapter page loads: vendor/highlight.min.js, vendor/dagre.min.js, chapters.js, guide.js (in that order).
(function () {
  const G = {};
  window.Guide = G;

  // ---------- tiny DOM helpers ----------
  // G.el("div", {class:"x", onclick: fn, style:"..."}, "text", childNode, ...)
  G.el = function (tag, attrs, ...kids) {
    const svgTags = new Set(["svg", "g", "rect", "path", "text", "line", "circle", "polygon", "polyline", "defs", "marker", "tspan", "ellipse"]);
    const e = svgTags.has(tag) ? document.createElementNS("http://www.w3.org/2000/svg", tag) : document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === undefined || v === null) continue;
      if (k.startsWith("on") && typeof v === "function") e.addEventListener(k.slice(2), v);
      else if (k === "html") e.innerHTML = v;
      else e.setAttribute(k, v);
    }
    for (const k of kids.flat()) if (k !== null && k !== undefined) e.append(k instanceof Node ? k : document.createTextNode(String(k)));
    return e;
  };
  G.$ = (sel, root = document) => root.querySelector(sel);
  G.$$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  G.sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  G.ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
  // animate a value: G.tween(0, 1, 600, v => ...) returns a promise
  G.tween = function (from, to, ms, fn) {
    return new Promise((res) => {
      const t0 = performance.now();
      const tick = (t) => {
        const k = Math.min(1, (t - t0) / ms);
        fn(from + (to - from) * G.ease(k));
        if (k < 1) requestAnimationFrame(tick); else res();
      };
      requestAnimationFrame(tick);
    });
  };
  G.loadJSON = async (path) => (await fetch(path)).json();
  G.escape = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  G.src = (path, line, text) => `https://github.com/tinygrad/tinygrad/blob/${window.TG_COMMIT}/${path}${line ? "#L" + line : ""}`;

  // ---------- op colors (consistent across the whole guide) ----------
  const OP_GROUPS = {
    buffer: { color: "#F0AC5F", ops: ["BUFFER", "ALLOC", "PARAM", "COPY", "STAGE", "MSELECT", "MSTACK", "GETADDR", "CUSTOM_FUNCTION", "UNSHARD"] },
    movement: { color: "#5CD0B3", ops: ["RESHAPE", "PERMUTE", "EXPAND", "PAD", "SHRINK", "FLIP"] },
    alu: { color: "#58C4DD", ops: ["ADD", "MUL", "SUB", "MAX", "CMPLT", "CMPNE", "CMPEQ", "XOR", "OR", "AND", "SHL", "SHR", "TRUNCDIV", "TRUNCMOD", "THREEFRY", "FDIV", "POW", "DIV", "MOD", "WHERE", "MULACC", "EXP2", "LOG2", "SIN", "SQRT", "RECIP", "NEG", "TRUNC", "CAST", "BITCAST", "WMMA"] },
    reduce: { color: "#FC6255", ops: ["REDUCE", "ALLREDUCE"] },
    memory: { color: "#83C167", ops: ["LOAD", "STORE", "INDEX"] },
    control: { color: "#9A72AC", ops: ["RANGE", "END", "IF", "ENDIF", "BARRIER", "SPECIAL", "BACKEDGE", "AFTER", "SINK", "CALL", "LINEAR", "PROGRAM", "SOURCE", "BINARY", "STACK", "NOOP"] },
    const: { color: "#888888", ops: ["CONST", "DEFINE_VAR", "CUSTOM", "CUSTOMI", "INS", "DETACH", "CONTIGUOUS_BACKWARD", "PYLITERAL"] },
  };
  const OP_COLOR = {};
  for (const g of Object.values(OP_GROUPS)) for (const o of g.ops) OP_COLOR[o] = g.color;
  G.OP_GROUPS = OP_GROUPS;
  G.opColor = (op) => OP_COLOR[op] || "#bbbbbb";
  G.legend = function () {
    const names = { buffer: "buffers / storage", movement: "movement", alu: "ALU / elementwise", reduce: "reduce", memory: "load / store / index", control: "control / structure", const: "const / misc" };
    return G.el("div", { class: "legend" }, Object.entries(OP_GROUPS).map(([k, g]) => G.el("span", { style: `--c:${g.color}` }, names[k])));
  };

  // ---------- UOp graph renderer ----------
  // graph = {nodes:[{id, op, dtype?, arg?, shape?, label?, sub?}], edges:[[srcId, dstId, idx?]]}
  // opts = {rankdir:"BT"|"TB"|"LR", height:px, showArg:true, showDtype:true, onClick:(node)=>{}, legend:true, compact:false}
  // returns controller {setGraph(g, {animate}), highlight(ids|[], {dimOthers}), clear(), svg, nodesById}
  G.uopGraph = function (container, graph, opts = {}) {
    opts = Object.assign({ rankdir: "BT", showArg: true, showDtype: true, legend: true, compact: false }, opts);
    if (typeof container === "string") container = G.$(container);
    container.innerHTML = "";
    if (opts.legend) container.append(G.legend());
    const wrap = G.el("div", { class: "uopgraph" });
    if (opts.height) wrap.style.maxHeight = opts.height + "px";
    container.append(wrap);
    const svg = G.el("svg", {});
    wrap.append(svg);
    const edgeLayer = G.el("g", {}), nodeLayer = G.el("g", {});
    svg.append(edgeLayer, nodeLayer);
    let tip = null;
    const nodeEls = new Map();
    const ctl = { svg, nodesById: new Map(), graph: null };

    function labelLines(n) {
      const lines = [{ t: n.label || n.op, cls: "op" }];
      if (n.sub) lines.push({ t: n.sub, cls: "meta" });
      else {
        if (opts.showArg && n.arg !== undefined && n.arg !== "None" && n.arg !== "" && n.arg !== null) lines.push({ t: String(n.arg).slice(0, opts.compact ? 22 : 40), cls: "meta" });
        if (opts.showDtype && (n.dtype || n.shape)) lines.push({ t: [String(n.dtype || "").replace("dtypes.", ""), n.shape || ""].filter(Boolean).join(" ").slice(0, 40), cls: "meta" });
      }
      return lines;
    }
    function layout(g) {
      const dg = new dagre.graphlib.Graph({ multigraph: true });
      dg.setGraph({ rankdir: opts.rankdir, nodesep: opts.compact ? 14 : 22, ranksep: opts.compact ? 26 : 38, marginx: 16, marginy: 16 });
      dg.setDefaultEdgeLabel(() => ({}));
      for (const n of g.nodes) {
        const lines = labelLines(n);
        const w = Math.max(54, Math.max(...lines.map((l) => l.t.length * (l.cls === "op" ? 7.6 : 6.3))) + 18);
        dg.setNode(n.id, { width: w, height: 10 + lines.length * 14, lines });
      }
      g.edges.forEach((e, i) => { if (dg.hasNode(e[0]) && dg.hasNode(e[1])) dg.setEdge(e[0], e[1], { idx: e[2] }, "e" + i); });
      dagre.layout(dg);
      return dg;
    }
    function showTip(ev, n) {
      if (!tip) { tip = G.el("div", { class: "uop-tooltip" }); document.body.append(tip); }
      tip.textContent = `UOp(Ops.${n.op}` + (n.dtype ? `, dtype=${n.dtype}` : "") + (n.arg && n.arg !== "None" ? `,\n    arg=${n.arg}` : "") + `)` + (n.shape ? `\nshape: ${n.shape}` : "") + (n.note ? `\n\n${n.note}` : "");
      tip.style.display = "block"; tip.style.left = Math.min(ev.clientX + 14, innerWidth - 540) + "px"; tip.style.top = ev.clientY + 14 + "px";
    }
    function hideTip() { if (tip) tip.style.display = "none"; }

    ctl.setGraph = function (g, { animate = true } = {}) {
      ctl.graph = g;
      ctl.nodesById = new Map(g.nodes.map((n) => [n.id, n]));
      const dg = layout(g);
      const W = dg.graph().width || 100, H = dg.graph().height || 100;
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      svg.setAttribute("width", W); svg.setAttribute("height", H);
      // edges: re-render
      edgeLayer.innerHTML = "";
      for (const e of dg.edges()) {
        const pts = dg.edge(e).points;
        const d = "M" + pts.map((p) => `${p.x},${p.y}`).join(" L");
        const ge = G.el("g", { class: "edge", "data-src": e.v, "data-dst": e.w }, G.el("path", { d, "marker-end": "url(#arrow)" }));
        edgeLayer.append(ge);
        if (animate) { ge.style.opacity = 0; requestAnimationFrame(() => (ge.style.opacity = 1)); }
      }
      if (!svg.querySelector("defs")) svg.prepend(G.el("defs", {}, G.el("marker", { id: "arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse" }, G.el("path", { d: "M0,0 L10,5 L0,10 z", fill: "#5b6470" }))));
      // nodes: keyed by id, animate moves
      const seen = new Set();
      for (const id of dg.nodes()) {
        const nd = dg.node(id), n = ctl.nodesById.get(id);
        seen.add(id);
        let gEl = nodeEls.get(id);
        const x = nd.x - nd.width / 2, y = nd.y - nd.height / 2;
        const fresh = !gEl;
        if (gEl) gEl.remove();
        gEl = G.el("g", { class: "node", "data-id": id });
        const color = n.color || G.opColor(n.op);
        gEl.append(G.el("rect", { width: nd.width, height: nd.height, fill: color + "22", stroke: color }));
        nd.lines.forEach((l, i) => gEl.append(G.el("text", { x: nd.width / 2, y: 17 + i * 14, "text-anchor": "middle", class: l.cls, fill: l.cls === "op" ? color : undefined }, l.t)));
        gEl.addEventListener("mousemove", (ev) => showTip(ev, n));
        gEl.addEventListener("mouseleave", hideTip);
        if (opts.onClick) gEl.addEventListener("click", () => opts.onClick(n, ctl));
        const prev = nodeEls.get(id + "::pos");
        if (animate && prev && !fresh) { gEl.style.transform = `translate(${prev.x}px,${prev.y}px)`; requestAnimationFrame(() => requestAnimationFrame(() => (gEl.style.transform = `translate(${x}px,${y}px)`))); }
        else { gEl.style.transform = `translate(${x}px,${y}px)`; if (animate) { gEl.style.opacity = 0; requestAnimationFrame(() => requestAnimationFrame(() => (gEl.style.opacity = 1))); } }
        nodeLayer.append(gEl);
        nodeEls.set(id, gEl); nodeEls.set(id + "::pos", { x, y });
      }
      for (const k of Array.from(nodeEls.keys())) if (!k.endsWith("::pos") && !seen.has(k)) { nodeEls.get(k).remove(); nodeEls.delete(k); nodeEls.delete(k + "::pos"); }
      return ctl;
    };
    // highlight a set of node ids (and edges between them). dimOthers fades everything else.
    ctl.highlight = function (ids, { dimOthers = true } = {}) {
      const s = new Set(ids || []);
      G.$$(".node", svg).forEach((e) => { const on = s.has(e.dataset.id); e.classList.toggle("hl", on); e.classList.toggle("dim", dimOthers && s.size > 0 && !on); });
      G.$$(".edge", svg).forEach((e) => { const on = s.has(e.dataset.src) && s.has(e.dataset.dst); e.classList.toggle("hl", on); e.classList.toggle("dim", dimOthers && s.size > 0 && !on); });
      return ctl;
    };
    ctl.clear = () => ctl.highlight([]);
    ctl.setGraph(graph, { animate: false });
    return ctl;
  };

  // build a graph object by hand: G.mkGraph([["a","ADD",{arg,dtype,shape,sub,label}], ...], [["b","a",0], ...])
  G.mkGraph = function (nodes, edges) {
    return { nodes: nodes.map(([id, op, extra]) => Object.assign({ id, op }, extra || {})), edges: edges.map((e) => [e[0], e[1], e[2] ?? 0]) };
  };

  // ---------- stepper: prev/next/play through a list of steps ----------
  // G.stepper(container, steps, onStep, {autoplayMs:1800, loop:false})
  // steps: [{caption:"html", ...anything}] ; onStep(i, step) is called on every change.
  G.stepper = function (container, steps, onStep, opts = {}) {
    if (typeof container === "string") container = G.$(container);
    const ms = opts.autoplayMs || 1800;
    let i = 0, timer = null;
    const cap = G.el("div", { class: "step-caption" });
    const count = G.el("span", { class: "step-count" });
    const bar = G.el("div", {});
    const prev = G.el("button", { onclick: () => go(i - 1) }, "◀ prev");
    const next = G.el("button", { class: "primary", onclick: () => go(i + 1) }, "next ▶");
    const play = G.el("button", { onclick: () => toggle() }, "▶ play");
    const reset = G.el("button", { onclick: () => { stop(); go(0); } }, "⟲");
    const box = G.el("div", { class: "stepper" }, G.el("div", { class: "controls" }, reset, prev, next, play, count), G.el("div", { class: "progress" }, bar), cap);
    container.append(box);
    function go(j) {
      j = Math.max(0, Math.min(steps.length - 1, j)); i = j;
      cap.innerHTML = steps[i].caption || "";
      count.textContent = `step ${i + 1} / ${steps.length}`;
      bar.style.width = ((i + 1) / steps.length) * 100 + "%";
      prev.disabled = i === 0; next.disabled = i === steps.length - 1;
      onStep(i, steps[i]);
      if (window.renderMathInElement) renderMathInElement(cap);
    }
    function stop() { clearInterval(timer); timer = null; play.textContent = "▶ play"; }
    function toggle() {
      if (timer) return stop();
      play.textContent = "❚❚ pause";
      if (i === steps.length - 1) go(0);
      timer = setInterval(() => { if (i >= steps.length - 1) { if (opts.loop) go(0); else stop(); } else go(i + 1); }, ms);
    }
    go(0);
    return { go, get index() { return i; }, stop };
  };

  // ---------- pipeline strip: clickable stages ----------
  // G.pipeline(container, [{name, sub}], onSelect(i)) -> {select(i)}
  G.pipeline = function (container, stages, onSelect) {
    if (typeof container === "string") container = G.$(container);
    const row = G.el("div", { class: "pipeline" });
    const els = stages.map((s, i) => {
      const e = G.el("div", { class: "stage", onclick: () => select(i) }, s.name, s.sub ? G.el("small", {}, s.sub) : null);
      if (i) row.append(G.el("span", { class: "arrow" }, "→"));
      row.append(e); return e;
    });
    container.append(row);
    function select(i) { els.forEach((e, j) => e.classList.toggle("active", i === j)); onSelect && onSelect(i, stages[i]); }
    return { select };
  };

  // ---------- grid of cells (for index / memory layout visualizations) ----------
  // G.cells(container, rows, cols, (r,c)=>label) -> {cell(r,c), set(r,c,cls), clear(cls)}
  G.cells = function (container, rows, cols, label) {
    if (typeof container === "string") container = G.$(container);
    const g = G.el("div", { class: "grid-cells", style: `grid-template-columns: repeat(${cols}, 34px)` });
    const cells = [];
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) { const e = G.el("div", { class: "cell" }, label ? label(r, c) : ""); cells.push(e); g.append(e); }
    container.append(g);
    return {
      root: g,
      cell: (r, c) => cells[r * cols + c],
      set: (r, c, cls, on = true) => cells[r * cols + c].classList.toggle(cls, on),
      clear: (cls) => cells.forEach((e) => e.classList.remove(cls)),
      label: (r, c, t) => (cells[r * cols + c].textContent = t),
    };
  };

  // ---------- page chrome: sidebar, toc, prev/next, highlighting, copy buttons, quizzes, tabs ----------
  function buildSidebar() {
    const side = G.$("#sidebar");
    if (!side || !window.CHAPTERS) return;
    const here = location.pathname.split("/").pop().replace(".html", "");
    const inCh = location.pathname.includes("/chapters/");
    const base = inCh ? "../" : "";
    side.append(G.el("a", { class: "brand", href: base + "index.html", html: "tiny<span>grad</span> internals" }), G.el("div", { class: "sub" }, "an interactive study guide · commit " + window.TG_COMMIT.slice(0, 8)));
    for (const c of window.CHAPTERS) {
      if (c.part) { side.append(G.el("div", { class: "part" }, c.part)); continue; }
      const a = G.el("a", { class: "ch" + (c.slug === here ? " active" : ""), href: base + "chapters/" + c.slug + ".html" }, G.el("span", { class: "num" }, String(c.n).padStart(2, "0")), c.title);
      side.append(a);
      if (c.slug === here) {
        const toc = G.el("div", { class: "toc" });
        G.$$("main h2").forEach((h, i) => { if (!h.id) h.id = "s" + (i + 1); const sec = h.querySelector(".sec"); const label = sec ? sec.textContent.trim() + " " + h.textContent.slice(sec.textContent.length).trim() : h.textContent; toc.append(G.el("a", { href: "#" + h.id }, label)); });
        side.append(toc);
      }
    }
    // prev / next
    const list = window.CHAPTERS.filter((c) => !c.part);
    const idx = list.findIndex((c) => c.slug === here);
    const main = G.$("main");
    if (idx >= 0 && main) {
      const nav = G.el("div", { class: "chapter-nav" });
      const p = list[idx - 1], n = list[idx + 1];
      nav.append(p ? G.el("a", { href: p.slug + ".html" }, G.el("small", {}, "← previous"), `${p.n}. ${p.title}`) : G.el("span", { style: "flex:1" }));
      nav.append(n ? G.el("a", { class: "next", href: n.slug + ".html" }, G.el("small", {}, "next →"), `${n.n}. ${n.title}`) : G.el("span", { style: "flex:1" }));
      main.append(nav);
      document.title = `${list[idx].n}. ${list[idx].title} — tinygrad internals`;
    }
  }
  function decorateCode() {
    G.$$("pre > code").forEach((code) => {
      if (window.hljs && !code.classList.contains("nohl") && !code.closest("pre.output")) hljs.highlightElement(code);
      const pre = code.parentElement;
      const btn = G.el("button", { class: "copy-btn", onclick: () => { navigator.clipboard.writeText(code.innerText); btn.textContent = "copied"; setTimeout(() => (btn.textContent = "copy"), 1200); } }, "copy");
      pre.append(btn);
    });
  }
  // declarative quiz: <div class="quiz" data-answer="1"><div class="q">Question?</div><button class="opt">A</button><button class="opt">B</button><div class="explain">Why</div></div>
  function wireQuizzes() {
    G.$$(".quiz").forEach((qz) => {
      const ans = +qz.dataset.answer;
      G.$$(".opt", qz).forEach((b, i) => b.addEventListener("click", () => { qz.classList.add("done"); b.classList.add(i === ans ? "right" : "wrong"); G.$$(".opt", qz)[ans].classList.add("right"); }));
    });
  }
  // declarative tabs: <div class="tabs"><div class="tabpane" data-title="CPU">…</div><div class="tabpane" data-title="CUDA">…</div></div>
  function wireTabs() {
    G.$$(".tabs").forEach((t) => {
      const panes = G.$$(":scope > .tabpane", t);
      const bar = G.el("div", { class: "tabbar" });
      panes.forEach((p, i) => bar.append(G.el("button", { onclick: () => sel(i) }, p.dataset.title || "tab " + (i + 1))));
      t.prepend(bar);
      function sel(i) { panes.forEach((p, j) => p.classList.toggle("active", i === j)); G.$$("button", bar).forEach((b, j) => b.classList.toggle("active", i === j)); }
      sel(0);
    });
  }
  // <a class="src" data-path="tinygrad/uop/ops.py" data-line="228"></a> -> GitHub permalink at the pinned commit
  function wireSrcLinks() {
    G.$$("a.src[data-path]").forEach((a) => {
      a.href = G.src(a.dataset.path, a.dataset.line);
      a.target = "_blank";
      if (!a.textContent.trim()) a.textContent = a.dataset.path + (a.dataset.line ? ":" + a.dataset.line : "");
    });
  }
  document.addEventListener("DOMContentLoaded", () => {
    wireSrcLinks(); buildSidebar(); decorateCode(); wireQuizzes(); wireTabs();
    if (window.renderMathInElement) renderMathInElement(document.body, { delimiters: [{ left: "\\(", right: "\\)", display: false }, { left: "\\[", right: "\\]", display: true }] });
  });
})();
