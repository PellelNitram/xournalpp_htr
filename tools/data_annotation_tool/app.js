// app.js - Main application logic

import { parseFile } from './parser.js';
import { Renderer } from './renderer.js';
import {
    autoSave, autoLoad, exportJSON, restoreState, loadGroundTruth,
    getAnnotatorId, saveAnnotatorId, TEXT_REQUIRED_CLASSES,
} from './storage.js';

// ── Fixed class vocabulary (closed, per ADR 004) ───────

const FIXED_CLASSES = [
    { name: 'word',                    color: '#e74c3c' },
    { name: 'digit',                   color: '#e67e22' },
    { name: 'mathematical_expression', color: '#f1c40f' },
    { name: 'arrow',                   color: '#2ecc71' },
    { name: 'diagram',                 color: '#1abc9c' },
    { name: 'table',                   color: '#3498db' },
    { name: 'drawing',                 color: '#9b59b6' },
    { name: 'separator',               color: '#95a5a6' },
    { name: 'correction',              color: '#e91e63' }, // pink — distinct from word's red
    { name: 'other',                   color: '#7f8c8d' },
];

// Abbreviated display names for long class names in compact list items.
const CLASS_ABBREV = {
    mathematical_expression: 'math_expr',
};

function classDisplayName(name) {
    return CLASS_ABBREV[name] || name;
}

// ── State ──────────────────────────────────────────────

const state = {
    fileName: null,
    sha256: null,
    pages: [],           // From parser: [{index, width, height, strokes}]
    currentPage: 0,
    activeClass: 'word',
    annotatorId: '',
    annotations: [],     // Per-page: [[{id, className, strokeIndices: Set, text}]]
    selectedStrokes: new Set(),
    highlightedAnnotation: null,
    tool: 'rect',        // 'rect' | 'stroke' | 'pan'
    createdAt: null,
};

// ── Undo stack ─────────────────────────────────────────

const undoStack = [];
const MAX_UNDO = 50;

function pushUndoState() {
    const snapshot = state.annotations.map(page =>
        page.map(ann => ({
            id: ann.id,
            className: ann.className,
            text: ann.text,
            strokeIndices: new Set(ann.strokeIndices),
        }))
    );
    undoStack.push(snapshot);
    if (undoStack.length > MAX_UNDO) undoStack.shift();
}

function undo() {
    if (undoStack.length === 0) return;
    state.annotations = undoStack.pop();
    state.selectedStrokes.clear();
    state.highlightedAnnotation = null;
    updateSelectionUI();
    renderAnnotationList();
    renderClassList();
    updateGlobalStats();
    save();
    render();
}

// ── DOM refs ───────────────────────────────────────────

const $ = (sel) => document.querySelector(sel);
const canvas = $('#canvas');
const container = $('#canvas-container');
const renderer = new Renderer(canvas);

// Toolbar
const fileInput = $('#file-input');
const fileNameSpan = $('#file-name');
const gtFileInput = $('#gt-file-input');
const toolButtons = { rect: $('#tool-rect'), stroke: $('#tool-stroke'), pan: $('#tool-pan') };
const zoomLevel = $('#zoom-level');
const annotatorIdInput = $('#annotator-id-input');

// Sidebar left
const classList = $('#class-list');
const activeClassName = $('#active-class-name');
const textInputSection = $('#text-input-section');
const nonTextSection = $('#non-text-assign-section');
const textInput = $('#text-input');
const selectionCount = $('#selection-count');

// Sidebar right
const annotationList = $('#annotation-list');
const statsTotal = $('#stats-total');
const statsUnannotated = $('#stats-unannotated');

// Page nav
const pageIndicator = $('#page-indicator');

// ── Interaction state ──────────────────────────────────

let isDragging = false;
let dragStart = { x: 0, y: 0 };
let selectionRect = null;
let isPanning = false;
let panLast = { x: 0, y: 0 };

// ── Init ───────────────────────────────────────────────

function init() {
    resizeCanvas();
    window.addEventListener('resize', () => { resizeCanvas(); render(); });

    // Annotator ID
    state.annotatorId = getAnnotatorId();
    annotatorIdInput.value = state.annotatorId;
    annotatorIdInput.addEventListener('input', () => {
        state.annotatorId = annotatorIdInput.value.trim();
        saveAnnotatorId(state.annotatorId);
        if (state.fileName) save();
    });

    // File open
    $('#btn-open').addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileOpen);

    // Load GT
    $('#btn-load-gt').addEventListener('click', () => {
        if (!state.sha256) {
            alert('Please open the source document (.xopp/.xoj) first.');
            return;
        }
        gtFileInput.click();
    });
    gtFileInput.addEventListener('change', handleLoadGT);

    // Tools
    $('#tool-rect').addEventListener('click', () => setTool('rect'));
    $('#tool-stroke').addEventListener('click', () => setTool('stroke'));
    $('#tool-pan').addEventListener('click', () => setTool('pan'));

    // Zoom
    $('#btn-zoom-in').addEventListener('click', () => {
        renderer.zoom(1, container.clientWidth / 2, container.clientHeight / 2);
        updateZoom(); render();
    });
    $('#btn-zoom-out').addEventListener('click', () => {
        renderer.zoom(-1, container.clientWidth / 2, container.clientHeight / 2);
        updateZoom(); render();
    });
    $('#btn-zoom-fit').addEventListener('click', fitToView);

    // Canvas events
    canvas.addEventListener('mousedown', handleMouseDown);
    canvas.addEventListener('mousemove', onMouseMove);
    canvas.addEventListener('mouseup', onMouseUp);
    canvas.addEventListener('wheel', onWheel, { passive: false });
    canvas.addEventListener('contextmenu', e => e.preventDefault());

    // Assignment
    $('#btn-assign-text').addEventListener('click', assignTextClass);
    $('#btn-assign-class').addEventListener('click', assignNonText);
    textInput.addEventListener('keydown', e => { if (e.key === 'Enter') assignTextClass(); });

    // Selection
    $('#btn-clear-selection').addEventListener('click', clearSelection);
    $('#btn-unassign').addEventListener('click', unassignSelected);

    // Pages
    $('#btn-prev-page').addEventListener('click', () => changePage(-1));
    $('#btn-next-page').addEventListener('click', () => changePage(1));

    // Export
    $('#btn-export').addEventListener('click', handleExport);

    // Keyboard shortcuts
    document.addEventListener('keydown', onKeyDown);

    renderClassList();
    updateActiveClassUI();
    updateGlobalStats();
    render();
}

// ── File handling ──────────────────────────────────────

async function handleFileOpen(e) {
    const file = e.target.files[0];
    if (!file) return;

    try {
        const doc = await parseFile(file);
        state.fileName = file.name;
        state.sha256 = doc.sha256;
        state.pages = doc.pages;
        state.currentPage = 0;
        state.selectedStrokes.clear();
        state.highlightedAnnotation = null;
        undoStack.length = 0;

        fileNameSpan.textContent = file.name;

        // Try to restore auto-saved annotations; discard if SHA-256 doesn't match.
        const saved = autoLoad(file.name);
        const restored = restoreState(saved, state.pages, state.sha256);
        if (restored) {
            state.annotations = restored.annotations;
            state.createdAt = restored.createdAt;
            if (restored.annotatorId) {
                state.annotatorId = restored.annotatorId;
                annotatorIdInput.value = restored.annotatorId;
                saveAnnotatorId(restored.annotatorId);
            }
        } else {
            state.annotations = state.pages.map(() => []);
            state.createdAt = new Date().toISOString();
        }

        // Ensure annotations array length matches pages
        while (state.annotations.length < state.pages.length) {
            state.annotations.push([]);
        }

        // Keep active class if valid, otherwise reset to 'word'
        if (!FIXED_CLASSES.find(c => c.name === state.activeClass)) {
            state.activeClass = 'word';
        }

        renderClassList();
        updateActiveClassUI();
        updateGlobalStats();
        fitToView();
        updatePageNav();
        renderAnnotationList();
        render();
        trackEvent('file-opened');
    } catch (err) {
        alert('Error loading file: ' + err.message);
        console.error(err);
    }

    fileInput.value = '';
}

async function handleLoadGT(e) {
    const file = e.target.files[0];
    if (!file) return;

    try {
        const restored = await loadGroundTruth(file, state.sha256, state.pages);
        state.annotations = restored.annotations;
        state.createdAt = restored.createdAt;
        undoStack.length = 0;
        if (restored.annotatorId) {
            state.annotatorId = restored.annotatorId;
            annotatorIdInput.value = restored.annotatorId;
            saveAnnotatorId(restored.annotatorId);
        }
        renderAnnotationList();
        renderClassList();
        updateGlobalStats();
        render();
        trackEvent('gt-loaded');
    } catch (err) {
        alert('Failed to load .gt.json:\n\n' + err.message);
    }

    gtFileInput.value = '';
}

// ── Export ─────────────────────────────────────────────

function handleExport() {
    if (!state.fileName) return;

    if (!state.annotatorId.trim()) {
        alert('Please enter your annotator ID before saving.');
        annotatorIdInput.focus();
        return;
    }

    const completenessProblems = checkCompleteness();
    if (completenessProblems.length > 0) {
        alert(
            'Cannot save: not all strokes are annotated.\n\n' + completenessProblems.join('\n') +
            '\n\nAnnotate every stroke before exporting.'
        );
        return;
    }

    const textProblems = checkTextPresence();
    if (textProblems.length > 0) {
        alert(
            'Cannot save: some transcription-class annotations have no text.\n\n' +
            textProblems.join('\n')
        );
        return;
    }

    try {
        exportJSON(state.fileName, state);
        trackEvent('annotation-exported');
    } catch (err) {
        alert('Export failed:\n\n' + err.message);
        console.error(err);
    }
}

function checkCompleteness() {
    const problems = [];
    for (let p = 0; p < state.pages.length; p++) {
        const page = state.pages[p];
        const annotated = new Set();
        for (const ann of state.annotations[p] || []) {
            for (const si of ann.strokeIndices) annotated.add(si);
        }
        const missing = page.strokes.length - annotated.size;
        if (missing > 0) {
            problems.push(`  Page ${p + 1}: ${missing} unannotated stroke(s)`);
        }
    }
    return problems;
}

function checkTextPresence() {
    const problems = [];
    for (let p = 0; p < state.annotations.length; p++) {
        for (const ann of state.annotations[p] || []) {
            if (TEXT_REQUIRED_CLASSES.has(ann.className) && !ann.text.trim()) {
                problems.push(`  Page ${p + 1}: "${ann.className}" annotation has no text`);
            }
        }
    }
    return problems;
}

// ── Global stats ───────────────────────────────────────

function updateGlobalStats() {
    const el = $('#global-progress');
    if (!el) return;

    if (state.pages.length === 0) {
        el.textContent = '–';
        el.classList.remove('complete');
        return;
    }

    let totalStrokes = 0;
    let annotatedStrokes = 0;
    for (let p = 0; p < state.pages.length; p++) {
        totalStrokes += state.pages[p].strokes.length;
        const annotated = new Set();
        for (const ann of state.annotations[p] || []) {
            for (const si of ann.strokeIndices) annotated.add(si);
        }
        annotatedStrokes += annotated.size;
    }

    el.textContent = `${annotatedStrokes} / ${totalStrokes} strokes`;
    el.classList.toggle('complete', totalStrokes > 0 && annotatedStrokes === totalStrokes);
}

// ── Canvas ─────────────────────────────────────────────

function resizeCanvas() {
    renderer.resize(container.clientWidth, container.clientHeight);
}

function fitToView() {
    const page = currentPage();
    if (!page) return;
    renderer.fitToView(page.width, page.height);
    updateZoom();
    render();
}

function updateZoom() {
    zoomLevel.textContent = renderer.getZoomPercent() + '%';
}

function render() {
    const page = currentPage();
    if (!page) {
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.fillStyle = '#151525';
        ctx.fillRect(0, 0, container.clientWidth, container.clientHeight);
        return;
    }

    const pageAnnotations = state.annotations[state.currentPage] || [];
    renderer.render(
        page.strokes,
        pageAnnotations,
        state.selectedStrokes,
        state.highlightedAnnotation,
        FIXED_CLASSES,
        selectionRect
    );
}

// ── Mouse events ───────────────────────────────────────

function onMouseDown(e) {
    if (e.button === 1 || (e.button === 0 && state.tool === 'pan')) {
        isPanning = true;
        panLast = { x: e.clientX, y: e.clientY };
        canvas.style.cursor = 'grabbing';
        return;
    }

    if (e.button !== 0) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (state.tool === 'rect') {
        isDragging = true;
        dragStart = { x, y };
        selectionRect = { x, y, w: 0, h: 0 };
    } else if (state.tool === 'stroke') {
        const page = currentPage();
        if (!page) return;
        const hitIdx = renderer.hitTestStroke(page.strokes, x, y, 8);
        if (hitIdx >= 0) {
            if (e.shiftKey) {
                if (state.selectedStrokes.has(hitIdx)) {
                    state.selectedStrokes.delete(hitIdx);
                } else {
                    state.selectedStrokes.add(hitIdx);
                }
            } else {
                state.selectedStrokes.clear();
                state.selectedStrokes.add(hitIdx);
            }
            updateSelectionUI();
            render();
        } else if (!e.shiftKey) {
            clearSelection();
        }
    }
}

function onMouseMove(e) {
    if (isPanning) {
        const dx = e.clientX - panLast.x;
        const dy = e.clientY - panLast.y;
        renderer.pan(dx, dy);
        panLast = { x: e.clientX, y: e.clientY };
        render();
        return;
    }

    if (isDragging && state.tool === 'rect') {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        selectionRect = {
            x: Math.min(dragStart.x, x),
            y: Math.min(dragStart.y, y),
            w: Math.abs(x - dragStart.x),
            h: Math.abs(y - dragStart.y),
        };
        render();
    }

    if (state.tool === 'stroke' && !isDragging) {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const page = currentPage();
        if (page) {
            const hitIdx = renderer.hitTestStroke(page.strokes, x, y, 8);
            canvas.style.cursor = hitIdx >= 0 ? 'pointer' : 'crosshair';
        }
    }
}

function onMouseUp(e) {
    if (isPanning) {
        isPanning = false;
        updateCursor();
        return;
    }

    if (isDragging && state.tool === 'rect' && selectionRect) {
        const page = currentPage();
        if (page && selectionRect.w > 3 && selectionRect.h > 3) {
            const indices = renderer.strokesInRect(page.strokes, selectionRect);
            if (e.shiftKey) {
                for (const idx of indices) state.selectedStrokes.add(idx);
            } else {
                state.selectedStrokes = new Set(indices);
            }
        }
        selectionRect = null;
        isDragging = false;
        updateSelectionUI();
        render();
    }
}

function onWheel(e) {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    renderer.zoom(-e.deltaY, x, y);
    updateZoom();
    render();
}

// ── Keyboard ───────────────────────────────────────────

let spaceDown = false;

function onKeyDown(e) {
    // Don't intercept typing in inputs or button activations
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'BUTTON') return;

    switch (e.key) {
        case 'z': case 'Z':
            if (e.ctrlKey || e.metaKey) { e.preventDefault(); undo(); }
            break;
        case 'r': setTool('rect'); break;
        case 's': setTool('stroke'); break;
        case 'p': setTool('pan'); break;
        case 'f': fitToView(); break;
        case '+': case '=':
            renderer.zoom(1, container.clientWidth / 2, container.clientHeight / 2);
            updateZoom(); render(); break;
        case '-':
            renderer.zoom(-1, container.clientWidth / 2, container.clientHeight / 2);
            updateZoom(); render(); break;
        case 'Escape':
            clearSelection(); break;
        case 'ArrowLeft':
            changePage(-1); break;
        case 'ArrowRight':
            changePage(1); break;
        case 'Delete': case 'Backspace':
            unassignSelected(); break;
        case 'Enter':
            // Assign selected strokes to the active non-text class
            if (!nonTextSection.classList.contains('hidden') && state.selectedStrokes.size > 0) {
                assignNonText();
            }
            break;
        case ' ':
            e.preventDefault();
            break;
    }
}

document.addEventListener('keydown', e => {
    if (e.code === 'Space' && !e.target.matches('input')) {
        spaceDown = true;
        canvas.style.cursor = 'grab';
    }
});
document.addEventListener('keyup', e => {
    if (e.code === 'Space') {
        spaceDown = false;
        updateCursor();
    }
});

function handleMouseDown(e) {
    if (spaceDown && e.button === 0) {
        isPanning = true;
        panLast = { x: e.clientX, y: e.clientY };
        canvas.style.cursor = 'grabbing';
        return;
    }
    onMouseDown(e);
}

// ── Tool ───────────────────────────────────────────────

function setTool(tool) {
    state.tool = tool;
    for (const [key, btn] of Object.entries(toolButtons)) {
        btn.classList.toggle('active', key === tool);
    }
    updateCursor();
}

function updateCursor() {
    switch (state.tool) {
        case 'rect': canvas.style.cursor = 'crosshair'; break;
        case 'stroke': canvas.style.cursor = 'crosshair'; break;
        case 'pan': canvas.style.cursor = 'grab'; break;
    }
}

// ── Class list (fixed vocabulary, selection only) ──────

function renderClassList() {
    classList.innerHTML = '';
    for (const cls of FIXED_CLASSES) {
        const count = countClassAnnotations(cls.name);
        const div = document.createElement('div');
        div.className = 'class-item' + (state.activeClass === cls.name ? ' active' : '');

        const swatch = document.createElement('span');
        swatch.className = 'class-swatch';
        swatch.style.background = cls.color;

        const nameSpan = document.createElement('span');
        nameSpan.className = 'class-name';
        nameSpan.textContent = cls.name;

        const countSpan = document.createElement('span');
        countSpan.className = 'class-count';
        countSpan.textContent = count;

        div.append(swatch, nameSpan, countSpan);
        div.addEventListener('click', () => {
            state.activeClass = cls.name;
            renderClassList();
            updateActiveClassUI();
        });
        classList.appendChild(div);
    }
}

function countClassAnnotations(className) {
    let count = 0;
    for (const pageAnns of state.annotations) {
        count += pageAnns.filter(a => a.className === className).length;
    }
    return count;
}

function updateActiveClassUI() {
    if (state.activeClass) {
        const cls = FIXED_CLASSES.find(c => c.name === state.activeClass);
        activeClassName.textContent = state.activeClass;
        activeClassName.style.color = cls ? cls.color : '';

        if (TEXT_REQUIRED_CLASSES.has(state.activeClass)) {
            textInputSection.classList.remove('hidden');
            nonTextSection.classList.add('hidden');
        } else {
            textInputSection.classList.add('hidden');
            nonTextSection.classList.remove('hidden');
        }
    } else {
        activeClassName.textContent = 'None selected';
        activeClassName.style.color = '';
        textInputSection.classList.add('hidden');
        nonTextSection.classList.add('hidden');
    }
}

// ── Annotation assignment ──────────────────────────────

function assignTextClass() {
    const text = textInput.value.trim();
    if (!text) {
        alert('Please enter the transcription text.');
        return;
    }
    if (state.selectedStrokes.size === 0) {
        alert('No strokes selected.');
        return;
    }
    assignStrokesToClass(state.activeClass, text);
    textInput.value = '';
    textInput.focus();
}

function assignNonText() {
    if (!state.activeClass) return;
    if (state.selectedStrokes.size === 0) {
        alert('No strokes selected.');
        return;
    }
    assignStrokesToClass(state.activeClass, '');
}

function assignStrokesToClass(className, text) {
    // Warn if selected strokes span multiple layers — they'll be split into separate
    // schema entries on export and restored as separate annotations on reload.
    const page = state.pages[state.currentPage];
    if (page) {
        const strokeByGlobal = new Map(page.strokes.map(s => [s.index, s]));
        const layers = new Set();
        for (const si of state.selectedStrokes) {
            const s = strokeByGlobal.get(si);
            if (s !== undefined) layers.add(s.layerIndex);
        }
        if (layers.size > 1) {
            if (!confirm(
                `The ${state.selectedStrokes.size} selected strokes span ${layers.size} layers.\n\n` +
                `They will be saved as ${layers.size} separate entries in the .gt.json (one per layer) ` +
                `and will be restored as separate annotations.\n\nContinue?`
            )) return;
        }
    }

    pushUndoState();

    const pageAnns = state.annotations[state.currentPage];

    // Remove selected strokes from any existing annotations on this page
    for (const ann of pageAnns) {
        for (const si of state.selectedStrokes) ann.strokeIndices.delete(si);
    }
    state.annotations[state.currentPage] = pageAnns.filter(a => a.strokeIndices.size > 0);

    state.annotations[state.currentPage].push({
        id: generateId(),
        className,
        strokeIndices: new Set(state.selectedStrokes),
        text: text || '',
    });

    state.selectedStrokes.clear();
    updateSelectionUI();
    renderAnnotationList();
    renderClassList();
    updateGlobalStats();
    save();
    render();
}

function unassignSelected() {
    if (state.selectedStrokes.size === 0) return;

    pushUndoState();

    const pageAnns = state.annotations[state.currentPage];
    for (const ann of pageAnns) {
        for (const si of state.selectedStrokes) ann.strokeIndices.delete(si);
    }
    state.annotations[state.currentPage] = pageAnns.filter(a => a.strokeIndices.size > 0);

    state.selectedStrokes.clear();
    updateSelectionUI();
    renderAnnotationList();
    renderClassList();
    updateGlobalStats();
    save();
    render();
}

// ── Selection UI ───────────────────────────────────────

function updateSelectionUI() {
    selectionCount.textContent = state.selectedStrokes.size + ' strokes selected';
}

function clearSelection() {
    state.selectedStrokes.clear();
    state.highlightedAnnotation = null;
    updateSelectionUI();
    render();
}

// ── Annotation list ────────────────────────────────────

function renderAnnotationList() {
    annotationList.innerHTML = '';
    const pageAnns = state.annotations[state.currentPage] || [];

    for (const ann of pageAnns) {
        const cls = FIXED_CLASSES.find(c => c.name === ann.className);
        const div = document.createElement('div');
        div.className = 'annotation-item';
        if (state.highlightedAnnotation === ann.id) div.classList.add('highlighted');

        const swatch = document.createElement('span');
        swatch.className = 'annotation-swatch';
        swatch.style.background = cls ? cls.color : '#888';

        const classSpan = document.createElement('span');
        classSpan.className = 'annotation-class';
        classSpan.textContent = classDisplayName(ann.className);
        classSpan.title = ann.className; // full name on hover

        const textSpan = document.createElement('span');
        const isTextRequired = TEXT_REQUIRED_CLASSES.has(ann.className);
        textSpan.className = 'annotation-text' + (isTextRequired ? ' editable' : '');
        textSpan.textContent = ann.text || (isTextRequired ? '(no text)' : '');
        if (isTextRequired) {
            textSpan.title = 'Click to edit transcription';
            textSpan.addEventListener('click', e => {
                e.stopPropagation();
                startInlineEdit(textSpan, ann);
            });
        }

        const strokesSpan = document.createElement('span');
        strokesSpan.className = 'annotation-strokes';
        strokesSpan.textContent = ann.strokeIndices.size + 's';

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'annotation-delete';
        deleteBtn.title = 'Delete annotation';
        deleteBtn.textContent = '×';
        deleteBtn.addEventListener('click', e => {
            e.stopPropagation();
            deleteAnnotation(ann.id);
        });

        div.append(swatch, classSpan, textSpan, strokesSpan, deleteBtn);

        div.addEventListener('click', () => {
            if (state.highlightedAnnotation === ann.id) {
                state.highlightedAnnotation = null;
                state.selectedStrokes.clear();
            } else {
                state.highlightedAnnotation = ann.id;
                state.selectedStrokes = new Set(ann.strokeIndices);
            }
            updateSelectionUI();
            renderAnnotationList();
            render();
        });

        annotationList.appendChild(div);
    }

    // Per-page stats
    const page = currentPage();
    const totalStrokes = page ? page.strokes.length : 0;
    const annotated = new Set();
    for (const ann of pageAnns) {
        for (const si of ann.strokeIndices) annotated.add(si);
    }
    statsTotal.textContent = `Annotations: ${pageAnns.length}`;
    statsUnannotated.textContent = `Unannotated: ${totalStrokes - annotated.size} / ${totalStrokes} strokes`;
}

function startInlineEdit(spanEl, ann) {
    const input = document.createElement('input');
    input.type = 'text';
    input.value = ann.text;
    input.className = 'annotation-text-edit';
    spanEl.replaceWith(input);
    input.focus();
    input.select();

    let done = false;

    const commit = () => {
        if (done) return;
        done = true;
        const newText = input.value.trim();
        if (newText && newText !== ann.text) {
            pushUndoState();
            ann.text = newText;
            save();
        }
        renderAnnotationList();
    };

    const cancel = () => {
        if (done) return;
        done = true;
        renderAnnotationList();
    };

    input.addEventListener('blur', commit);
    input.addEventListener('keydown', e => {
        e.stopPropagation(); // Prevent canvas keyboard shortcuts from firing
        if (e.key === 'Enter') { e.preventDefault(); commit(); }
        if (e.key === 'Escape') { e.preventDefault(); cancel(); }
    });
}

function deleteAnnotation(id) {
    pushUndoState();
    state.annotations[state.currentPage] = state.annotations[state.currentPage].filter(a => a.id !== id);
    if (state.highlightedAnnotation === id) {
        state.highlightedAnnotation = null;
    }
    renderAnnotationList();
    renderClassList();
    updateGlobalStats();
    save();
    render();
}

// ── Page navigation ────────────────────────────────────

function changePage(delta) {
    if (state.pages.length === 0) return;
    const newPage = state.currentPage + delta;
    if (newPage < 0 || newPage >= state.pages.length) return;
    state.currentPage = newPage;
    state.selectedStrokes.clear();
    state.highlightedAnnotation = null;
    updateSelectionUI();
    updatePageNav();
    renderAnnotationList();
    fitToView();
    render();
}

function updatePageNav() {
    pageIndicator.textContent = state.pages.length > 0
        ? `Page ${state.currentPage + 1} / ${state.pages.length}`
        : 'Page 0 / 0';
}

// ── Helpers ────────────────────────────────────────────

function currentPage() {
    return state.pages[state.currentPage] || null;
}

function generateId() {
    return Date.now().toString(36) + Math.random().toString(36).substring(2, 8);
}

function save() {
    if (state.fileName) {
        autoSave(state.fileName, state);
    }
}

// ── Analytics ──────────────────────────────────────────

function trackEvent(name) {
    if (window.goatcounter && window.goatcounter.count) {
        window.goatcounter.count({ path: name, event: true });
    }
}

// ── Start ──────────────────────────────────────────────

init();
