// app.js - Main application logic

import { parseFile } from './parser.js';
import { Renderer } from './renderer.js';
import { autoSave, autoLoad, exportJSON, restoreState } from './storage.js';

// ── State ──────────────────────────────────────────────

const state = {
    fileName: null,
    pages: [],           // From parser: [{index, width, height, strokes}]
    currentPage: 0,
    classes: [
        { name: 'word', color: '#e74c3c' },
    ],
    activeClass: null,   // class name string
    annotations: [],     // Per-page: [[{id, className, strokeIndices: Set, text}]]
    selectedStrokes: new Set(),
    highlightedAnnotation: null,
    tool: 'rect',        // 'rect' | 'stroke' | 'pan'
    createdAt: null,
};

// ── DOM refs ───────────────────────────────────────────

const $ = (sel) => document.querySelector(sel);
const canvas = $('#canvas');
const container = $('#canvas-container');
const renderer = new Renderer(canvas);

// Toolbar
const fileInput = $('#file-input');
const fileNameSpan = $('#file-name');
const toolButtons = { rect: $('#tool-rect'), stroke: $('#tool-stroke'), pan: $('#tool-pan') };
const zoomLevel = $('#zoom-level');

// Sidebar left
const classList = $('#class-list');
const classNameInput = $('#class-name-input');
const classColorInput = $('#class-color-input');
const activeClassName = $('#active-class-name');
const wordInputSection = $('#word-input-section');
const nonWordSection = $('#non-word-assign-section');
const wordTextInput = $('#word-text-input');
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

    // File open
    $('#btn-open').addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileOpen);

    // Tools
    $('#tool-rect').addEventListener('click', () => setTool('rect'));
    $('#tool-stroke').addEventListener('click', () => setTool('stroke'));
    $('#tool-pan').addEventListener('click', () => setTool('pan'));

    // Zoom
    $('#btn-zoom-in').addEventListener('click', () => { renderer.zoom(1, container.clientWidth / 2, container.clientHeight / 2); updateZoom(); render(); });
    $('#btn-zoom-out').addEventListener('click', () => { renderer.zoom(-1, container.clientWidth / 2, container.clientHeight / 2); updateZoom(); render(); });
    $('#btn-zoom-fit').addEventListener('click', fitToView);

    // Canvas events
    canvas.addEventListener('mousedown', handleMouseDown);
    canvas.addEventListener('mousemove', onMouseMove);
    canvas.addEventListener('mouseup', onMouseUp);
    canvas.addEventListener('wheel', onWheel, { passive: false });
    canvas.addEventListener('contextmenu', e => e.preventDefault());

    // Class management
    $('#btn-add-class').addEventListener('click', addClass);
    classNameInput.addEventListener('keydown', e => { if (e.key === 'Enter') addClass(); });

    // Assignment
    $('#btn-assign-word').addEventListener('click', assignWord);
    $('#btn-assign-class').addEventListener('click', assignNonWord);
    wordTextInput.addEventListener('keydown', e => { if (e.key === 'Enter') assignWord(); });

    // Selection
    $('#btn-clear-selection').addEventListener('click', clearSelection);
    $('#btn-unassign').addEventListener('click', unassignSelected);

    // Pages
    $('#btn-prev-page').addEventListener('click', () => changePage(-1));
    $('#btn-next-page').addEventListener('click', () => changePage(1));

    // Export
    $('#btn-export').addEventListener('click', () => {
        if (state.fileName) exportJSON(state.fileName, state);
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', onKeyDown);

    // Set default active class
    state.activeClass = 'word';
    renderClassList();
    updateActiveClassUI();
    render();
}

// ── File handling ──────────────────────────────────────

async function handleFileOpen(e) {
    const file = e.target.files[0];
    if (!file) return;

    try {
        const doc = await parseFile(file);
        state.fileName = file.name;
        state.pages = doc.pages;
        state.currentPage = 0;
        state.selectedStrokes.clear();
        state.highlightedAnnotation = null;

        fileNameSpan.textContent = file.name;

        // Try to restore saved annotations
        const saved = autoLoad(file.name);
        const restored = restoreState(saved);
        if (restored) {
            state.classes = restored.classes;
            state.annotations = restored.annotations;
            state.createdAt = restored.createdAt;
            // Ensure annotations array length matches pages
            while (state.annotations.length < state.pages.length) {
                state.annotations.push([]);
            }
        } else {
            state.annotations = state.pages.map(() => []);
            state.createdAt = new Date().toISOString();
        }

        if (!state.activeClass && state.classes.length > 0) {
            state.activeClass = state.classes[0].name;
        }

        renderClassList();
        updateActiveClassUI();
        fitToView();
        updatePageNav();
        renderAnnotationList();
        render();
    } catch (err) {
        alert('Error loading file: ' + err.message);
        console.error(err);
    }

    // Reset input so same file can be re-opened
    fileInput.value = '';
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
        state.classes,
        selectionRect
    );
}

// ── Mouse events ───────────────────────────────────────

function onMouseDown(e) {
    if (e.button === 1 || (e.button === 0 && (state.tool === 'pan' || e.spaceKey))) {
        // Pan
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
                // Toggle individual stroke
                if (state.selectedStrokes.has(hitIdx)) {
                    state.selectedStrokes.delete(hitIdx);
                } else {
                    state.selectedStrokes.add(hitIdx);
                }
            } else {
                // Single select
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

    // Update cursor for stroke tool
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
        // Find strokes in rect
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
    // Don't intercept when typing in inputs
    if (e.target.tagName === 'INPUT') return;

    switch (e.key) {
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
        case ' ':
            e.preventDefault();
            break;
    }
}

// Track space key for pan
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

// ── Class management ───────────────────────────────────

function addClass() {
    const name = classNameInput.value.trim().toLowerCase();
    if (!name) return;
    if (state.classes.find(c => c.name === name)) {
        alert('Class "' + name + '" already exists.');
        return;
    }
    state.classes.push({ name, color: classColorInput.value });
    classNameInput.value = '';
    state.activeClass = name;
    renderClassList();
    updateActiveClassUI();
    save();
}

function deleteClass(className) {
    // Remove all annotations of this class
    for (let p = 0; p < state.annotations.length; p++) {
        state.annotations[p] = state.annotations[p].filter(a => a.className !== className);
    }
    state.classes = state.classes.filter(c => c.name !== className);
    if (state.activeClass === className) {
        state.activeClass = state.classes.length > 0 ? state.classes[0].name : null;
    }
    renderClassList();
    updateActiveClassUI();
    renderAnnotationList();
    save();
    render();
}

function renderClassList() {
    classList.innerHTML = '';
    for (const cls of state.classes) {
        const count = countClassAnnotations(cls.name);
        const div = document.createElement('div');
        div.className = 'class-item' + (state.activeClass === cls.name ? ' active' : '');
        div.innerHTML = `
            <span class="class-swatch" style="background:${cls.color}"></span>
            <span class="class-name">${cls.name}</span>
            <span class="class-count">${count}</span>
            <button class="class-delete" title="Delete class">&times;</button>
        `;
        div.addEventListener('click', (e) => {
            if (e.target.classList.contains('class-delete')) {
                if (confirm(`Delete class "${cls.name}" and all its annotations?`)) {
                    deleteClass(cls.name);
                }
                return;
            }
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
        const cls = state.classes.find(c => c.name === state.activeClass);
        activeClassName.textContent = state.activeClass;
        activeClassName.style.color = cls ? cls.color : '';

        if (state.activeClass === 'word') {
            wordInputSection.classList.remove('hidden');
            nonWordSection.classList.add('hidden');
        } else {
            wordInputSection.classList.add('hidden');
            nonWordSection.classList.remove('hidden');
        }
    } else {
        activeClassName.textContent = 'None selected';
        activeClassName.style.color = '';
        wordInputSection.classList.add('hidden');
        nonWordSection.classList.add('hidden');
    }
}

// ── Annotation assignment ──────────────────────────────

function assignWord() {
    const text = wordTextInput.value.trim();
    if (!text) {
        alert('Please enter the word text.');
        return;
    }
    if (state.selectedStrokes.size === 0) {
        alert('No strokes selected.');
        return;
    }
    assignStrokesToClass('word', text);
    wordTextInput.value = '';
    wordTextInput.focus();
}

function assignNonWord() {
    if (!state.activeClass) return;
    if (state.selectedStrokes.size === 0) {
        alert('No strokes selected.');
        return;
    }
    assignStrokesToClass(state.activeClass, '');
}

function assignStrokesToClass(className, text) {
    const pageAnns = state.annotations[state.currentPage];

    // Remove selected strokes from any existing annotations on this page
    for (const ann of pageAnns) {
        for (const si of state.selectedStrokes) {
            ann.strokeIndices.delete(si);
        }
    }
    // Clean up empty annotations
    state.annotations[state.currentPage] = pageAnns.filter(a => a.strokeIndices.size > 0);

    // Create new annotation
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
    save();
    render();
}

function unassignSelected() {
    if (state.selectedStrokes.size === 0) return;

    const pageAnns = state.annotations[state.currentPage];
    for (const ann of pageAnns) {
        for (const si of state.selectedStrokes) {
            ann.strokeIndices.delete(si);
        }
    }
    state.annotations[state.currentPage] = pageAnns.filter(a => a.strokeIndices.size > 0);

    state.selectedStrokes.clear();
    updateSelectionUI();
    renderAnnotationList();
    renderClassList();
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
        const cls = state.classes.find(c => c.name === ann.className);
        const div = document.createElement('div');
        div.className = 'annotation-item';
        if (state.highlightedAnnotation === ann.id) {
            div.classList.add('highlighted');
        }
        div.innerHTML = `
            <span class="annotation-swatch" style="background:${cls ? cls.color : '#888'}"></span>
            <span class="annotation-class">${ann.className}</span>
            <span class="annotation-text">${ann.text || ''}</span>
            <span class="annotation-strokes">${ann.strokeIndices.size}s</span>
            <button class="annotation-delete" title="Delete annotation">&times;</button>
        `;

        div.addEventListener('click', (e) => {
            if (e.target.classList.contains('annotation-delete')) {
                deleteAnnotation(ann.id);
                return;
            }
            // Highlight / select strokes
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

    // Stats
    const page = currentPage();
    const totalStrokes = page ? page.strokes.length : 0;
    const annotatedStrokes = new Set();
    for (const ann of pageAnns) {
        for (const si of ann.strokeIndices) annotatedStrokes.add(si);
    }
    statsTotal.textContent = `Annotations: ${pageAnns.length}`;
    statsUnannotated.textContent = `Unannotated: ${totalStrokes - annotatedStrokes.size} / ${totalStrokes} strokes`;
}

function deleteAnnotation(id) {
    state.annotations[state.currentPage] = state.annotations[state.currentPage].filter(a => a.id !== id);
    if (state.highlightedAnnotation === id) {
        state.highlightedAnnotation = null;
    }
    renderAnnotationList();
    renderClassList();
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

// ── Start ──────────────────────────────────────────────

init();
