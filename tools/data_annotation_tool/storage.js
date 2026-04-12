// storage.js - localStorage auto-save and JSON export / import

const STORAGE_PREFIX = 'xopp-annotator:';
const ANNOTATOR_ID_KEY = 'xopp-annotator:annotator-id';

export const SCHEMA_VERSION = '1.0.0';

// Classes that require a text transcription (enforced by the schema).
export const TEXT_REQUIRED_CLASSES = new Set(['word', 'digit', 'mathematical_expression']);

// ── Annotator ID persistence ───────────────────────────

export function getAnnotatorId() {
    return localStorage.getItem(ANNOTATOR_ID_KEY) || '';
}

export function saveAnnotatorId(id) {
    localStorage.setItem(ANNOTATOR_ID_KEY, id);
}

// ── Auto-save (localStorage) ───────────────────────────

export function getStorageKey(fileName) {
    return STORAGE_PREFIX + fileName;
}

export function autoSave(fileName, state) {
    const key = getStorageKey(fileName);
    const data = buildExportData(fileName, state);
    try {
        localStorage.setItem(key, JSON.stringify(data));
    } catch (e) {
        console.warn('Auto-save failed:', e);
    }
}

export function autoLoad(fileName) {
    const key = getStorageKey(fileName);
    try {
        const raw = localStorage.getItem(key);
        if (!raw) return null;
        return JSON.parse(raw);
    } catch {
        return null;
    }
}

// ── Export ─────────────────────────────────────────────

/**
 * Build a ground_truth.schema.json-compliant object from app state.
 * Annotations are output as a flat array with page_index, layer_index,
 * and stroke_indices (positions within the layer, matching the source XML).
 */
export function buildExportData(fileName, state) {
    const annotations = [];

    for (let pageIdx = 0; pageIdx < state.annotations.length; pageIdx++) {
        const pageAnns = state.annotations[pageIdx] || [];
        const page = state.pages[pageIdx];
        if (!page) continue;

        const strokeByGlobal = new Map();
        for (const stroke of page.strokes) {
            strokeByGlobal.set(stroke.index, stroke);
        }

        for (const ann of pageAnns) {
            // Group selected strokes by layer so each schema entry has a single layer_index.
            const byLayer = new Map();
            for (const globalIdx of ann.strokeIndices) {
                const stroke = strokeByGlobal.get(globalIdx);
                if (!stroke) continue;
                const li = stroke.layerIndex;
                if (!byLayer.has(li)) byLayer.set(li, []);
                byLayer.get(li).push(stroke.indexInLayer);
            }

            for (const [layerIdx, strokeIndices] of byLayer) {
                const entry = {
                    class: ann.className,
                    page_index: pageIdx,
                    layer_index: layerIdx,
                    stroke_indices: strokeIndices.sort((a, b) => a - b),
                };
                if (TEXT_REQUIRED_CLASSES.has(ann.className) && ann.text) {
                    entry.text = ann.text;
                }
                annotations.push(entry);
            }
        }
    }

    return {
        schema_version: SCHEMA_VERSION,
        annotator_id: state.annotatorId || '',
        created_at: state.createdAt || new Date().toISOString(),
        source_document: {
            filename: fileName,
            sha256: state.sha256 || '',
        },
        annotations,
    };
}

export function exportJSON(fileName, state) {
    const data = buildExportData(fileName, state);
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    const baseName = fileName.replace(/\.(xopp|xoj)$/i, '');
    a.download = `${baseName}.gt.json`;
    a.click();

    URL.revokeObjectURL(url);
}

// ── Import / restore ───────────────────────────────────

/**
 * Reconstruct internal state (annotations as Sets of global stroke indices)
 * from a saved schema-format object.
 *
 * @param {object} savedData   Parsed JSON from localStorage or a .gt.json file.
 * @param {Array}  pages       Pages produced by parser.js (strokes have layerIndex/indexInLayer).
 * @param {string} [expectedSha256]  When provided, returns null if hash doesn't match.
 * @returns {{ annotations, createdAt, annotatorId } | null}
 */
export function restoreState(savedData, pages, expectedSha256 = null) {
    if (!savedData || savedData.schema_version !== SCHEMA_VERSION) return null;

    if (expectedSha256 && savedData.source_document?.sha256 !== expectedSha256) {
        console.warn('SHA-256 mismatch in auto-saved data — discarding stale annotations');
        return null;
    }

    // Build lookup: "pageIdx:layerIdx:indexInLayer" → globalStrokeIndex
    const strokeLookup = new Map();
    for (const page of pages) {
        for (const stroke of page.strokes) {
            const key = `${page.index}:${stroke.layerIndex}:${stroke.indexInLayer}`;
            strokeLookup.set(key, stroke.index);
        }
    }

    const annotations = pages.map(() => []);

    for (const ann of savedData.annotations) {
        if (ann.page_index >= pages.length) continue;
        const strokeIndices = new Set();
        for (const si of ann.stroke_indices) {
            const key = `${ann.page_index}:${ann.layer_index}:${si}`;
            const globalIdx = strokeLookup.get(key);
            if (globalIdx !== undefined) strokeIndices.add(globalIdx);
        }
        if (strokeIndices.size > 0) {
            annotations[ann.page_index].push({
                id: generateId(),
                className: ann.class,
                strokeIndices,
                text: ann.text || '',
            });
        }
    }

    return {
        annotations,
        createdAt: savedData.created_at,
        annotatorId: savedData.annotator_id || '',
    };
}

/**
 * Load and validate a .gt.json file.
 * Throws a descriptive Error if the file is invalid or the SHA-256 doesn't match.
 */
export async function loadGroundTruth(file, expectedSha256, pages) {
    const text = await file.text();
    let data;
    try {
        data = JSON.parse(text);
    } catch {
        throw new Error('Invalid JSON file.');
    }

    if (data.schema_version !== SCHEMA_VERSION) {
        throw new Error(
            `Unsupported schema version: "${data.schema_version}". Expected "${SCHEMA_VERSION}".`
        );
    }

    const fileSha256 = data.source_document?.sha256;
    if (fileSha256 !== expectedSha256) {
        throw new Error(
            `SHA-256 mismatch — the .gt.json was created from a different version of the source document.\n\n` +
            `Expected: ${expectedSha256}\n` +
            `In file:  ${fileSha256}`
        );
    }

    const restored = restoreState(data, pages);
    if (!restored) throw new Error('Failed to restore annotations from .gt.json.');
    return restored;
}

function generateId() {
    return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}
