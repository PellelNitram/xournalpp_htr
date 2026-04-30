// storage.js - localStorage auto-save and JSON export / import

const STORAGE_PREFIX = 'xopp-annotator:';
const ANNOTATOR_ID_KEY = 'xopp-annotator:annotator-id';

export const SCHEMA_VERSION = '1.0.0';

// Classes that require a text transcription (enforced by the schema).
export const TEXT_REQUIRED_CLASSES = new Set(['word', 'digit', 'mathematical_expression']);

const VALID_CLASSES = new Set([
    'word', 'digit', 'mathematical_expression', 'arrow', 'diagram',
    'table', 'drawing', 'separator', 'correction', 'other',
]);

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

// ── Schema validation ──────────────────────────────────

/**
 * Validate a ground_truth.schema.json-compliant object.
 * Covers all constraints expressed in the schema including the if/then/else
 * rule for the `text` field.
 *
 * @returns {{ valid: boolean, errors: string[] }}
 */
export function validateExportData(data) {
    const errors = [];

    if (!data || typeof data !== 'object') {
        return { valid: false, errors: ['Root value must be an object'] };
    }

    // Required top-level fields
    for (const field of ['schema_version', 'source_document', 'annotator_id', 'created_at', 'annotations']) {
        if (!(field in data)) errors.push(`Missing required field: "${field}"`);
    }
    if (errors.length) return { valid: false, errors };

    if (data.schema_version !== SCHEMA_VERSION) {
        errors.push(`schema_version must be "${SCHEMA_VERSION}", got "${data.schema_version}"`);
    }

    if (typeof data.annotator_id !== 'string' || data.annotator_id.length < 1) {
        errors.push('annotator_id must be a non-empty string');
    }

    if (typeof data.created_at !== 'string' || isNaN(Date.parse(data.created_at))) {
        errors.push('created_at must be a valid ISO 8601 date-time string');
    }

    if (!data.source_document || typeof data.source_document !== 'object') {
        errors.push('source_document must be an object');
    } else {
        if (!data.source_document.filename) {
            errors.push('source_document.filename is required');
        }
        if (!/^[a-f0-9]{64}$/.test(data.source_document.sha256 || '')) {
            errors.push('source_document.sha256 must be a 64-character lowercase hex string');
        }
    }

    if (!Array.isArray(data.annotations)) {
        errors.push('annotations must be an array');
    } else {
        for (let i = 0; i < data.annotations.length; i++) {
            const ann = data.annotations[i];
            const prefix = `annotations[${i}]`;

            for (const field of ['class', 'page_index', 'layer_index', 'stroke_indices']) {
                if (!(field in ann)) errors.push(`${prefix}: missing required field "${field}"`);
            }

            if (ann.class !== undefined && !VALID_CLASSES.has(ann.class)) {
                errors.push(`${prefix}: invalid class "${ann.class}"`);
            }

            if (typeof ann.page_index !== 'number' || !Number.isInteger(ann.page_index) || ann.page_index < 0) {
                errors.push(`${prefix}: page_index must be a non-negative integer`);
            }

            if (typeof ann.layer_index !== 'number' || !Number.isInteger(ann.layer_index) || ann.layer_index < 0) {
                errors.push(`${prefix}: layer_index must be a non-negative integer`);
            }

            if (!Array.isArray(ann.stroke_indices) || ann.stroke_indices.length === 0) {
                errors.push(`${prefix}: stroke_indices must be a non-empty array`);
            } else {
                const unique = new Set(ann.stroke_indices);
                if (unique.size !== ann.stroke_indices.length) {
                    errors.push(`${prefix}: stroke_indices must contain unique values`);
                }
            }

            // if/then/else: text is required for word/digit/mathematical_expression,
            // and forbidden for all other classes.
            if (ann.class && TEXT_REQUIRED_CLASSES.has(ann.class)) {
                if (typeof ann.text !== 'string' || ann.text.length === 0) {
                    errors.push(`${prefix}: class "${ann.class}" requires a non-empty "text" field`);
                }
            } else if (ann.class && !TEXT_REQUIRED_CLASSES.has(ann.class)) {
                if ('text' in ann) {
                    errors.push(`${prefix}: class "${ann.class}" must not have a "text" field`);
                }
            }
        }
    }

    return { valid: errors.length === 0, errors };
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

    const validation = validateExportData(data);
    if (!validation.valid) {
        throw new Error('Schema validation failed:\n\n' + validation.errors.join('\n'));
    }

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
 * @param {object} savedData          Parsed JSON from localStorage or a .gt.json file.
 * @param {Array}  pages              Pages produced by parser.js (strokes have layerIndex/indexInLayer).
 * @param {string} [expectedSha256]   When provided, returns null if hash doesn't match.
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

    for (let i = 0; i < savedData.annotations.length; i++) {
        const ann = savedData.annotations[i];
        if (ann.page_index >= pages.length) continue;

        const strokeIndices = new Set();
        for (const si of ann.stroke_indices) {
            const key = `${ann.page_index}:${ann.layer_index}:${si}`;
            const globalIdx = strokeLookup.get(key);
            if (globalIdx !== undefined) strokeIndices.add(globalIdx);
        }

        if (strokeIndices.size > 0) {
            const textValue = ann.text || '';
            if (TEXT_REQUIRED_CLASSES.has(ann.class) && !textValue) {
                console.warn(`annotations[${i}] (class "${ann.class}") is missing required text — loaded with empty text`);
            }
            annotations[ann.page_index].push({
                id: generateId(),
                className: ann.class,
                strokeIndices,
                text: textValue,
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
 * Throws a descriptive Error if the file is invalid, has schema violations,
 * or the SHA-256 doesn't match.
 */
export async function loadGroundTruth(file, expectedSha256, pages) {
    const text = await file.text();
    let data;
    try {
        data = JSON.parse(text);
    } catch {
        throw new Error('Invalid JSON file.');
    }

    // Full schema validation first
    const validation = validateExportData(data);
    if (!validation.valid) {
        throw new Error('Schema validation failed:\n\n' + validation.errors.join('\n'));
    }

    // Then check SHA-256 identity with the currently loaded document
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
