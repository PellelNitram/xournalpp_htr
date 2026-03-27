// storage.js - localStorage auto-save and JSON export

const STORAGE_PREFIX = 'xopp-annotator:';
const SCHEMA_VERSION = '1.0.0';
const SCHEMA_ID = 'xopp-stroke-annotator/v1';

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

export function buildExportData(fileName, state) {
    return {
        $schema: SCHEMA_ID,
        version: SCHEMA_VERSION,
        metadata: {
            sourceFile: fileName,
            createdAt: state.createdAt || new Date().toISOString(),
            modifiedAt: new Date().toISOString(),
        },
        classes: state.classes.map(c => ({
            name: c.name,
            color: c.color,
        })),
        pages: state.pages.map((page, pageIdx) => ({
            pageIndex: pageIdx,
            totalStrokes: page.strokes.length,
            annotations: (state.annotations[pageIdx] || []).map(ann => {
                const obj = {
                    id: ann.id,
                    class: ann.className,
                    strokeIndices: [...ann.strokeIndices].sort((a, b) => a - b),
                };
                if (ann.className === 'word' && ann.text) {
                    obj.text = ann.text;
                }
                return obj;
            }),
        })),
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
    a.download = `${baseName}_annotations.json`;
    a.click();

    URL.revokeObjectURL(url);
}

export function restoreState(savedData) {
    if (!savedData || savedData.$schema !== SCHEMA_ID) return null;

    return {
        classes: savedData.classes.map(c => ({ name: c.name, color: c.color })),
        annotations: savedData.pages.map(page =>
            page.annotations.map(ann => ({
                id: ann.id,
                className: ann.class,
                strokeIndices: new Set(ann.strokeIndices),
                text: ann.text || '',
            }))
        ),
        createdAt: savedData.metadata.createdAt,
    };
}
