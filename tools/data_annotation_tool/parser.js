// parser.js - Parse .xopp (gzipped XML) and .xoj (plain XML) files

export async function parseFile(file) {
    const buffer = await file.arrayBuffer();

    const sha256 = await computeSha256(buffer);

    let xmlString;
    try {
        xmlString = await decompressGzip(buffer);
    } catch {
        xmlString = new TextDecoder().decode(buffer);
    }

    const parser = new DOMParser();
    const doc = parser.parseFromString(xmlString, 'text/xml');

    const parserError = doc.querySelector('parsererror');
    if (parserError) {
        throw new Error('Failed to parse XML: ' + parserError.textContent);
    }

    const result = extractDocument(doc);
    result.sha256 = sha256;
    return result;
}

async function computeSha256(buffer) {
    const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

async function decompressGzip(buffer) {
    const ds = new DecompressionStream('gzip');
    const writer = ds.writable.getWriter();
    writer.write(new Uint8Array(buffer));
    writer.close();

    const reader = ds.readable.getReader();
    const chunks = [];
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
    }

    const totalLength = chunks.reduce((sum, c) => sum + c.length, 0);
    const result = new Uint8Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
        result.set(chunk, offset);
        offset += chunk.length;
    }

    return new TextDecoder().decode(result);
}

function extractDocument(doc) {
    const root = doc.documentElement;
    const pages = [];

    const pageElements = root.querySelectorAll('page');
    for (let pageIdx = 0; pageIdx < pageElements.length; pageIdx++) {
        const pageEl = pageElements[pageIdx];
        const width = parseFloat(pageEl.getAttribute('width') || '612');
        const height = parseFloat(pageEl.getAttribute('height') || '792');

        const strokes = [];
        let globalStrokeIdx = 0;

        // Each layer is tracked separately so stroke_indices in the schema
        // refer to positions within the layer, matching the source XML.
        const layers = pageEl.querySelectorAll('layer');
        for (let layerIdx = 0; layerIdx < layers.length; layerIdx++) {
            const layer = layers[layerIdx];
            const strokeElements = layer.querySelectorAll('stroke');
            for (let inLayerIdx = 0; inLayerIdx < strokeElements.length; inLayerIdx++) {
                const stroke = parseStroke(
                    strokeElements[inLayerIdx],
                    globalStrokeIdx,
                    layerIdx,
                    inLayerIdx
                );
                if (stroke) {
                    strokes.push(stroke);
                    globalStrokeIdx++;
                }
                // inLayerIdx always advances so it matches the XML element position,
                // even when a stroke element is skipped (no points).
            }
        }

        pages.push({ index: pageIdx, width, height, strokes });
    }

    return { pages };
}

function parseStroke(strokeEl, globalIndex, layerIndex, indexInLayer) {
    const tool = strokeEl.getAttribute('tool') || 'pen';
    const colorAttr = strokeEl.getAttribute('color') || 'black';
    const widthAttr = strokeEl.getAttribute('width') || '1.0';

    const widths = widthAttr.trim().split(/\s+/).map(Number);
    const baseWidth = widths[0];

    const text = strokeEl.textContent.trim();
    if (!text) return null;

    const values = text.split(/\s+/).map(Number);
    if (values.length < 4) return null;

    const points = [];
    for (let i = 0; i < values.length - 1; i += 2) {
        points.push({ x: values[i], y: values[i + 1] });
    }

    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of points) {
        if (p.x < minX) minX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.x > maxX) maxX = p.x;
        if (p.y > maxY) maxY = p.y;
    }

    return {
        index: globalIndex,    // flat index across all layers, used for selection
        layerIndex,            // which <layer> element within the page (0-based)
        indexInLayer,          // position within the layer's <stroke> elements (0-based)
        tool,
        color: resolveColor(colorAttr),
        width: baseWidth,
        widths: widths.length > 1 ? widths : null,
        points,
        bbox: { minX, minY, maxX, maxY },
    };
}

function resolveColor(colorAttr) {
    const named = {
        black: '#000000',
        blue: '#3333cc',
        red: '#ff0000',
        green: '#008000',
        gray: '#808080',
        lightblue: '#00c0ff',
        lightgreen: '#00ff00',
        magenta: '#ff00ff',
        orange: '#ff8000',
        yellow: '#ffff00',
        white: '#ffffff',
    };

    if (named[colorAttr]) return named[colorAttr];

    if (colorAttr.startsWith('#')) {
        return colorAttr.substring(0, 7);
    }

    if (colorAttr.startsWith('rgba') || colorAttr.startsWith('rgb')) {
        return colorAttr;
    }

    return '#000000';
}
