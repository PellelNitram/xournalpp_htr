// parser.js - Parse .xopp (gzipped XML) and .xoj (plain XML) files

export async function parseFile(file) {
    const buffer = await file.arrayBuffer();
    let xmlString;

    // Try to decompress as gzip first (xopp files)
    try {
        xmlString = await decompressGzip(buffer);
    } catch {
        // Not gzipped, treat as plain XML (xoj files)
        xmlString = new TextDecoder().decode(buffer);
    }

    const parser = new DOMParser();
    const doc = parser.parseFromString(xmlString, 'text/xml');

    const parserError = doc.querySelector('parsererror');
    if (parserError) {
        throw new Error('Failed to parse XML: ' + parserError.textContent);
    }

    return extractDocument(doc);
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
        let strokeIdx = 0;

        // Iterate all layers
        const layers = pageEl.querySelectorAll('layer');
        for (const layer of layers) {
            const strokeElements = layer.querySelectorAll('stroke');
            for (const strokeEl of strokeElements) {
                const stroke = parseStroke(strokeEl, strokeIdx);
                if (stroke) {
                    strokes.push(stroke);
                    strokeIdx++;
                }
            }
        }

        pages.push({ index: pageIdx, width, height, strokes });
    }

    return { pages };
}

function parseStroke(strokeEl, index) {
    const tool = strokeEl.getAttribute('tool') || 'pen';
    const colorAttr = strokeEl.getAttribute('color') || 'black';
    const widthAttr = strokeEl.getAttribute('width') || '1.0';

    // Parse widths (can be single or per-point)
    const widths = widthAttr.trim().split(/\s+/).map(Number);
    const baseWidth = widths[0];

    // Parse coordinate pairs from text content
    const text = strokeEl.textContent.trim();
    if (!text) return null;

    const values = text.split(/\s+/).map(Number);
    if (values.length < 4) return null; // Need at least 2 points

    const points = [];
    for (let i = 0; i < values.length - 1; i += 2) {
        points.push({ x: values[i], y: values[i + 1] });
    }

    // Compute bounding box
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of points) {
        if (p.x < minX) minX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.x > maxX) maxX = p.x;
        if (p.y > maxY) maxY = p.y;
    }

    return {
        index,
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

    // Handle hex format like #rrggbbaa or #rrggbb
    if (colorAttr.startsWith('#')) {
        // Take only RGB, ignore alpha
        return colorAttr.substring(0, 7);
    }

    // Handle rgba() format
    if (colorAttr.startsWith('rgba') || colorAttr.startsWith('rgb')) {
        return colorAttr;
    }

    return '#000000';
}
