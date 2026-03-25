// renderer.js - Canvas rendering with pan/zoom and stroke display

export class Renderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.scale = 1;
        this.offsetX = 0;
        this.offsetY = 0;
        this.pageWidth = 0;
        this.pageHeight = 0;
        this.dpr = window.devicePixelRatio || 1;
    }

    resize(containerWidth, containerHeight) {
        this.canvas.width = containerWidth * this.dpr;
        this.canvas.height = containerHeight * this.dpr;
        this.canvas.style.width = containerWidth + 'px';
        this.canvas.style.height = containerHeight + 'px';
        this.containerWidth = containerWidth;
        this.containerHeight = containerHeight;
    }

    fitToView(pageWidth, pageHeight) {
        this.pageWidth = pageWidth;
        this.pageHeight = pageHeight;
        const padding = 40;
        const scaleX = (this.containerWidth - padding * 2) / pageWidth;
        const scaleY = (this.containerHeight - padding * 2) / pageHeight;
        this.scale = Math.min(scaleX, scaleY);
        this.offsetX = (this.containerWidth - pageWidth * this.scale) / 2;
        this.offsetY = (this.containerHeight - pageHeight * this.scale) / 2;
    }

    getZoomPercent() {
        return Math.round(this.scale * 100);
    }

    zoom(delta, centerX, centerY) {
        const oldScale = this.scale;
        const factor = delta > 0 ? 1.1 : 1 / 1.1;
        this.scale = Math.max(0.05, Math.min(20, this.scale * factor));

        // Zoom toward the cursor position
        const ratio = this.scale / oldScale;
        this.offsetX = centerX - (centerX - this.offsetX) * ratio;
        this.offsetY = centerY - (centerY - this.offsetY) * ratio;
    }

    pan(dx, dy) {
        this.offsetX += dx;
        this.offsetY += dy;
    }

    // Convert screen coordinates to document coordinates
    screenToDoc(sx, sy) {
        return {
            x: (sx - this.offsetX) / this.scale,
            y: (sy - this.offsetY) / this.scale,
        };
    }

    // Convert document coordinates to screen coordinates
    docToScreen(dx, dy) {
        return {
            x: dx * this.scale + this.offsetX,
            y: dy * this.scale + this.offsetY,
        };
    }

    render(strokes, annotations, selectedIndices, highlightedAnnotationId, classes, selectionRect) {
        const ctx = this.ctx;
        const dpr = this.dpr;

        // Clear
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.fillStyle = '#151525';
        ctx.fillRect(0, 0, this.containerWidth, this.containerHeight);

        // Draw page background
        ctx.save();
        ctx.translate(this.offsetX, this.offsetY);
        ctx.scale(this.scale, this.scale);

        ctx.fillStyle = '#ffffff';
        ctx.shadowColor = 'rgba(0,0,0,0.3)';
        ctx.shadowBlur = 10 / this.scale;
        ctx.fillRect(0, 0, this.pageWidth, this.pageHeight);
        ctx.shadowColor = 'transparent';

        // Build stroke-to-class map
        const strokeClassMap = new Map();
        const classMap = new Map();
        for (const cls of classes) {
            classMap.set(cls.name, cls);
        }
        for (const ann of annotations) {
            for (const si of ann.strokeIndices) {
                strokeClassMap.set(si, ann);
            }
        }

        // Draw strokes
        for (const stroke of strokes) {
            const ann = strokeClassMap.get(stroke.index);
            const isSelected = selectedIndices.has(stroke.index);
            const isHighlighted = ann && ann.id === highlightedAnnotationId;

            let color;
            let lineWidth = stroke.width;

            if (ann) {
                const cls = classMap.get(ann.className);
                color = cls ? cls.color : '#888888';
            } else {
                color = '#aaaaaa';
            }

            if (isSelected) {
                lineWidth = stroke.width * 1.5;
            }

            if (isHighlighted) {
                lineWidth = stroke.width * 2;
            }

            this.drawStroke(stroke, color, lineWidth, isSelected);
        }

        ctx.restore();

        // Draw selection rectangle (in screen coordinates)
        if (selectionRect) {
            ctx.save();
            ctx.strokeStyle = '#7c8ef5';
            ctx.lineWidth = 2;
            ctx.setLineDash([6, 3]);
            ctx.fillStyle = 'rgba(124, 142, 245, 0.1)';
            const { x, y, w, h } = selectionRect;
            ctx.fillRect(x, y, w, h);
            ctx.strokeRect(x, y, w, h);
            ctx.restore();
        }
    }

    drawStroke(stroke, color, lineWidth, isSelected) {
        const ctx = this.ctx;
        const points = stroke.points;
        if (points.length < 2) return;

        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
            ctx.lineTo(points[i].x, points[i].y);
        }
        ctx.stroke();

        // Draw selection highlight halo
        if (isSelected) {
            ctx.beginPath();
            ctx.strokeStyle = 'rgba(124, 142, 245, 0.4)';
            ctx.lineWidth = lineWidth + 4;
            ctx.moveTo(points[0].x, points[0].y);
            for (let i = 1; i < points.length; i++) {
                ctx.lineTo(points[i].x, points[i].y);
            }
            ctx.stroke();
        }
    }

    // Hit test: check if a screen point is near any stroke
    hitTestStroke(strokes, sx, sy, tolerance) {
        const doc = this.screenToDoc(sx, sy);
        const tolDoc = tolerance / this.scale;

        for (let i = strokes.length - 1; i >= 0; i--) {
            const stroke = strokes[i];
            // Quick bbox check
            if (doc.x < stroke.bbox.minX - tolDoc || doc.x > stroke.bbox.maxX + tolDoc ||
                doc.y < stroke.bbox.minY - tolDoc || doc.y > stroke.bbox.maxY + tolDoc) {
                continue;
            }
            // Detailed point-to-segment distance check
            for (let j = 0; j < stroke.points.length - 1; j++) {
                const dist = pointToSegmentDist(
                    doc.x, doc.y,
                    stroke.points[j].x, stroke.points[j].y,
                    stroke.points[j + 1].x, stroke.points[j + 1].y
                );
                if (dist < tolDoc + stroke.width / 2) {
                    return stroke.index;
                }
            }
        }
        return -1;
    }

    // Find all strokes whose bbox intersects the selection rectangle (in screen coords)
    strokesInRect(strokes, rectScreen) {
        const topLeft = this.screenToDoc(rectScreen.x, rectScreen.y);
        const bottomRight = this.screenToDoc(
            rectScreen.x + rectScreen.w,
            rectScreen.y + rectScreen.h
        );

        const selMinX = Math.min(topLeft.x, bottomRight.x);
        const selMinY = Math.min(topLeft.y, bottomRight.y);
        const selMaxX = Math.max(topLeft.x, bottomRight.x);
        const selMaxY = Math.max(topLeft.y, bottomRight.y);

        const result = [];
        for (const stroke of strokes) {
            // Check if stroke bbox center is inside selection rect
            const cx = (stroke.bbox.minX + stroke.bbox.maxX) / 2;
            const cy = (stroke.bbox.minY + stroke.bbox.maxY) / 2;
            if (cx >= selMinX && cx <= selMaxX && cy >= selMinY && cy <= selMaxY) {
                result.push(stroke.index);
            }
        }
        return result;
    }
}

function pointToSegmentDist(px, py, ax, ay, bx, by) {
    const dx = bx - ax;
    const dy = by - ay;
    const lenSq = dx * dx + dy * dy;
    if (lenSq === 0) return Math.hypot(px - ax, py - ay);

    let t = ((px - ax) * dx + (py - ay) * dy) / lenSq;
    t = Math.max(0, Math.min(1, t));

    const projX = ax + t * dx;
    const projY = ay + t * dy;
    return Math.hypot(px - projX, py - projY);
}
