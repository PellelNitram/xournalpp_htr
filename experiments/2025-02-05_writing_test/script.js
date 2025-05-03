document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const exportButton = document.getElementById('exportButton');

    let drawing = false;
    const strokes = [];

    canvas.addEventListener('mousedown', (e) => {
        drawing = true;
        const { offsetX, offsetY } = e;
        const time = new Date().toISOString();
        strokes.push({ x: offsetX, y: offsetY, time });
        ctx.beginPath();
        ctx.moveTo(offsetX, offsetY);
    });

    canvas.addEventListener('mousemove', (e) => {
        if (!drawing) return;
        const { offsetX, offsetY } = e;
        const time = new Date().toISOString();
        strokes.push({ x: offsetX, y: offsetY, time });
        ctx.lineTo(offsetX, offsetY);
        ctx.stroke();
    });

    canvas.addEventListener('mouseup', () => {
        drawing = false;
        ctx.closePath();
    });

    canvas.addEventListener('mouseleave', () => {
        drawing = false;
        ctx.closePath();
    });

    exportButton.addEventListener('click', () => {
        const json = JSON.stringify(strokes, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'strokes.json';
        a.click();
        URL.revokeObjectURL(url);
    });
});