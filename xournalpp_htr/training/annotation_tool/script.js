function readGZFile() {
    const fileInput = document.getElementById('fileInput');
    const output = document.getElementById('output');

    if (fileInput.files.length === 0) {
        output.textContent = "Please select a GZ file to upload.";
        return;
    }

    const file = fileInput.files[0];
    const reader = new FileReader();

    reader.onload = function(event) {
        try {
            const gzData = new Uint8Array(event.target.result);
            const decompressedData = pako.ungzip(gzData, { to: 'string' });
            
            if (isXML(decompressedData)) {
                const formattedXML = formatXML(decompressedData);
                output.textContent = formattedXML;
            } else {
                output.textContent = decompressedData;
            }
        } catch (e) {
            output.textContent = "An error occurred while reading the file: " + e.message;
        }
    };

    reader.onerror = function() {
        output.textContent = "Error reading file.";
    };

    reader.readAsArrayBuffer(file);

}


function isXML(data) {
    // Simple check to determine if the string looks like XML
    return data.trim().startsWith("<") && data.trim().endsWith(">");
}

function formatXML(xmlString) {
    try {
        const parser = new DOMParser();
        const xmlDoc = parser.parseFromString(xmlString, "application/xml");
        const serializer = new XMLSerializer();
        let formatted = serializer.serializeToString(xmlDoc);
        
        // Format the XML string with indentation
        formatted = formatted.replace(/(>)(<)(\/*)/g, '$1\n$2$3');
        const lines = formatted.split('\n');
        let indent = 0;
        for (let i = 0; i < lines.length; i++) {
            if (lines[i].match(/.+<\/\w[^>]*>$/)) {
                // No change in indent for self-closing tags
            } else if (lines[i].match(/^<\/\w/) && indent > 0) {
                indent--;
            }
            lines[i] = '  '.repeat(indent) + lines[i];
            if (lines[i].match(/^<\w([^>]*[^/])?>.*$/)) {
                indent++;
            }
        }
        return lines.join('\n');
    } catch (e) {
        return "Error parsing XML: " + e.message;
    }
}