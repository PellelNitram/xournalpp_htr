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

                const strokes = getStrokesByPage(decompressedData);
                console.log(strokes);

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

function getStrokesByPage(xmlString) {

    const parser = new DOMParser();
    const xmlDoc = parser.parseFromString(xmlString, "application/xml");

    const pageElements = xmlDoc.getElementsByTagName('page');

    let data = [];

    for (let iPage = 0; iPage < pageElements.length; iPage++){

        const strokeElements = pageElements[iPage].getElementsByTagName('stroke');

        let xCoordinates = [];
        let yCoordinates = [];

        for (let i = 0; i < strokeElements.length; i++) {

            // Get the stroke value and split it by spaces into an array of coordinates
            let coordinates = strokeElements[i].textContent.trim().split(/\s+/);

            // Iterate through coordinates array, separating x and y values
            for (let j = 0; j < coordinates.length; j++) {
                if (j % 2 === 0) {
                    // Even index: x coordinate
                    xCoordinates.push(Number(coordinates[j]));
                } else {
                    // Odd index: y coordinate
                    yCoordinates.push(Number(coordinates[j]));
                }
            }
        }

        // Add the coordinates as tuples to the data structure that stores page-wise information
        let pageData = [];

        for (let k = 0; k < xCoordinates.length; k++) {
            pageData.push([xCoordinates[k], yCoordinates[k]]);
        }

        // Add the page data to the overall data array
        data.push(pageData);
    }

    // Return the data array containing all page-wise coordinates
    return data;
}
