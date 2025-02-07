document.addEventListener("DOMContentLoaded", function () {
    // Initialize WebGL context and canvas
    const canvas = document.getElementById("chart-canvas");
    const gl = canvas.getContext("webgl", { antialias: true }) || canvas.getContext("experimental-webgl");
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
    const tooltip = createTooltip();
  
    // Ensure WebGL is supported
    if (!gl) {
        console.error("WebGL not supported in this browser.");
        return;
    }
  
    // Set up the WebGL viewport and clear the canvas
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    gl.clearColor(0.95, 0.95, 0.95, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
  
    // Enable Blending for Transparency
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  
    // Vertex Shader (with u_viewMatrix)
    const vertexShaderSource = `
        attribute vec2 a_position;
        attribute float a_colorValue;
        uniform mat4 u_viewMatrix;
        uniform float u_pointScale;
        varying float v_colorValue;
        void main() {
            gl_Position = u_viewMatrix * vec4(a_position, 0, 1);
            gl_PointSize = 5.0 * u_pointScale; // Scale the point size
            v_colorValue = a_colorValue;
        }
    `;
  
    // Fragment Shader
    const fragmentShaderSource = `
        precision mediump float;
        varying float v_colorValue;
  
        void main() {
            vec3 color1 = vec3(103.0 / 255.0, 0.0 / 255.0, 31.0 / 255.0);   // #67001F
            vec3 color2 = vec3(142.0 / 255.0, 1.0 / 255.0, 82.0 / 255.0);   // #8E0152
            vec3 color3 = vec3(197.0 / 255.0, 27.0 / 255.0, 125.0 / 255.0); // #C51B7D
            vec3 color4 = vec3(178.0 / 255.0, 24.0 / 255.0, 43.0 / 255.0);  // #B2182B
            vec3 color5 = vec3(214.0 / 255.0, 96.0 / 255.0, 77.0 / 255.0);  // #D6604D
            vec3 color6 = vec3(67.0 / 255.0, 147.0 / 255.0, 195.0 / 255.0); // #4393C3
            vec3 color7 = vec3(33.0 / 255.0, 102.0 / 255.0, 172.0 / 255.0); // #2166AC
            vec3 color8 = vec3(5.0 / 255.0, 48.0 / 255.0, 97.0 / 255.0);    // #053061
        
            vec3 color;
            if (v_colorValue < 0.125) {
                color = mix(color1, color2, v_colorValue / 0.125);
            } else if (v_colorValue < 0.25) {
                color = mix(color2, color3, (v_colorValue - 0.125) / 0.125);
            } else if (v_colorValue < 0.375) {
                color = mix(color3, color4, (v_colorValue - 0.25) / 0.125);
            } else if (v_colorValue < 0.5) {
                color = mix(color4, color5, (v_colorValue - 0.375) / 0.125);
            } else if (v_colorValue < 0.625) {
                color = mix(color5, color6, (v_colorValue - 0.5) / 0.125);
            } else if (v_colorValue < 0.75) {
                color = mix(color6, color7, (v_colorValue - 0.625) / 0.125);
            } else if (v_colorValue < 0.875) {
                color = mix(color7, color8, (v_colorValue - 0.75) / 0.125);
            } else {
                color = color8;
            }
            float darkeningFactor = 0.9; // Reduce brightness
            color = color * darkeningFactor;
            gl_FragColor = vec4(color, 0.9);
        }
    `;

    // Utility function: Create a shader
    function createShader(gl, type, source) {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.error("Shader compilation failed:", gl.getShaderInfoLog(shader));
            gl.deleteShader(shader);
            return null;
        }
        return shader;
    }
  
    // Compile shaders and link them into a program
    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
  
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.error("Program linking failed.");
        return;
    }
  
    gl.useProgram(program);
  
    // Adjust Point Size in Relation to Viewport
    const pointScaleLocation = gl.getUniformLocation(program, "u_pointScale");
    if (pointScaleLocation === null) {
        console.error("Uniform 'u_pointScale' not found in shader program.");
        return;
    }
  
    const updatePointScale = () => {
        const pointScale = Math.min(gl.drawingBufferWidth, gl.drawingBufferHeight) / 1000; // Adjust scaling factor
        gl.uniform1f(pointScaleLocation, pointScale);
    };
    updatePointScale(); // Initial update
    window.addEventListener("resize", () => {
        canvas.width = canvas.clientWidth;
        canvas.height = canvas.clientHeight;
        gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
        updatePointScale();
        updateVisualization(); // Redraw the graph after resizing
    });
    
    // Global variables
    let responseData = {}; // Stores response data from the server
    let pcaData = {}; // Stores PCA data
    let umapData = {}; // Stores UMAP data
    let currentData = []; // Current PCA or UMAP data being visualized
    let currentAlgorithm = 'PCA'; // Can be 'PCA' or 'UMAP' based on user selection
    let pcaColorMap = null;
    let umapColorMap = null;
    let isDataFlyActive = false; // Tracks if DataFly is active
    let currentPairIndex = 0; // Current projection pair index for DataFly
    let dataFlyInterval; // Interval for DataFly
    let pauseDuration = 500; // Pause duration between transitions (1 seconds)
    let isHovered = false; // Track hover state
    let transitionSpeed = 1500; // Default speed (ms)
	let kdbushIndex;
    let zoomLevel = 1.0;
    let offsetX = 0.0;
    let offsetY = 0.0;

    const zoomSensitivity = 0.1; // Adjust this for zoom speed
    const panSensitivity = 0.01; // Adjust this for panning speed

    // Uniform location for view matrix
    const viewMatrixLocation = gl.getUniformLocation(program, "u_viewMatrix");

    // Update the view matrix in shaders
    function updateViewMatrix() {
        const viewMatrix = new Float32Array([
            zoomLevel, 0, 0, 0,
            0, zoomLevel, 0, 0,
            0, 0, 1, 0,
            offsetX, offsetY, 0, 1
        ]);
        gl.uniformMatrix4fv(viewMatrixLocation, false, viewMatrix);
    }

    // Add mouse scroll (zoom) event with passive option
    canvas.addEventListener("wheel", function (event) {
        event.preventDefault(); // Required for zoom handling, despite passive flag
        const delta = Math.sign(event.deltaY);
        zoomLevel *= 1 - delta * zoomSensitivity;
        zoomLevel = Math.max(0.1, Math.min(zoomLevel, 10)); // Clamp zoom level
        updateViewMatrix();
        drawGraph(
            currentData,
            currentAlgorithm === 'PCA' ? pcaColorMap : umapColorMap,
            currentAlgorithm === 'PCA' ? pcaData.edges : umapData.edges
        );
    }, { passive: true });

    // Reset zoom and pan to default on double-click
    canvas.addEventListener("dblclick", function () {
        // Reset zoom level and pan offsets
        zoomLevel = 1.0;
        offsetX = 0.0;
        offsetY = 0.0;

        // Update the view matrix to reflect the reset
        updateViewMatrix();

        // Redraw the graph with the current data and settings
        drawGraph(
            currentData,
            currentAlgorithm === 'PCA' ? pcaColorMap : umapColorMap,
            currentAlgorithm === 'PCA' ? pcaData.edges : umapData.edges
        );

        console.log("Zoom and pan reset to default.");
    });

    // Add panning with mouse drag
    let isDragging = false;
    let startX, startY;

    canvas.addEventListener("mousedown", function (event) {
        isDragging = true;
        startX = event.clientX;
        startY = event.clientY;
    });

    canvas.addEventListener("mousemove", function (event) {
        if (!isDragging) return;
        const dx = event.clientX - startX;
        const dy = event.clientY - startY;
        startX = event.clientX;
        startY = event.clientY;

        offsetX += dx * panSensitivity / zoomLevel;
        offsetY -= dy * panSensitivity / zoomLevel;

        updateViewMatrix();
        drawGraph(currentData, currentAlgorithm === 'PCA' ? pcaColorMap : umapColorMap,
            currentAlgorithm === 'PCA' ? pcaData.edges : umapData.edges);
    });

    canvas.addEventListener("mouseup", function () {
        isDragging = false;
    });

    canvas.addEventListener("mouseout", function () {
        isDragging = false;
    });

    // Helper to update the info message
    function updateInfoMessage(algorithm, projection1, projection2, metricValue, metricType) {
        const message = (algorithm === 'PCA')
            ? `Currently viewing: PC${projection1} vs PC${projection2} (Combined ${metricType} explained: ${metricValue.toFixed(2)}%).`
            : `Currently viewing: Projection ${projection1} vs Projection ${projection2} (Combined ${metricType}: ${metricValue.toFixed(2)}).`;
        document.getElementById("info-message").textContent = message;
    }

    // Helper to get projection data and metric
    function getProjectionData(algorithm, projection1, projection2) {
        if (algorithm === 'PCA') {
            return {
                data: pcaData[`pca${projection1}`].map((value, i) => [value, pcaData[`pca${projection2}`][i]]),
                metricValue: pcaData.explained_variance_percentage[projection1 - 1] +
                            pcaData.explained_variance_percentage[projection2 - 1],
                metricType: 'variance'
            };
        } else if (algorithm === 'UMAP') {
            return {
                data: umapData[`umap${projection1}`].map((value, i) => [value, umapData[`umap${projection2}`][i]]),
                metricValue: umapData.projection_densities[projection1 - 1] +
                            umapData.projection_densities[projection2 - 1],
                metricType: 'density'
            };
        }
        return null;
    }

    // Handle data upload
    const uploadForm = document.getElementById("data-upload-form");
    uploadForm.addEventListener("submit", async function (event) {
        event.preventDefault();
        reloadData();
    });
    
    // Helper function to reload data
    async function reloadData() {
        showLoadingSpinner(); // Show spinner

        // Get inputs
        const mainFileInput = document.getElementById("main-file-input");
        const edgeFileInput = document.getElementById("edge-file-input");
        const builtinFileSelect = document.getElementById("builtin-file-select");
        const selectedDemoFile = builtinFileSelect.value;
    
        const formData = new FormData();
    
        try {
            if (selectedDemoFile) {
                // If a demo file is selected, use it
                formData.append("builtin_file", selectedDemoFile);
            } else if (mainFileInput.files.length > 0) {
                // If no demo file is selected, use uploaded files
                formData.append("main_file", mainFileInput.files[0]);
    
                if (edgeFileInput.files.length > 0) {
                    formData.append("edge_file", edgeFileInput.files[0]);
                }
            } else {
                // No valid input
                alert("Please select a demo dataset or upload a main file.");
                return;
            }
    
            // Send the request to the server
            const response = await fetch("/upload_data", { method: "POST", body: formData });
    
            if (!response.ok) {
                const errorMessage = await response.text();
                throw new Error(`Server error: ${errorMessage}`);
            }
    
            responseData = await response.json();
            processDataResponse(responseData);
        } catch (error) {
            console.error("Error processing data:", error);
            alert("Failed to process the data. Please try again.");
        } finally {
            hideLoadingSpinner(); // Hide spinner
        }
    }

    // Pause DataFly when the algorithm dropdown is clicked (user interacting with algorithm selection)
    document.getElementById("algorithm-select").addEventListener("click", function () {
        if (isDataFlyActive) {
            stopDataFly(); // Pause DataFly
        }
    });

    // Reset the application when a new algorithm is selected
    document.getElementById("algorithm-select").addEventListener("change", function () {
        const selectedAlgorithm = this.value;

        console.log(`Current Algorithm: ${currentAlgorithm}`);

        if (selectedAlgorithm !== currentAlgorithm) {
            currentAlgorithm = selectedAlgorithm; // Update current algorithm
            // Different algorithm: Reset canvas, sliders, and DataFly
            console.log(`Switching to new algorithm: ${selectedAlgorithm}`);
            stopDataFlyAndReset(); // Stop DataFly and reset state
            reloadData();
            currentPairIndex = 0; // Reset to the first pair

            // Reset sliders and update visualization for the new algorithm
            const numComponents = currentAlgorithm === 'PCA' ? pcaData.num_components : umapData.num_components;
            setupSliders(numComponents, true); // Pass `true` to reset sliders
            
            // Automatically trigger update to visualization
            updateVisualization(); // Render the default view for the new algorithm
        }
    });
    
    function stopDataFlyAndReset() {
        stopDataFly(); // Stop DataFly if active
        currentPairIndex = 0; // Reset the index to the first pair
        console.log("DataFly reset to default state.");
    
        // Immediately update the graph with the selected algorithm data
        resetCanvas();  // Clear canvas
        updateVisualization();  // Render the first pair of the new algorithm
    }   
    
    function resetCanvas() {
        gl.clear(gl.COLOR_BUFFER_BIT);  // Clear the previous drawing
    }
    
    // Process response data from the server
    function processDataResponse(responseData) {
        if (!responseData) {
            alert("No response data received. Please try again.");
            return;
        }

        pcaData = responseData.pcaData || {};
        umapData = responseData.umapData || {};

        // Update global variables with the response data
        if (currentAlgorithm === 'PCA') {
            pcaColorMap = pcaData.color_map;
            if (!pcaColorMap || pcaColorMap.length === 0) {
                pcaColorMap = new Array(pcaData.pca1.length).fill(0).map((_, i) => i / pcaData.pca1.length);
                console.error("No PCA color map provided! Created by default.");
            }
        } else if (currentAlgorithm === 'UMAP') {
            umapColorMap = umapData.color_map;
            if (!umapColorMap || umapColorMap.length === 0) {
                umapColorMap = new Array(umapData.umap1.length).fill(0).map((_, i) => i / umapData.umap1.length);
                console.error("No UMAP color map provided! Created by default.");
            }
        } else {
            console.error("No valid selection of algorithm!");
        }

        if (!Object.keys(pcaData).length && !Object.keys(umapData).length) {
            alert("No valid PCA or UMAP data received. Check your input.");
            return;
        }

        // Reset view matrix and update the visualization
        zoomLevel = 1.0; // Reset zoom
        offsetX = 0.0; // Reset pan
        offsetY = 0.0;
        updateViewMatrix(); // Reset view matrix
        updateVisualization();
    }

    // Update graph data, sliders, and draw the graph
    function updateVisualization() {
        // Reset canvas before updating
        resetCanvas();

        // Clear the previous data and set up the algorithm-specific data
        currentData = [];
        currentPairIndex = 0; // Reset projection pair index

        const projection1 = 1;
        const projection2 = 2;

        const { data, metricValue, metricType } = getProjectionData(currentAlgorithm, projection1, projection2);

        // Update the canvas with the new data
        currentData = data;
        updateInfoMessage(currentAlgorithm, projection1, projection2, metricValue, metricType);

        // Reset the sliders to the correct number of components for the new algorithm
        const numComponents = currentAlgorithm === 'PCA' ? pcaData.num_components : umapData.num_components;
        setupSliders(numComponents, true); // Reset sliders with the new component count

        // Draw the new graph
        drawGraph(currentData, currentAlgorithm === 'PCA' ? pcaColorMap : umapColorMap,
            currentAlgorithm === 'PCA' ? pcaData.edges : umapData.edges);
    }

    // Initialize and configure sliders
    function setupSliders(numComponents, resetToDefault = false) {
        const projection1Slider = document.getElementById("projection1-slider");
        const projection2Slider = document.getElementById("projection2-slider");

        projection1Slider.max = numComponents;
        projection2Slider.max = numComponents;

        if (resetToDefault) {
            projection1Slider.value = 1;
            projection2Slider.value = 2;
            updateSliderLabels(); 
        }

        // Stop DataFly and update labels when sliders are adjusted
        function stopDataFlyForSlider() {
            if (isDataFlyActive) {
                stopDataFly();
                dataFlyButton.textContent = "DataFly"; // Reset button label
            }
        }

        projection1Slider.addEventListener("input", stopDataFlyForSlider);
        projection2Slider.addEventListener("input", stopDataFlyForSlider);

        projection1Slider.addEventListener("input", updateSliderLabels);
        projection2Slider.addEventListener("input", updateSliderLabels);

        updateSliderLabels();
    }

    // Update slider labels with selected values
    function updateSliderLabels() {
        const projection1 = document.getElementById("projection1-slider").value;
        const projection2 = document.getElementById("projection2-slider").value;
        document.getElementById("projection1-value").textContent = isNaN(projection1) ? 'N/A' : projection1;
        document.getElementById("projection2-value").textContent = isNaN(projection2) ? 'N/A' : projection2;
    }

    // Normalize data to fit within WebGL coordinates
    function normalizeData(data) {
        const xMin = Math.min(...data.map(point => point[0]));
        const xMax = Math.max(...data.map(point => point[0]));
        const yMin = Math.min(...data.map(point => point[1]));
        const yMax = Math.max(...data.map(point => point[1]));

        const xRange = xMax - xMin || 1;
        const yRange = yMax - yMin || 1;
        const margin = 0.01;

        return data.map(point => [
            Math.min(Math.max(((point[0] - xMin) / xRange) * (2 - 2 * margin) - (1 - margin), -1.0), 1.0),
            Math.min(Math.max(((point[1] - yMin) / yRange) * (2 - 2 * margin) - (1 - margin), -1.0), 1.0)
        ]);
    }

    // Draw graph with nodes, colors, and optional edges
    function drawGraph(nodes, colorMap, edges = []) {
        const normalizedNodes = normalizeData(nodes);
        checkOutOfBounds(normalizedNodes);

        // Prepare position and color buffers
        const positions = new Float32Array(normalizedNodes.flat());
        const colors = new Float32Array(colorMap);

        // Configure WebGL attributes for position
        const positionLocation = gl.getAttribLocation(program, "a_position");
        const buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(positionLocation);
        gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

        // Configure WebGL attributes for color
        const colorLocation = gl.getAttribLocation(program, "a_colorValue");
        const colorBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(colorLocation);
        gl.vertexAttribPointer(colorLocation, 1, gl.FLOAT, false, 0, 0);

        // Clear and draw points
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.drawArrays(gl.POINTS, 0, nodes.length);

        // Draw edges if provided
        edges.forEach(edge => {
            if (edge.length === 2) {  // Ensure edge has exactly two indices
                const [startIdx, endIdx] = edge;
                if (normalizedNodes[startIdx] && normalizedNodes[endIdx]) {
                    drawEdge(normalizedNodes[startIdx], normalizedNodes[endIdx]);
                }
            }
        });

        	//Prepare points data for hover
        const pointsData = normalizedNodes.map(([x, y], i) => ({
          x,
          y,
          label: pcaData.labels ? pcaData.labels[i] : `Point ${i + 1}`,
        }));

        if (typeof kdbush === "undefined") {
          console.error("kdbush is not defined. Ensure it is loaded correctly.");
          return;
        }

        //Create Spatial index using kdbush
        kdbushIndex = kdbush(
          pointsData,
          (p) => p.x,
          (p) => p.y,
          64,
          Float64Array
        );

		hoverAndFindNearest(canvas, pointsData, tooltip, kdbushIndex);

    }

    // Draw a single edge between two points
    function drawEdge(start, end) {
        const linePositions = new Float32Array([start[0], start[1], end[0], end[1]]);
        const buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, linePositions, gl.STATIC_DRAW);
        gl.vertexAttribPointer(gl.getAttribLocation(program, "a_position"), 2, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.LINES, 0, 2);
    }

    // Handle the "Go" button for manual transitions
    const goButton = document.getElementById("go-button");

    goButton.addEventListener("click", function () {
        // Stop DataFly if it's active
        stopDataFly();

        const projection1 = parseInt(document.getElementById("projection1-slider").value);
        const projection2 = parseInt(document.getElementById("projection2-slider").value);

        // Fetch projection data and validate
        const { data: targetData, metricValue: combinedMetric, metricType } = getProjectionData(currentAlgorithm, projection1, projection2) || {};

        // If valid target data exists, proceed with smooth transition
        if (targetData) {
            updateInfoMessage(currentAlgorithm, projection1, projection2, combinedMetric, metricType);
            smoothTransition(currentData, targetData, currentAlgorithm === 'PCA' ? pcaColorMap : umapColorMap,
                currentAlgorithm === 'PCA' ? pcaData.edges : umapData.edges);
        } else {
            console.error("Invalid projection selection for transition.");
        }
    });      

    function smoothTransition(current, target, colors, edges) {
        const duration = transitionSpeed; // Transition duration (controlled by slider)
        const startTime = Date.now();

        function animate() {
            const elapsed = Date.now() - startTime;
            const t = Math.min(elapsed / duration, 1); // Normalized time

            // Interpolate between current and target data
            const interpolatedData = current.map((point, i) => [
                point[0] + (target[i][0] - point[0]) * t,
                point[1] + (target[i][1] - point[1]) * t,
            ]);

            // Draw the interpolated graph
            drawGraph(interpolatedData, colors, edges);

            if (t < 1) {
                requestAnimationFrame(animate); // Continue animating
            } else {
                // Update currentData after completing the transition
                currentData = target;
            }
        }

        animate();
    }

    // Reset the application
    const resetButton = document.getElementById("reset-button");
    resetButton.addEventListener("click", () => {
        console.log("Resetting...");
        location.reload();
    });

    // Handle the "DataFly" button functionality
    const dataFlyButton = document.getElementById("datafly-button");
    dataFlyButton.addEventListener("click", function () {
        if (isDataFlyActive) {
            stopDataFly();
        } else {
            startDataFly();
        }
    });

    // Pause DataFly when hovered on canvas
    canvas.addEventListener("mouseover", function () {
        if (isDataFlyActive) {
            isHovered = true;
            clearTimeout(dataFlyInterval); // Stop ongoing transition
            dataFlyButton.textContent = "Stop"; // Display Stop when hovered
            dataFlyButton.style.backgroundColor = "#64b0d9";
        }
    });

    // Resume DataFly on unhover
    canvas.addEventListener("mouseout", function () {
        if (isDataFlyActive && isHovered) {
            isHovered = false;
            startDataFly(); // Continue DataFly transitions
        }
    });

    // Start DataFly
    function startDataFly() {
        if (isHovered) return; // Prevent starting during hover
        isDataFlyActive = true;
        dataFlyButton.textContent = "DataFlying"; // Set button label to active
        dataFlyButton.style.backgroundColor = "#e0f7fa";
        transitionAndPause(); // Start DataFly sequence
    }

    // Transition and pause function
    function transitionAndPause() {
        if (!isDataFlyActive || isHovered) return;

        const [projection1, projection2] = (currentAlgorithm === 'PCA'
            ? pcaData.sorted_pairs[currentPairIndex]?.pair
            : umapData.sorted_pairs[currentPairIndex]?.pair) || [];
        
        const { data: targetData, metricValue: combinedMetric, metricType } = getProjectionData(currentAlgorithm, projection1, projection2) || {};

        if (targetData) {
            // Use currentData as the starting point and smoothly transition
            updateInfoMessage(currentAlgorithm, projection1, projection2, combinedMetric, metricType);
            smoothTransition(currentData, targetData, currentAlgorithm === 'PCA' ? pcaColorMap : umapColorMap,
                currentAlgorithm === 'PCA' ? pcaData.edges : umapData.edges);
    
            if (!isHovered) {
                currentPairIndex = (currentPairIndex + 1) % (currentAlgorithm === 'PCA' ? pcaData.sorted_pairs.length : umapData.sorted_pairs.length);
                dataFlyInterval = setTimeout(transitionAndPause, transitionSpeed + pauseDuration);
            }
        } else {
            console.warn("Invalid projection selection for DataFly transition.");
        }
    }

    // Stop DataFly
    function stopDataFly() {
        if (isDataFlyActive) {
            clearTimeout(dataFlyInterval); // Stop ongoing transitions
            isDataFlyActive = false; // Update DataFly status
            dataFlyButton.textContent = "DataFly"; // Reset button text
            dataFlyButton.style.backgroundColor = "#E0E0E0";
        }
    }

    // Handle transition speed control
    const speedSlider = document.getElementById("transition-speed");
    const speedValueDisplay = document.getElementById("speed-value");

    speedSlider.addEventListener("input", function () {
        transitionSpeed = parseInt(this.value, 10);
        speedValueDisplay.textContent = transitionSpeed;
        speedSlider.value = transitionSpeed;

        // Restart DataFly if active with new speed
        if (isDataFlyActive) {
            stopDataFly();
            startDataFly();
        }
    });   

  //hoverAndFindNearest function tracking mousemove and display tooltip with fast spacial query
  function hoverAndFindNearest(canvas, pointsData, tooltip, index) {
    canvas.addEventListener("mousemove", async (event) => {
      const rect = canvas.getBoundingClientRect();
      const mouseX = event.clientX - rect.left;
      const mouseY = event.clientY - rect.top;
      // Transform mouse coordinates to normalized space
      const normalizedMouseX = (mouseX / canvas.width) * 2 - 1;
      const normalizedMouseY = -((mouseY / canvas.height*0.97) * 2 - 1);

      if (!index || !pointsData.length) {
        console.warn("Needs index and points data! ");
        return;
      }

      // Find the nearest point within a given radius， adjusting to normalized positions
      const radius = Math.min(canvas.offsetWidth, canvas.offsetHeight) * 0.01 ;
      const normalizedRadius = radius / Math.min(canvas.width, canvas.height) * 2;
      const nearest = index.within(normalizedMouseX, normalizedMouseY, normalizedRadius);

      let minDistance = Infinity;
      let nearestPoint = null;
      if (nearest.length > 0) {
          nearest.forEach((i) => {
          const point = pointsData[i];
          const distance = Math.sqrt(
              Math.pow(normalizedMouseX - point.x, 2) + Math.pow(normalizedMouseY - point.y, 2)
          );
          if (distance < minDistance) {
            minDistance = distance;
            nearestPoint = point;
          }
      });

      if (nearestPoint) {
          tooltip.show(nearestPoint.label, event.clientX, event.clientY);
      } else {
          tooltip.hide();
      }

    } else {
        //Hide tooltip if no point is nearby
          tooltip.hide();
      }
    });

    canvas.addEventListener("mouseout", () => {
      tooltip.hide();
    });
  }

  function createTooltip() {
    const tooltip = document.createElement("div");
    tooltip.id = "tooltip";
    document.body.appendChild(tooltip);

    return {
        show(content, x, y, offset = 20) {
            tooltip.textContent = content;
            tooltip.style.left = `${x - offset}px`;
            tooltip.style.top = `${y + offset}px`;
            tooltip.style.display = "block";
        },
        hide() {
            tooltip.style.display = "none";
        },
        element: tooltip
    };
  }

  function showLoadingSpinner() {
    document.getElementById("loading-spinner").classList.remove("hidden");
  }

  function hideLoadingSpinner() {
    document.getElementById("loading-spinner").classList.add("hidden");
  }

  // Debugging: Check for out-of-bounds points
  function checkOutOfBounds(data) {
        const outOfBounds = data.filter(([x, y]) => x < -1 || x > 1 || y < -1 || y > 1);
        if (outOfBounds.length > 0) {
            console.warn("Out-of-bounds points detected:", outOfBounds);
            console.warn(`Total out-of-bounds points: ${outOfBounds.length}`);
        } 
    }
});
