document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const caseSelect = document.getElementById('case-select');
    const hospitalSelect = document.getElementById('hospital-select');
    const methodSelect = document.getElementById('method-select');
    const sliceSlider = document.getElementById('slice-slider');
    const sliceVal = document.getElementById('slice-val');
    const predictBtn = document.getElementById('predict-btn');
    const shiftDesc = document.getElementById('shift-desc');
    const insightText = document.getElementById('insight-text');
    
    const mriImg = document.getElementById('mri-img');
    const gtImg = document.getElementById('gt-img');
    const predImg = document.getElementById('pred-img');
    
    const loadingMri = document.getElementById('loading-mri');
    const loadingGt = document.getElementById('loading-gt');
    const loadingPred = document.getElementById('loading-pred');
    const loading3d = document.getElementById('loading-3d');
    
    // View Toggles
    const grid2D = document.getElementById('grid-2d');
    const viewport3D = document.getElementById('viewport-3d');
    const mode2DBtn = document.getElementById('mode-2d-btn');
    const mode3DBtn = document.getElementById('mode-3d-btn');
    
    // Rings
    const wtRing = document.getElementById('wt-ring');
    const tcRing = document.getElementById('tc-ring');
    const etRing = document.getElementById('et-ring');
    const wtVal = document.getElementById('wt-val');
    const tcVal = document.getElementById('tc-val');
    const etVal = document.getElementById('et-val');

    // State Variables
    let casesData = {};
    let activeModality = 'flair';
    let activeDim = '2d';
    let activeViewMode = '2d'; // '2d' | '3d'
    let currentInferenceResult = null; // cached metrics
    
    // Three.js State
    let scene, camera, renderer, controls;
    let tumorGroup = null;

    // Scanner Shift Descriptions
    const SHIFT_INFO = {
        'None': 'Unshifted scanner intensities. Ideal representation of standard training distribution.',
        'H1': 'Hospital 1 scanner simulation: minor nonlinear contrast (gamma=1.06) and small blur (sigma=0.3).',
        'H2': 'Hospital 2 scanner simulation: mild contrast shift (gamma=1.13) and medium blur (sigma=0.5).',
        'H3': 'Hospital 3 scanner simulation: medium contrast shift (gamma=1.20) and spatial bias field (0.12).',
        'H4': 'Hospital 4 scanner simulation (STRONG OUTLIER): high contrast skew (gamma=1.85), large bias field (0.34), and heavy blur (sigma=1.7).'
    };

    // Initialize Gauge rings (circumference = 2 * PI * r = 314.16)
    function setGauge(ringElement, valueElement, value) {
        const circumference = 314.16;
        if (value === null || isNaN(value)) {
            ringElement.style.strokeDashoffset = circumference;
            valueElement.innerText = '--';
            return;
        }
        const offset = circumference - (value * circumference);
        ringElement.style.strokeDashoffset = offset;
        valueElement.innerText = (value * 100).toFixed(1) + '%';
    }

    // Three.js Scene Setup
    function init3D() {
        const container = document.getElementById('canvas-3d-container');
        if (!container) return;

        // Clear container first
        container.innerHTML = '';

        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0f1d);

        camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 1, 1000);
        camera.position.set(0, 0, 160);

        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.rotateSpeed = 0.8;
        controls.zoomSpeed = 1.0;
        controls.autoRotate = true;
        controls.autoRotateSpeed = 1.5;

        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambientLight);

        const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.7);
        dirLight1.position.set(1, 1, 1).normalize();
        scene.add(dirLight1);

        const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
        dirLight2.position.set(-1, -1, -1).normalize();
        scene.add(dirLight2);

        // Group to hold tumor meshes
        tumorGroup = new THREE.Group();
        scene.add(tumorGroup);

        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            if (controls) controls.update();
            renderer.render(scene, camera);
        }
        animate();

        // Resize handler
        window.addEventListener('resize', onWindowResize);
    }

    function onWindowResize() {
        const container = document.getElementById('canvas-3d-container');
        if (!container || !renderer || !camera) return;
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    }

    // Build 3D meshes inside Three.js scene
    function update3DMeshes(meshes) {
        if (!tumorGroup) return;
        clear3DStatus();

        // Clear old meshes
        while(tumorGroup.children.length > 0) { 
            const obj = tumorGroup.children[0];
            obj.geometry.dispose();
            obj.material.dispose();
            tumorGroup.remove(obj); 
        }

        const colors = {
            brain: 0x4b5563, // Brain - Cool Gray
            wt: 0x10b981,    // WT - Green
            tc: 0x3b82f6,    // TC - Blue
            et: 0xec4899     // ET - Pink/Red
        };

        const opacities = {
            brain: 0.06,     // Very faint transparent shell
            wt: 0.25,
            tc: 0.45,
            et: 0.80
        };

        let hasAnyGeometry = false;

        Object.keys(meshes).forEach(name => {
            const meshData = meshes[name];
            if (!meshData.vertices || meshData.vertices.length === 0) return;

            const geometry = new THREE.BufferGeometry();
            
            // meshData.vertices is already a flat Float32Array (xyz triples)
            // meshData.indices is already a flat Uint32Array
            geometry.setAttribute('position', new THREE.BufferAttribute(meshData.vertices, 3));
            geometry.setIndex(new THREE.BufferAttribute(meshData.indices, 1));
            geometry.computeVertexNormals();

            const material = new THREE.MeshPhongMaterial({
                color: colors[name],
                transparent: true,
                opacity: opacities[name],
                side: THREE.DoubleSide,
                // Critical WebGL trick: turn off depth write for the outer brain shell so nested inner objects render correctly
                depthWrite: name === 'brain' ? false : true,
                shininess: name === 'brain' ? 10 : 40,
                specular: 0x222222
            });

            const mesh = new THREE.Mesh(geometry, material);
            tumorGroup.add(mesh);
            hasAnyGeometry = true;
        });

        if (hasAnyGeometry) {
            // Re-center camera to frame the meshes
            const box = new THREE.Box3().setFromObject(tumorGroup);
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            
            // Zoom camera to fit bounding box
            const fov = camera.fov * (Math.PI / 180);
            let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
            cameraZ *= 1.4; // multiplier to pad the frame
            camera.position.set(0, 0, cameraZ);
            camera.lookAt(new THREE.Vector3(0, 0, 0));
            if (controls) {
                controls.target.set(0, 0, 0);
                controls.update();
            }
        }
    }

    // Toggle 2D / 3D mode views
    mode2DBtn.addEventListener('click', () => {
        mode2DBtn.classList.add('active');
        mode3DBtn.classList.remove('active');
        grid2D.style.display = 'grid';
        viewport3D.style.display = 'none';
        activeViewMode = '2d';
    });

    mode3DBtn.addEventListener('click', () => {
        mode3DBtn.classList.add('active');
        mode2DBtn.classList.remove('active');
        grid2D.style.display = 'none';
        viewport3D.style.display = 'block';
        activeViewMode = '3d';

        // Lazy initialize the scene on first switch
        if (!scene) {
            init3D();
        } else {
            // Trigger container resize
            setTimeout(onWindowResize, 50);
        }
        
        // If we already have predicted segmentations, load 3D mesh
        if (currentInferenceResult && tumorGroup && tumorGroup.children.length === 0) {
            fetch3DGeometry();
        } else if (!currentInferenceResult) {
            show3DStatus('Run Segmentation first, then switch to 3D view to see the volumetric mesh.');
        }
    });

    // Toggle dimension buttons (2D vs 3D)
    document.querySelectorAll('#dim-toggle button').forEach(button => {
        button.addEventListener('click', (e) => {
            document.querySelectorAll('#dim-toggle button').forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            activeDim = button.dataset.value;
            updateExperimentInsight();
        });
    });

    // Toggle Modality Tabs
    document.querySelectorAll('.tabs button').forEach(button => {
        button.addEventListener('click', (e) => {
            document.querySelectorAll('.tabs button').forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            activeModality = button.dataset.mod;
            if (caseSelect.value) {
                updateViews();
            }
        });
    });

    // Scanner Shift Description
    hospitalSelect.addEventListener('change', () => {
        shiftDesc.innerText = SHIFT_INFO[hospitalSelect.value] || '';
        updateExperimentInsight();
    });
    
    methodSelect.addEventListener('change', () => {
        updateExperimentInsight();
    });

    // Slider range change
    sliceSlider.addEventListener('input', () => {
        sliceVal.innerText = sliceSlider.value;
    });

    sliceSlider.addEventListener('change', () => {
        if (caseSelect.value) {
            updateViews();
        }
    });

    // Fetch and populate cases
    async function initCases() {
        try {
            const res = await fetch('/api/cases');
            if (!res.ok) throw new Error('Failed to load cases index');
            const data = await res.json();
            casesData = data;
            
            caseSelect.innerHTML = '';
            Object.keys(data).forEach((caseId, idx) => {
                const opt = document.createElement('option');
                opt.value = caseId;
                opt.text = `${caseId} (${data[caseId].hospital} - ${data[caseId].split})`;
                if (idx === 0) opt.selected = true;
                caseSelect.appendChild(opt);
            });

            handleCaseSelectChange();
        } catch (err) {
            console.error(err);
            caseSelect.innerHTML = '<option value="" disabled>Error loading cases</option>';
        }
    }

    function handleCaseSelectChange() {
        const caseId = caseSelect.value;
        if (!caseId || !casesData[caseId]) return;

        const metadata = casesData[caseId];
        const shape = metadata.shape; // [H, W, Z]
        const zMax = shape[2] - 1;

        sliceSlider.max = zMax;
        
        // Default slider value: if tumor slices exist, choose the median tumor slice
        const tumorZ = metadata.tumor_z;
        if (tumorZ && tumorZ.length > 0) {
            const midIdx = Math.floor(tumorZ.length / 2);
            sliceSlider.value = tumorZ[midIdx];
        } else {
            sliceSlider.value = Math.floor(zMax / 2);
        }
        sliceVal.innerText = sliceSlider.value;
        
        // Clear cached 3D prediction mesh
        currentInferenceResult = null;
        if (tumorGroup) {
            while(tumorGroup.children.length > 0) {
                const obj = tumorGroup.children[0];
                obj.geometry.dispose();
                obj.material.dispose();
                tumorGroup.remove(obj);
            }
        }
        
        // Load default views
        updateViews();
    }

    caseSelect.addEventListener('change', handleCaseSelectChange);

    // Fetch views (Modality + GT only, no prediction)
    async function updateViews() {
        const caseId = caseSelect.value;
        const sliceIdx = sliceSlider.value;
        const hospital = hospitalSelect.value;
        
        if (!caseId) return;

        loadingMri.style.display = 'flex';
        loadingGt.style.display = 'flex';

        try {
            const params = new URLSearchParams({
                case_id: caseId,
                slice_idx: sliceIdx,
                modality: activeModality,
                hospital: hospital
            });
            const res = await fetch(`/api/view?${params.toString()}`);
            if (!res.ok) throw new Error('Failed to load views');
            
            const data = await res.json();
            mriImg.src = 'data:image/png;base64,' + data.mri_base64;
            gtImg.src = 'data:image/png;base64,' + data.gt_base64;
        } catch (err) {
            console.error('Error loading views:', err);
        } finally {
            loadingMri.style.display = 'none';
            loadingGt.style.display = 'none';
        }
    }

    // Base64 → Uint8Array helper
    function base64ToUint8Array(base64) {
        const binaryString = atob(base64);
        const len = binaryString.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        return bytes;
    }

    // Center a flat Float32Array of xyz triples around a given centroid
    function centerVertices(vertices, cx, cy, cz) {
        for (let i = 0; i < vertices.length; i += 3) {
            vertices[i]     -= cx;
            vertices[i + 1] -= cy;
            vertices[i + 2] -= cz;
        }
    }

    // Compute centroid of a flat Float32Array of xyz triples
    function computeCentroid(vertices) {
        let cx = 0, cy = 0, cz = 0;
        const count = vertices.length / 3;
        if (count === 0) return [0, 0, 0];
        for (let i = 0; i < vertices.length; i += 3) {
            cx += vertices[i];
            cy += vertices[i + 1];
            cz += vertices[i + 2];
        }
        return [cx / count, cy / count, cz / count];
    }

    // Fetch raw binary volumes and run Marching Cubes client-side using pure JavaScript
    async function fetch3DGeometry() {
        const caseId = caseSelect.value;
        const hospital = hospitalSelect.value;
        const method = methodSelect.value;
        
        if (!caseId || !scene) return;

        loading3d.style.display = 'flex';

        try {
            // Fetch raw preprocessed binary volumes from the backend
            const res = await fetch('/api/mesh', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    case_id: caseId,
                    dim: activeDim,
                    method: method,
                    hospital: hospital
                })
            });

            if (!res.ok) throw new Error('Failed to fetch raw volume data');
            const data = await res.json();
            
            if (data.error) {
                console.error(data.error);
                return;
            }

            const dims = data.shape; // [nx, ny, nz] — matches NumPy C-order

            // Decode base64 volumes to Uint8Array
            const brainVol = base64ToUint8Array(data.brain);
            const wtVol    = base64ToUint8Array(data.wt);
            const tcVol    = base64ToUint8Array(data.tc);
            const etVol    = base64ToUint8Array(data.et);

            // Run pure JS marching cubes on each volume
            // Brain uses coarser step (4) for performance; tumor regions use finer step (2)
            console.time('marchingCubes_brain');
            const brainMesh = marchingCubes(brainVol, dims, 4, 128);
            console.timeEnd('marchingCubes_brain');

            console.time('marchingCubes_tumors');
            const wtMesh = marchingCubes(wtVol, dims, 2, 128);
            const tcMesh = marchingCubes(tcVol, dims, 2, 128);
            const etMesh = marchingCubes(etVol, dims, 2, 128);
            console.timeEnd('marchingCubes_tumors');

            console.log(`Brain: ${brainMesh.vertices.length / 3} verts, ${brainMesh.indices.length / 3} tris`);
            console.log(`WT: ${wtMesh.vertices.length / 3} verts, TC: ${tcMesh.vertices.length / 3} verts, ET: ${etMesh.vertices.length / 3} verts`);

            // Compute centroid from brain mesh and center all meshes
            const [cx, cy, cz] = computeCentroid(brainMesh.vertices);
            centerVertices(brainMesh.vertices, cx, cy, cz);
            centerVertices(wtMesh.vertices, cx, cy, cz);
            centerVertices(tcMesh.vertices, cx, cy, cz);
            centerVertices(etMesh.vertices, cx, cy, cz);

            // Render meshes in Three.js WebGL
            update3DMeshes({
                brain: brainMesh,
                wt: wtMesh,
                tc: tcMesh,
                et: etMesh
            });

        } catch (err) {
            console.error('3D mesh generation error:', err);
            show3DStatus('Error generating 3D mesh. Check console for details.');
        } finally {
            loading3d.style.display = 'none';
        }
    }

    // Show a status/hint message inside the 3D viewport
    function show3DStatus(message) {
        const container = document.getElementById('canvas-3d-container');
        if (!container) return;
        // Remove any existing status message
        const existing = container.querySelector('.viewport-status');
        if (existing) existing.remove();
        // Create status overlay
        const div = document.createElement('div');
        div.className = 'viewport-status';
        div.style.cssText = 'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);' +
            'color:#9ca3af;font-size:0.95rem;text-align:center;max-width:320px;line-height:1.6;' +
            'pointer-events:none;z-index:5;';
        div.textContent = message;
        container.style.position = 'relative';
        container.appendChild(div);
    }

    // Clear 3D status message (called when meshes are successfully loaded)
    function clear3DStatus() {
        const container = document.getElementById('canvas-3d-container');
        if (!container) return;
        const existing = container.querySelector('.viewport-status');
        if (existing) existing.remove();
    }

    // Run Segmentation Prediction
    async function runInference() {
        const caseId = caseSelect.value;
        const sliceIdx = sliceSlider.value;
        const hospital = hospitalSelect.value;
        const method = methodSelect.value;
        
        if (!caseId) return;

        predictBtn.disabled = true;
        predictBtn.innerHTML = '<div class="spinner" style="width:16px;height:16px;border-width:2px;display:inline-block;margin-right:8px;"></div> Running...';
        loadingPred.style.display = 'flex';

        // Reset metrics
        setGauge(wtRing, wtVal, null);
        setGauge(tcRing, tcVal, null);
        setGauge(etRing, etVal, null);
        currentInferenceResult = null;

        try {
            // Run slice-level predict
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    case_id: caseId,
                    dim: activeDim,
                    method: method,
                    hospital: hospital,
                    slice_idx: parseInt(sliceIdx),
                    modality: activeModality
                })
            });

            if (!res.ok) throw new Error('Prediction request failed');
            const result = await res.json();
            
            if (result.error) {
                insightText.innerHTML = `<span class="text-danger">Error: ${result.error}</span>`;
                return;
            }

            predImg.src = 'data:image/png;base64,' + result.pred_base64;
            
            // Set gauges
            setGauge(wtRing, wtVal, result.dice.wt);
            setGauge(tcRing, tcVal, result.dice.tc);
            setGauge(etRing, etVal, result.dice.et);

            currentInferenceResult = result;
            updatePostInferenceInsight(result.dice);

            // If 3D viewport is open, trigger mesh rendering
            if (activeViewMode === '3d') {
                fetch3DGeometry();
            } else {
                // Clear old meshes so it re-fetches if switched to 3D later
                if (tumorGroup) {
                    while(tumorGroup.children.length > 0) {
                        const obj = tumorGroup.children[0];
                        obj.geometry.dispose();
                        obj.material.dispose();
                        tumorGroup.remove(obj);
                    }
                }
            }
        } catch (err) {
            console.error('Prediction error:', err);
            insightText.innerHTML = '<span class="text-danger">Error: Could not run inference. Make sure the backend server is running and the checkpoints exist.</span>';
        } finally {
            predictBtn.disabled = false;
            predictBtn.innerHTML = '<span class="btn-icon">⚡</span> Run Segmentation';
            loadingPred.style.display = 'none';
        }
    }

    predictBtn.addEventListener('click', runInference);

    // Explanatory Insights for different selections
    function updateExperimentInsight() {
        const hospital = hospitalSelect.value;
        const method = methodSelect.value;
        const dim = activeDim;

        if (dim === '2d') {
            if (hospital === 'H4') {
                if (method === 'fedavg') {
                    insightText.innerHTML = '⚠️ <strong>Expected Behavior:</strong> In 2D, the global <strong>FedAvg</strong> model collapses on <strong>Hospital 4</strong> due to the strong scanner shift. You should see a poorly aligned prediction mask with low Dice scores (typically under 74%).';
                } else if (method === 'fedbn') {
                    insightText.innerHTML = '✨ <strong>Expected Behavior:</strong> Keeping Batch Normalization local in <strong>FedBN</strong> allows the model to adapt specifically to Hospital 4\'s scanner profile, successfully recovering performance (typically matching/beating 83% WT Dice).';
                } else {
                    insightText.innerHTML = 'In 2D, local models perform well locally but cannot collaborate. Centralized serves as the absolute ceiling.';
                }
            } else {
                insightText.innerHTML = `Running standard 2D U-Net parameters on typical scanner ${hospital}. Both FedAvg and FedBN are expected to show strong collaborative performance.`;
            }
        } else {
            // 3D
            if (hospital === 'H4') {
                if (method === 'fedavg') {
                    insightText.innerHTML = '💡 <strong>Expected Behavior:</strong> Unlike 2D, 3D <strong>FedAvg</strong> performs robustly on Hospital 4! The massive context of 3D spatial convolutions acts as a regularizer, helping the pooled data overcome scanner shifts.';
                } else if (method === 'fedbn') {
                    insightText.innerHTML = '⚠️ <strong>Expected Behavior:</strong> In 3D, <strong>FedBN</strong> actually performs worse than FedAvg. This is because clients have only 150 local cases, which is statistically insufficient to estimate stable local BN running statistics.';
                } else {
                    insightText.innerHTML = 'In 3D, Local-only suffers from severe overfitting due to small sample sizes (150 volumes).';
                }
            } else {
                insightText.innerHTML = 'Running 3D U-Net. In 3D architectures, FedAvg serves as the superior general model due to high data pooling advantages.';
            }
        }
    }

    function updatePostInferenceInsight(dice) {
        const hospital = hospitalSelect.value;
        const method = methodSelect.value;
        const dim = activeDim;
        const wt = dice.wt;

        let txt = `WT Dice is <strong>${(wt * 100).toFixed(1)}%</strong>. `;

        if (dim === '2d' && hospital === 'H4') {
            if (method === 'fedavg') {
                txt += 'Notice how the segmentation borders are blurred or missed completely—this is the classic FedAvg scanner drift collapse.';
            } else if (method === 'fedbn') {
                txt += 'Observe how sharp the segmentation boundaries are. Local BN layers correctly normalized H4\'s skewed intensities before feeding them to the shared CNN body!';
            }
        } else if (dim === '3d' && hospital === 'H4') {
            if (method === 'fedavg') {
                txt += 'In 3D, the collaborative power of FedAvg yields excellent boundaries. Data volume acts as a natural noise filter.';
            } else if (method === 'fedbn') {
                txt += 'Note that local BN layers struggled to estimate normalization values, resulting in slightly lower segment quality than FedAvg.';
            }
        }
        
        insightText.innerHTML = txt;
    }

    // Run Initialization
    initCases();
    updateExperimentInsight();
});
