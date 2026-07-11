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
                // If a case is loaded, trigger update of input view
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

        try {
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
            
            predImg.src = 'data:image/png;base64,' + result.pred_base64;
            
            // Animate and set metric gauges
            setGauge(wtRing, wtVal, result.dice.wt);
            setGauge(tcRing, tcVal, result.dice.tc);
            setGauge(etRing, etVal, result.dice.et);

            updatePostInferenceInsight(result.dice);
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
