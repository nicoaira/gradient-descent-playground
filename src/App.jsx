import React, { useState, useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';
import createPlotlyComponent from 'react-plotly.js/factory';
import { Play, Pause, FastForward, RotateCcw, BrainCircuit, Sun, Moon, Eye, EyeOff, ChevronLeft, ChevronRight } from 'lucide-react';
import { MLP, generateClassificationData } from './MLP';

// Step 4: Hardcoded classification data. Class 1 clusters around (3.3, 3.5),
// class 0 surrounds it — with three noisy class-0 points inside the cluster.
const TRAIN_DATA_4 = [
    // Class 0 — surrounding "X" points
    { x: 0.5, y: 0.5, label: 0 }, { x: 1.5, y: 1.0, label: 0 },
    { x: 1.0, y: 2.0, label: 0 }, { x: 2.0, y: 0.5, label: 0 },
    { x: 3.0, y: 1.0, label: 0 }, { x: 4.0, y: 1.5, label: 0 },
    { x: 5.0, y: 0.5, label: 0 }, { x: 0.5, y: 3.0, label: 0 },
    { x: 0.5, y: 4.5, label: 0 }, { x: 1.5, y: 5.0, label: 0 },
    { x: 5.5, y: 2.0, label: 0 }, { x: 5.5, y: 5.0, label: 0 },
    { x: 4.5, y: 5.5, label: 0 }, { x: 2.5, y: 5.5, label: 0 },
    { x: 0.5, y: 2.0, label: 0 }, { x: 4.0, y: 0.5, label: 0 },
    { x: 5.3, y: 2.5, label: 0 }, { x: 5.5, y: 4.0, label: 0 },
    { x: 1.0, y: 5.5, label: 0 }, { x: 3.5, y: 5.5, label: 0 },
    // Class 0 — noise points that fell inside the class-1 cluster
    { x: 3.0, y: 3.5, label: 0 }, { x: 3.8, y: 2.8, label: 0 },
    { x: 2.8, y: 4.2, label: 0 },
    // Class 1 — "O" cluster near center
    { x: 2.5, y: 4.0, label: 1 }, { x: 3.8, y: 3.2, label: 1 },
    { x: 3.0, y: 4.5, label: 1 }, { x: 4.2, y: 4.0, label: 1 },
    { x: 2.0, y: 3.5, label: 1 }, { x: 3.5, y: 2.5, label: 1 },
    { x: 4.5, y: 3.5, label: 1 }, { x: 2.8, y: 2.8, label: 1 },
    { x: 4.0, y: 4.5, label: 1 }, { x: 3.3, y: 3.8, label: 1 },
    { x: 2.3, y: 3.0, label: 1 }, { x: 4.5, y: 2.5, label: 1 },
    { x: 3.8, y: 4.3, label: 1 }, { x: 3.5, y: 3.0, label: 1 },
    { x: 2.5, y: 2.5, label: 1 }
];
const VAL_DATA_4 = [
    { x: 1.0, y: 1.5, label: 0 }, { x: 5.0, y: 1.0, label: 0 },
    { x: 1.0, y: 4.0, label: 0 }, { x: 5.3, y: 4.5, label: 0 },
    { x: 5.3, y: 4.2, label: 0 }, { x: 2.0, y: 5.2, label: 0 },
    { x: 2.5, y: 3.5, label: 1 }, { x: 3.5, y: 4.0, label: 1 },
    // These four sit on top of the class-0 noise points — overfit misses them.
    { x: 4.0, y: 3.0, label: 1 }, { x: 3.1, y: 3.5, label: 1 },
    { x: 3.9, y: 2.7, label: 1 }, { x: 2.9, y: 4.1, label: 1 }
];

// Three pretrained decision boundaries.
const modelUnderfit = (x, y) => (x + y > 4.5 ? 1 : 0);
const modelGood = (x, y) => ((x - 3.3) ** 2 + (y - 3.5) ** 2 < 4 ? 1 : 0);
const modelOverfit = (x, y) => {
    if ((x - 3.3) ** 2 + (y - 3.5) ** 2 >= 4) return 0;
    // Carve out the three class-0 noise points inside the cluster.
    if ((x - 3.0) ** 2 + (y - 3.5) ** 2 < 0.1) return 0;
    if ((x - 3.8) ** 2 + (y - 2.8) ** 2 < 0.1) return 0;
    if ((x - 2.8) ** 2 + (y - 4.2) ** 2 < 0.1) return 0;
    return 1;
};

const accuracy = (model, data) => {
    let ok = 0;
    for (const pt of data) if (model(pt.x, pt.y) === pt.label) ok++;
    return ok / data.length;
};

const MODELS_4 = [
    { key: 'underfit', name: 'Under-fitting', color: '#ef4444', predict: modelUnderfit, description: 'Too simple — a straight line can\'t carve out the center cluster.' },
    { key: 'good', name: 'Appropriate Fit', color: '#10b981', predict: modelGood, description: 'Captures the cluster shape cleanly without contorting around noise.' },
    { key: 'overfit', name: 'Over-fitting', color: '#a855f7', predict: modelOverfit, description: 'Carves tight regions around every training point — even the mislabeled noise.' }
];

// Plotly.js 3+ often crashes in React 19 during unmount due to strict DOM detachments.
// Wrapping its purge method protects the app from unmount crashes.
const originalPurge = Plotly.purge;
Plotly.purge = (gd) => {
    try {
        if (originalPurge) originalPurge(gd);
    } catch (e) {
        console.warn("Caught Plotly purge error on unmount:", e);
    }
};

const Plot = createPlotlyComponent(Plotly);

const generateData = (trueM, trueB) => {
    const pts = [];
    for (let i = 0; i < 20; i++) {
        const x = Math.random() * 10;
        const y = trueM * x + trueB + (Math.random() - 0.5) * 4; // Add noise
        pts.push({ x, y });
    }
    return pts.sort((a, b) => a.x - b.x);
};

// Calculate Cost (MSE)
const calcCost = (data, m, b) => {
    let error = 0;
    for (let pt of data) {
        const pred = m * pt.x + b;
        error += Math.pow(pred - pt.y, 2);
    }
    return error / (2 * data.length);
};

export default function App() {
    const [step, setStep] = useState(1);
    const [trueM, setTrueM] = useState("2.8");
    const [trueB, setTrueB] = useState("0");
    const [data, setData] = useState(() => generateData(2.8, 0));

    // Model parameters
    const [m, setM] = useState(0);
    const [b, setB] = useState(0);
    const [lr, setLr] = useState(0.01);
    const [isPlaying, setIsPlaying] = useState(false);
    const [show3D, setShow3D] = useState(false);
    const [showErrors, setShowErrors] = useState(true);

    // New state for visualizing the slope/gradient before stepping
    const [pendingGradient, setPendingGradient] = useState(null);

    // Tracing the path of gradient descent
    const [history, setHistory] = useState([{ m: 0, b: 0, cost: calcCost(data, 0, 0) }]);

    // Track the viewport of the 1D plot to keep curve "infinite"
    const [mView, setMView] = useState({ min: -2, max: 6 });

    // Step 3 (Neural Network) State
    const [hiddenNeurons, setHiddenNeurons] = useState(4);
    const mlpRef = useRef(new MLP(4));
    const [datasetType, setDatasetType] = useState('circles');
    const [classData, setClassData] = useState(() => generateClassificationData('circles'));
    const [nnEpochs, setNnEpochs] = useState(0);
    const [nnLoss, setNnLoss] = useState(1);
    const [nnLossHistory, setNnLossHistory] = useState([]);

    const [showProbSurface, setShowProbSurface] = useState(false);

    // Step 4 (Train/Validation) state
    const [revealValidation, setRevealValidation] = useState(false);
    const [selectedModel4, setSelectedModel4] = useState(1); // start on "Appropriate Fit"

    const [theme, setTheme] = useState('dark');

    useEffect(() => {
        document.documentElement.setAttribute('data-theme', theme);
    }, [theme]);

    const pTheme = {
        fontColor: theme === 'dark' ? '#e2e8f0' : '#1e293b',
        gridColor: theme === 'dark' ? '#334155' : '#cbd5e1'
    };

    // Reference for the animation loop
    const requestRef = useRef();
    // Preserve 3D camera angle across data updates
    const cameraRef = useRef(null);

    const resetMLP = (neurons) => {
        mlpRef.current = new MLP(neurons);
        setNnEpochs(0);
        setNnLossHistory([]);
        setPendingGradient(null);
    };

    useEffect(() => { resetMLP(hiddenNeurons); }, [hiddenNeurons]);

    // Reset model
    const resetModel = () => {
        if (step === 3) {
            resetMLP(hiddenNeurons);
            setIsPlaying(false);
            return;
        }
        setM(0);
        setB(0);
        setHistory([{ m: 0, b: 0, cost: calcCost(data, 0, 0) }]);
        setIsPlaying(false);
        setPendingGradient(null);
    };

    const regenerateData = (type) => {
        const t = type ?? datasetType;
        if (step === 3) {
            setClassData(generateClassificationData(t));
        } else {
            setData(generateData(parseFloat(trueM) || 0, parseFloat(trueB) || 0));
        }
        resetModel();
    };

    // Use refs so the animation loop always reads the latest values
    const mRef = useRef(m);
    const bRef = useRef(b);
    const lrRef = useRef(lr);
    const stepRef = useRef(step);
    const dataRef = useRef(data);
    const classDataRef = useRef(classData);
    useEffect(() => { mRef.current = m; }, [m]);
    useEffect(() => { bRef.current = b; }, [b]);
    useEffect(() => { lrRef.current = lr; }, [lr]);
    useEffect(() => { stepRef.current = step; }, [step]);
    useEffect(() => { dataRef.current = data; }, [data]);
    useEffect(() => { classDataRef.current = classData; }, [classData]);

    // Calculate Gradient using explicit m,b values (no stale closures)
    const calculateGradientAt = (curM, curB, curStep, curData) => {
        const N = curData.length;
        let dj_dm = 0;
        let dj_db = 0;

        for (let pt of curData) {
            const usedB = curStep === 1 ? 0 : curB;
            const pred = curM * pt.x + usedB;
            const error = pred - pt.y;

            dj_dm += error * pt.x;
            if (curStep === 2) {
                dj_db += error;
            }
        }
        return { dj_dm: dj_dm / N, dj_db: dj_db / N };
    };

    // Convenience wrapper using current state (for manual steps)
    const calculateGradient = () => calculateGradientAt(m, b, step, data);

    const applyGradient = (dj_dm, dj_db) => {
        setM((prevM) => {
            const curLr = lrRef.current;
            const curStep = stepRef.current;
            const curData = dataRef.current;
            const newM = prevM - curLr * dj_dm;
            setB((prevB) => {
                const newB = curStep === 1 ? 0 : prevB - curLr * dj_db;
                const newCost = calcCost(curData, newM, newB);
                setHistory(prev => [...prev, { m: newM, b: newB, cost: newCost }]);
                return newB;
            });
            return newM;
        });
    };

    // One full step: compute gradient from current refs, then apply
    const takeStepFromRefs = () => {
        setM((prevM) => {
            const curB = bRef.current;
            const curLr = lrRef.current;
            const curStep = stepRef.current;
            const curData = dataRef.current;
            const grads = calculateGradientAt(prevM, curB, curStep, curData);
            const newM = prevM - curLr * grads.dj_dm;
            setB((prevB) => {
                const newB = curStep === 1 ? 0 : prevB - curLr * grads.dj_db;
                const newCost = calcCost(curData, newM, newB);
                setHistory(prev => [...prev, { m: newM, b: newB, cost: newCost }]);
                return newB;
            });
            return newM;
        });
    };

    // Manual interaction step (uses current rendered state)
    const handleManualStep = () => {
        if (!pendingGradient) {
            setPendingGradient(calculateGradient());
        } else {
            applyGradient(pendingGradient.dj_dm, pendingGradient.dj_db);
            setPendingGradient(null);
        }
    };

    // Handle clicking directly on the plot to set m/b
    const handlePlotClick = (evt, isStep1) => {
        if (!evt || !evt.points || evt.points.length === 0) return;
        const pt = evt.points[0];
        const newM = pt.x;
        const newB = isStep1 ? 0 : pt.y;

        setIsPlaying(false);
        setM(newM);
        setB(newB);
        setPendingGradient(null);
        setHistory([{ m: newM, b: newB, cost: calcCost(data, newM, newB) }]);
    };

    // Animation Loop – stable interval, no dependency on m/b
    useEffect(() => {
        let intervalId;
        if (isPlaying) {
            intervalId = setInterval(() => {
                if (stepRef.current === 3) {
                    const lrVal = lrRef.current;
                    let lastLoss = 0;
                    for (let i = 0; i < 5; i++) {
                        mlpRef.current.forward(classDataRef.current);
                        lastLoss = mlpRef.current.backward(classDataRef.current, lrVal);
                    }
                    setNnEpochs(e => {
                        const next = e + 5;
                        setNnLoss(lastLoss);
                        if (next % 10 === 0) {
                            setNnLossHistory(prev => [...prev, { epoch: next, loss: lastLoss }]);
                        }
                        return next;
                    });
                } else {
                    takeStepFromRefs();
                }
            }, 100);
        }
        return () => { if (intervalId) clearInterval(intervalId); };
    }, [isPlaying]);


    // Prepare Plot Data
    const xVals = data.map(d => d.x);
    const yVals = data.map(d => d.y);
    const max_X = Math.max(...xVals, 10);
    const line_X = [0, max_X];
    const currentB = step === 1 ? 0 : b;
    const line_Y = [currentB, m * max_X + currentB];

    // 1D Cost Curve Setup (Cost vs m, assuming b=0)
    // 1D Cost Curve Setup (Cost vs m, assuming b=0)
    const calc1DCurve = () => {
        const m_vals = [];
        const cost_vals = [];
        const span = mView.max - mView.min;
        const step = span / 100; // 100 points is enough for a smooth parabola

        // Draw slightly beyond the view to avoid edges during panning
        for (let i = mView.min - span; i <= mView.max + span; i += step) {
            m_vals.push(i);
            cost_vals.push(calcCost(data, i, 0));
        }
        return { m_vals, cost_vals };
    };

    // 3D Cost Surface Setup (Cost vs m vs b)
    const calc2DSurface = () => {
        const m_vals = [];
        const b_vals = [];
        const cost_grid = [];

        // Dynamically size the grid to ensure it always encompasses the entire path
        let min_m = -2;
        let max_m = 6;
        let min_b = -5;
        let max_b = 15;

        for (let h of history) {
            if (h.m < min_m) min_m = h.m;
            if (h.m > max_m) max_m = h.m;
            if (h.b < min_b) min_b = h.b;
            if (h.b > max_b) max_b = h.b;
        }

        // Add 10% padding so the red points never sit exactly on the edge
        const m_pad = Math.max((max_m - min_m) * 0.1, 1);
        const b_pad = Math.max((max_b - min_b) * 0.1, 1);
        min_m -= m_pad;
        max_m += m_pad;
        min_b -= b_pad;
        max_b += b_pad;

        const m_step = (max_m - min_m) / 30; // ~30 grid points for performance
        const b_step = (max_b - min_b) / 30;

        for (let i = min_b; i <= max_b; i += b_step) b_vals.push(i);
        for (let i = min_m; i <= max_m; i += m_step) m_vals.push(i);

        for (let i = 0; i < b_vals.length; i++) {
            const cost_row = [];
            for (let j = 0; j < m_vals.length; j++) {
                cost_row.push(calcCost(data, m_vals[j], b_vals[i]));
            }
            cost_grid.push(cost_row);
        }
        return { m_vals, b_vals, cost_grid };
    };

    // Generate Neural Net Contour Grid on the fly if Step 3
    let contourZ = [];
    let contourX = [];
    let contourY = [];
    if (step === 3) {
        for (let i = -6; i <= 6; i += 0.4) contourX.push(i);
        for (let j = -6; j <= 6; j += 0.4) contourY.push(j);
        for (let y of contourY) {
            let row = [];
            for (let x of contourX) {
                const prob = mlpRef.current.predict(x, y);
                row.push(showProbSurface ? prob : (prob >= 0.5 ? 1 : 0));
            }
            contourZ.push(row);
        }
    }

    return (
        <div className="app-container">
            <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                    <div className="title">Gradient Descent Playground</div>
                    <div className="subtitle">Visualizing Optimization for Linear Regression</div>
                </div>
                <button
                    className="btn btn-secondary"
                    onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
                    style={{ padding: '0.6rem', borderRadius: '50%', width: '42px', height: '42px', justifyContent: 'center' }}
                    title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
                >
                    {theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
                </button>
            </header>

            <div className="nav-tabs">
                <div className={`nav-tab ${step === 1 ? 'active' : ''}`} onClick={() => {
                    setStep(1); setTrueM("2.8"); setTrueB("0"); setData(generateData(2.8, 0)); resetModel();
                }}>
                    Step 1: 1D Search (Slope Only)
                </div>
                <div className={`nav-tab ${step === 2 ? 'active' : ''}`} onClick={() => {
                    setStep(2); setTrueM("3.5"); setTrueB("1.8"); setData(generateData(3.5, 1.8)); resetModel();
                }}>
                    Step 2: 2D Search (Slope & Intercept)
                </div>
                <div className={`nav-tab ${step === 3 ? 'active' : ''}`} onClick={() => {
                    setStep(3); resetMLP(hiddenNeurons); setClassData(generateClassificationData()); setIsPlaying(false);
                }}>
                    Step 3: Neural Network (Classification)
                </div>
                <div className={`nav-tab ${step === 4 ? 'active' : ''}`} onClick={() => {
                    setStep(4); setIsPlaying(false); setRevealValidation(false);
                }}>
                    Step 4: Train / Validation Split
                </div>
            </div>

            <div className="layout-grid glass-panel">
                {/* LEFT PANEL: Data & Fit */}
                <div className="control-panel" style={{ borderRight: '1px solid var(--border-color)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <h3 style={{ margin: 0, color: 'var(--accent)' }}>
                            {step === 3 ? "Decision Boundary" : step === 4 ? "Three Pretrained Models" : "Data & Regression Fit"}
                        </h3>
                        {step === 3 ? (
                            <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.85rem', color: '#94a3b8', cursor: 'pointer', userSelect: 'none' }}>
                                <input type="checkbox" checked={showProbSurface} onChange={e => setShowProbSurface(e.target.checked)} style={{ accentColor: '#3b82f6' }} />
                                Prob. Surface
                            </label>
                        ) : step === 4 ? null : (
                            <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.85rem', color: '#94a3b8', cursor: 'pointer' }}>
                                <input type="checkbox" checked={showErrors} onChange={e => setShowErrors(e.target.checked)} style={{ accentColor: '#ef4444' }} />
                                Show Errors
                            </label>
                        )}
                    </div>
                    {step === 4 ? (
                        (() => {
                            const active = MODELS_4[selectedModel4];
                            const gridX = [];
                            const gridY = [];
                            for (let i = 0; i <= 6; i += 0.06) { gridX.push(Number(i.toFixed(2))); gridY.push(Number(i.toFixed(2))); }
                            const gridZ = gridY.map(yv => gridX.map(xv => active.predict(xv, yv)));

                            const trainClass0 = TRAIN_DATA_4.filter(d => d.label === 0);
                            const trainClass1 = TRAIN_DATA_4.filter(d => d.label === 1);
                            const valClass0 = VAL_DATA_4.filter(d => d.label === 0);
                            const valClass1 = VAL_DATA_4.filter(d => d.label === 1);

                            return (
                                <>
                                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '1rem', marginTop: '-0.25rem' }}>
                                        <button
                                            className="btn btn-secondary"
                                            onClick={() => setSelectedModel4((selectedModel4 + MODELS_4.length - 1) % MODELS_4.length)}
                                            style={{ padding: '0.4rem', borderRadius: '50%', width: '36px', height: '36px', justifyContent: 'center' }}
                                            title="Previous model"
                                        >
                                            <ChevronLeft size={18} />
                                        </button>
                                        <div style={{ minWidth: '200px', textAlign: 'center' }}>
                                            <div style={{ color: active.color, fontWeight: 700, fontSize: '1.15rem' }}>{active.name}</div>
                                            <div style={{ color: '#94a3b8', fontSize: '0.75rem' }}>{selectedModel4 + 1} of {MODELS_4.length}</div>
                                        </div>
                                        <button
                                            className="btn btn-secondary"
                                            onClick={() => setSelectedModel4((selectedModel4 + 1) % MODELS_4.length)}
                                            style={{ padding: '0.4rem', borderRadius: '50%', width: '36px', height: '36px', justifyContent: 'center' }}
                                            title="Next model"
                                        >
                                            <ChevronRight size={18} />
                                        </button>
                                    </div>
                                    <Plot
                                        data={[
                                            {
                                                z: gridZ, x: gridX, y: gridY,
                                                type: 'contour',
                                                colorscale: [
                                                    [0, 'rgba(34, 197, 94, 0.25)'],
                                                    [0.5, 'rgba(255, 255, 255, 0)'],
                                                    [1, 'rgba(234, 179, 8, 0.25)']
                                                ],
                                                showscale: false,
                                                hoverinfo: 'skip',
                                                contours: { coloring: 'heatmap' },
                                                line: { color: active.color, width: 2 }
                                            },
                                            {
                                                x: trainClass0.map(d => d.x), y: trainClass0.map(d => d.y),
                                                mode: 'markers', type: 'scatter',
                                                marker: { color: '#22c55e', size: 14, symbol: 'x', line: { width: 2 } },
                                                name: 'Train · Class 0'
                                            },
                                            {
                                                x: trainClass1.map(d => d.x), y: trainClass1.map(d => d.y),
                                                mode: 'markers', type: 'scatter',
                                                marker: { color: '#eab308', size: 13, symbol: 'circle-open', line: { width: 3 } },
                                                name: 'Train · Class 1'
                                            },
                                            ...(revealValidation ? [
                                                {
                                                    x: valClass0.map(d => d.x), y: valClass0.map(d => d.y),
                                                    mode: 'markers', type: 'scatter',
                                                    marker: { color: '#22c55e', size: 17, symbol: 'x', line: { width: 3, color: '#ffffff' } },
                                                    name: 'Val · Class 0'
                                                },
                                                {
                                                    x: valClass1.map(d => d.x), y: valClass1.map(d => d.y),
                                                    mode: 'markers', type: 'scatter',
                                                    marker: { color: '#eab308', size: 16, symbol: 'circle-open', line: { width: 4, color: '#ffffff' } },
                                                    name: 'Val · Class 1'
                                                }
                                            ] : [])
                                        ]}
                                        layout={{
                                            autosize: true,
                                            showlegend: true,
                                            legend: { orientation: 'h', y: -0.18, x: 0.5, xanchor: 'center', font: { size: 10 } },
                                            paper_bgcolor: 'transparent',
                                            plot_bgcolor: 'transparent',
                                            font: { color: pTheme.fontColor },
                                            margin: { t: 10, r: 20, l: 50, b: 70 },
                                            xaxis: { title: { text: 'x1' }, gridcolor: pTheme.gridColor, automargin: true, range: [0, 6], fixedrange: true },
                                            yaxis: { title: { text: 'x2' }, gridcolor: pTheme.gridColor, automargin: true, range: [0, 6], fixedrange: true }
                                        }}
                                        useResizeHandler={true}
                                        style={{ width: "100%", height: "350px" }}
                                    />
                                </>
                            );
                        })()
                    ) : step === 3 ? (
                        <Plot
                            data={[
                                {
                                    z: contourZ,
                                    x: contourX,
                                    y: contourY,
                                    type: 'contour',
                                    colorscale: [
                                        [0, 'rgba(59, 130, 246, 0.5)'],
                                        [0.5, 'transparent'],
                                        [1, 'rgba(239, 68, 68, 0.5)']
                                    ],
                                    showscale: false,
                                    hoverinfo: 'skip'
                                },
                                {
                                    x: classData.filter(d => d.label === 0).map(d => d.x),
                                    y: classData.filter(d => d.label === 0).map(d => d.y),
                                    mode: 'markers',
                                    type: 'scatter',
                                    marker: { color: '#3b82f6', size: 10, line: { color: 'white', width: 1 } },
                                    name: 'Class 0'
                                },
                                {
                                    x: classData.filter(d => d.label === 1).map(d => d.x),
                                    y: classData.filter(d => d.label === 1).map(d => d.y),
                                    mode: 'markers',
                                    type: 'scatter',
                                    marker: { color: '#ef4444', size: 10, line: { color: 'white', width: 1 } },
                                    name: 'Class 1'
                                }
                            ]}
                            layout={{
                                autosize: true,
                                paper_bgcolor: 'transparent',
                                plot_bgcolor: 'transparent',
                                font: { color: pTheme.fontColor },
                                margin: { t: 30, r: 20, l: 60, b: 60 },
                                xaxis: { title: { text: 'x1' }, gridcolor: pTheme.gridColor, automargin: true, range: [-6, 6] },
                                yaxis: { title: { text: 'x2' }, gridcolor: pTheme.gridColor, automargin: true, range: [-6, 6] }
                            }}
                            useResizeHandler={true}
                            style={{ width: "100%", height: "350px" }}
                        />
                    ) : (
                        <Plot
                            data={[
                                {
                                    x: xVals,
                                    y: yVals,
                                    mode: 'markers',
                                    type: 'scatter',
                                    marker: { color: '#3b82f6', size: 8 },
                                    name: 'Data Points'
                                },
                                {
                                    x: line_X,
                                    y: line_Y,
                                    mode: 'lines',
                                    type: 'scatter',
                                    line: { color: '#ef4444', width: 3 },
                                    name: 'Prediction'
                                },
                                ...(showErrors ? data.map((pt, i) => {
                                    const pred = m * pt.x + currentB;
                                    return {
                                        x: [pt.x, pt.x],
                                        y: [pt.y, pred],
                                        mode: 'lines',
                                        type: 'scatter',
                                        line: { color: 'rgba(239, 68, 68, 0.5)', width: 1.5, dash: 'dash' },
                                        showlegend: i === 0,
                                        name: i === 0 ? 'Residual' : undefined,
                                        hoverinfo: 'skip'
                                    };
                                }) : [])
                            ]}
                            layout={{
                                autosize: true,
                                paper_bgcolor: 'transparent',
                                plot_bgcolor: 'transparent',
                                font: { color: pTheme.fontColor },
                                margin: { t: 30, r: 20, l: 80, b: 80 },
                                xaxis: { title: { text: 'x', standoff: 15 }, gridcolor: pTheme.gridColor, automargin: true },
                                yaxis: { title: { text: 'y', standoff: 15 }, gridcolor: pTheme.gridColor, automargin: true }
                            }}
                            useResizeHandler={true}
                            style={{ width: "100%", height: "350px" }}
                        />
                    )}

                    {step !== 4 && (
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                            <div className="metric-card">
                                <div className="metric-value">{step === 3 ? nnLoss.toFixed(4) : m.toFixed(3)}</div>
                                <div className="metric-label">{step === 3 ? 'Cross Entropy Loss' : 'Slope (m)'}</div>
                            </div>
                            {step === 2 && (
                                <div className="metric-card">
                                    <div className="metric-value">{b.toFixed(3)}</div>
                                    <div className="metric-label">Intercept (b)</div>
                                </div>
                            )}
                            {step !== 3 && (
                                <div className="metric-card">
                                    <div className="metric-value">{calcCost(data, m, step === 1 ? 0 : b).toFixed(3)}</div>
                                    <div className="metric-label">MSE (Cost)</div>
                                </div>
                            )}
                        </div>
                    )}

                    {step === 4 ? (
                        <div style={{ marginTop: '0.5rem', padding: '1rem', background: 'var(--card-bg)', borderRadius: '12px', border: '1px solid var(--border-color)', fontSize: '0.88rem', color: '#94a3b8', lineHeight: 1.5 }}>
                            <div style={{ marginBottom: '0.5rem' }}>
                                <strong style={{ color: MODELS_4[selectedModel4].color }}>{MODELS_4[selectedModel4].name}:</strong> {MODELS_4[selectedModel4].description}
                            </div>
                            <div style={{ fontSize: '0.82rem', paddingTop: '0.5rem', borderTop: '1px solid var(--border-color)' }}>
                                Three models were pretrained only on the <span style={{ color: '#3b82f6' }}>training set</span>. Training accuracy alone can't tell if they'll generalize — use the <span style={{ color: '#f59e0b' }}>validation set</span> to find out.
                            </div>
                        </div>
                    ) : step === 3 ? (
                        <div style={{ marginTop: '1.5rem', padding: '1rem', background: 'var(--card-bg)', borderRadius: '12px', border: '1px solid var(--border-color)', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}>
                            <h4 style={{ margin: '0 0 0.75rem 0', color: '#94a3b8' }}>Network Architecture (Complexity)</h4>
                            <div style={{ marginBottom: '1rem' }}>
                                <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '0.5rem' }}>Dataset Pattern</div>
                                <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
                                    {['circles', 'moons', 'spirals'].map(t => (
                                        <button
                                            key={t}
                                            onClick={() => {
                                                setDatasetType(t);
                                                setClassData(generateClassificationData(t));
                                                resetMLP(hiddenNeurons);
                                                setIsPlaying(false);
                                            }}
                                            style={{
                                                padding: '0.3rem 0.75rem',
                                                borderRadius: '999px',
                                                border: `1px solid ${datasetType === t ? 'var(--primary)' : 'var(--border-color)'}`,
                                                background: datasetType === t ? 'var(--primary)' : 'transparent',
                                                color: datasetType === t ? 'white' : 'var(--text-color)',
                                                fontSize: '0.8rem',
                                                cursor: 'pointer',
                                                textTransform: 'capitalize',
                                                transition: 'all 0.2s'
                                            }}
                                        >
                                            {t}
                                        </button>
                                    ))}
                                </div>
                            </div>
                            <div className="slider-container">
                                <div className="slider-header">
                                    <span>Hidden Neurons</span>
                                    <span style={{ color: 'var(--primary)' }}>{hiddenNeurons} Neurons</span>
                                </div>
                                <input
                                    type="range"
                                    min="2" max="25" step="1"
                                    value={hiddenNeurons}
                                    onChange={e => setHiddenNeurons(Number(e.target.value))}
                                />
                                <small style={{ color: '#94a3b8', fontSize: '0.75rem', marginTop: '0.25rem', display: 'block' }}>
                                    More neurons = more complex boundaries. Try switching dataset patterns to see how even 2 neurons can fail on spirals!
                                </small>
                            </div>
                            <button className="btn btn-secondary" onClick={() => regenerateData()} style={{ padding: '0.5rem 1rem', marginTop: '1rem' }}>
                                <RotateCcw size={16} /> Rescatter Dataset
                            </button>
                        </div>
                    ) : (
                        <div style={{ marginTop: '1.5rem', padding: '1rem', background: 'var(--card-bg)', borderRadius: '12px', border: '1px solid var(--border-color)', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}>
                            <h4 style={{ margin: '0 0 0.5rem 0', color: '#94a3b8' }}>Set Values</h4>
                            <div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-end', flexWrap: 'wrap' }}>
                                <div>
                                    <label style={{ display: 'block', fontSize: '0.8rem', color: '#94a3b8', marginBottom: '0.25rem' }}>True Slope</label>
                                    <input
                                        type="text"
                                        value={trueM}
                                        onChange={e => setTrueM(e.target.value)}
                                        style={{ width: '80px', background: 'var(--bg-color)', border: '1px solid var(--input-border)', color: 'var(--text-color)', padding: '0.5rem', borderRadius: '4px' }}
                                    />
                                </div>
                                {step === 2 && (
                                    <div>
                                        <label style={{ display: 'block', fontSize: '0.8rem', color: '#94a3b8', marginBottom: '0.25rem' }}>True Intercept</label>
                                        <input
                                            type="text"
                                            value={trueB}
                                            onChange={e => setTrueB(e.target.value)}
                                            style={{ width: '80px', background: 'var(--bg-color)', border: '1px solid var(--input-border)', color: 'var(--text-color)', padding: '0.5rem', borderRadius: '4px' }}
                                        />
                                    </div>
                                )}
                                <button className="btn btn-secondary" onClick={regenerateData} style={{ padding: '0.5rem 1rem' }}>
                                    <RotateCcw size={16} /> Data
                                </button>
                            </div>
                        </div>
                    )}
                </div>

                {/* RIGHT PANEL: Optimization Surface / Curve */}
                <div className="control-panel">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <h3 style={{ margin: 0, color: 'var(--accent)' }}>
                            {step === 3 ? "Learning Curve" : step === 4 ? "Train vs Validation Loss" : "Cost Landscape"}
                        </h3>
                        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                            {step === 2 && (
                                <button className="btn btn-secondary" onClick={() => setShow3D(!show3D)} style={{ padding: '0.25rem 0.5rem', fontSize: '0.8rem' }}>
                                    {show3D ? "Show 2D Contour" : "Show 3D Surface"}
                                </button>
                            )}
                            {isPlaying && <div className="state-badge">Optimizing...</div>}
                        </div>
                    </div>

                    {/* STEP 1 PLOT – always mounted, hidden when step !== 1 */}
                    <div style={{ display: step === 1 ? 'block' : 'none' }}>
                        {(() => {
                            const costM = calcCost(data, m, 0);
                            const curve = calc1DCurve();
                            const maxCost = Math.max(...curve.cost_vals);
                            const traces = [
                                {
                                    x: curve.m_vals,
                                    y: curve.cost_vals,
                                    mode: 'lines',
                                    type: 'scatter',
                                    line: { color: '#10b981', width: 2 },
                                    name: 'Cost Function J(m)'
                                },
                                {
                                    x: [m],
                                    y: [costM],
                                    mode: 'markers',
                                    type: 'scatter',
                                    marker: { color: '#ef4444', size: 10, symbol: 'diamond' },
                                    cliponaxis: false,
                                    name: 'Current m'
                                }
                            ];
                            if (pendingGradient && step === 1) {
                                const size = 2.0;
                                traces.push({
                                    x: [m - size, m + size],
                                    y: [costM - size * pendingGradient.dj_dm, costM + size * pendingGradient.dj_dm],
                                    mode: 'lines',
                                    type: 'scatter',
                                    line: { color: '#facc15', width: 4, dash: 'dot' },
                                    name: 'Slope (Derivative)'
                                });
                            }
                            return (
                                <Plot
                                    data={traces}
                                    onClick={(evt) => handlePlotClick(evt, true)}
                                    onRelayout={(ed) => {
                                        if (ed['xaxis.range[0]'] !== undefined) {
                                            setMView({ min: ed['xaxis.range[0]'], max: ed['xaxis.range[1]'] });
                                        } else if (ed['xaxis.autorange']) {
                                            setMView({ min: -2, max: 6 });
                                        }
                                    }}
                                    layout={{
                                        showlegend: false,
                                        uirevision: 'keep-zoom',
                                        autosize: true,
                                        paper_bgcolor: 'transparent',
                                        plot_bgcolor: 'transparent',
                                        font: { color: pTheme.fontColor },
                                        margin: { t: 30, r: 20, l: 80, b: 80 },
                                        xaxis: { title: { text: 'm (slope)', standoff: 15 }, gridcolor: pTheme.gridColor, automargin: true },
                                        yaxis: { title: { text: 'Cost J(m)', standoff: 15 }, gridcolor: pTheme.gridColor, automargin: true }
                                    }}
                                    useResizeHandler={true}
                                    style={{ width: "100%", height: "350px" }}
                                />
                            );
                        })()}
                    </div>
                    {/* STEP 2 CONTOUR – always mounted, hidden when not active */}
                    <div style={{ display: (step === 2 && !show3D) ? 'block' : 'none' }}>
                        {(() => {
                            const surface = calc2DSurface();
                            const traces = [
                                {
                                    z: surface.cost_grid,
                                    x: surface.m_vals,
                                    y: surface.b_vals,
                                    type: 'contour',
                                    colorscale: 'Viridis',
                                    contours: { coloring: 'heatmap' },
                                    name: 'Cost Surface'
                                },
                                {
                                    x: history.map(h => h.m),
                                    y: history.map(h => h.b),
                                    mode: 'lines+markers',
                                    type: 'scatter',
                                    marker: { color: '#ef4444', size: 6 },
                                    line: { color: '#ef4444', width: 2 },
                                    name: 'Descent Path'
                                }
                            ];
                            if (pendingGradient && step === 2) {
                                const size = 2.0;
                                traces.push({
                                    x: [m, m - size * pendingGradient.dj_dm],
                                    y: [b, b - size * pendingGradient.dj_db],
                                    mode: 'lines',
                                    type: 'scatter',
                                    line: { color: '#facc15', width: 4, dash: 'dot' },
                                    name: 'Gradient Direction'
                                });
                            }
                            return (
                                <Plot
                                    data={traces}
                                    onClick={(evt) => handlePlotClick(evt, false)}
                                    layout={{
                                        showlegend: false,
                                        autosize: true,
                                        paper_bgcolor: 'transparent',
                                        plot_bgcolor: 'transparent',
                                        font: { color: pTheme.fontColor },
                                        margin: { t: 30, r: 20, l: 80, b: 80 },
                                        xaxis: { title: { text: 'm (slope)', standoff: 15 }, gridcolor: pTheme.gridColor, automargin: true },
                                        yaxis: { title: { text: 'b (intercept)', standoff: 15 }, gridcolor: pTheme.gridColor, automargin: true }
                                    }}
                                    useResizeHandler={true}
                                    style={{ width: "100%", height: "350px" }}
                                />
                            );
                        })()}
                    </div>
                    {/* STEP 2 3D SURFACE – always mounted, hidden when not active */}
                    <div style={{ display: (step === 2 && show3D) ? 'block' : 'none' }}>
                        {(() => {
                            const surface = calc2DSurface();
                            const traces = [
                                {
                                    z: surface.cost_grid,
                                    x: surface.m_vals,
                                    y: surface.b_vals,
                                    type: 'surface',
                                    colorscale: 'Viridis',
                                    showscale: false,
                                    name: 'Cost Surface',
                                    hovertemplate: 'm: %{x:.3f}<br>b: %{y:.3f}<br>cost: %{z:.2f}<extra></extra>'
                                },
                                {
                                    x: history.map(h => h.m),
                                    y: history.map(h => h.b),
                                    z: history.map(h => h.cost + 1),
                                    mode: 'lines+markers',
                                    type: 'scatter3d',
                                    marker: { color: '#ef4444', size: 4 },
                                    line: { color: '#ef4444', width: 4 },
                                    name: 'Descent Path',
                                    hovertemplate: 'm: %{x:.3f}<br>b: %{y:.3f}<br>cost: %{z:.2f}<extra></extra>'
                                }
                            ];
                            if (pendingGradient && step === 2) {
                                const arrowLen = 1.5;
                                const gradMag = Math.sqrt(
                                    pendingGradient.dj_dm ** 2 + pendingGradient.dj_db ** 2
                                ) || 1;
                                const dm_n = (pendingGradient.dj_dm / gradMag) * arrowLen;
                                const db_n = (pendingGradient.dj_db / gradMag) * arrowLen;
                                const J0 = calcCost(data, m, b);
                                const J1 = calcCost(data, m - dm_n, b - db_n);
                                const nDashes = 12;
                                const gx = [], gy = [], gz = [];
                                for (let i = 0; i <= nDashes; i++) {
                                    if (i % 2 === 0) {
                                        const t1 = i / nDashes;
                                        const t2 = (i + 1) / nDashes;
                                        gx.push(m - dm_n * t1, m - dm_n * t2, null);
                                        gy.push(b - db_n * t1, b - db_n * t2, null);
                                        gz.push(J0 + (J1 - J0) * t1, J0 + (J1 - J0) * t2, null);
                                    }
                                }
                                traces.push({
                                    x: gx, y: gy, z: gz,
                                    mode: 'lines',
                                    type: 'scatter3d',
                                    line: { color: '#facc15', width: 6 },
                                    name: 'Gradient Direction'
                                });
                            }
                            return (
                                <Plot
                                    data={traces}
                                    onClick={(evt) => handlePlotClick(evt, false)}
                                    layout={{
                                        showlegend: false,
                                        autosize: true,
                                        uirevision: 'keep-camera',
                                        paper_bgcolor: 'transparent',
                                        plot_bgcolor: 'transparent',
                                        font: { color: pTheme.fontColor },
                                        margin: { t: 0, r: 0, l: 0, b: 0 },
                                        scene: {
                                            xaxis: { title: { text: 'm' }, gridcolor: pTheme.gridColor },
                                            yaxis: { title: { text: 'b' }, gridcolor: pTheme.gridColor },
                                            zaxis: { title: { text: 'cost' }, gridcolor: pTheme.gridColor }
                                        }
                                    }}
                                    useResizeHandler={true}
                                    style={{ width: "100%", height: "350px" }}
                                />
                            );
                        })()}
                    </div>

                    {/* STEP 4 - Train vs Validation Accuracy Cards */}
                    <div style={{ display: step === 4 ? 'flex' : 'none', flexDirection: 'column', gap: '0.75rem', flex: 1 }}>
                        {MODELS_4.map((m4, idx) => {
                            const trainAcc = accuracy(m4.predict, TRAIN_DATA_4);
                            const valAcc = accuracy(m4.predict, VAL_DATA_4);
                            const isActive = idx === selectedModel4;
                            return (
                                <div
                                    key={m4.key}
                                    onClick={() => setSelectedModel4(idx)}
                                    style={{
                                        padding: '0.9rem 1rem',
                                        background: isActive ? `${m4.color}15` : 'var(--card-bg)',
                                        borderRadius: '12px',
                                        border: `${isActive ? '2px' : '1px'} solid ${isActive ? m4.color : m4.color + '55'}`,
                                        boxShadow: isActive ? `0 4px 18px ${m4.color}33` : '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                                        cursor: 'pointer',
                                        transition: 'all 0.2s'
                                    }}
                                >
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.6rem' }}>
                                        <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: m4.color }} />
                                        <strong style={{ color: m4.color }}>{m4.name}</strong>
                                    </div>
                                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
                                        <div style={{ padding: '0.5rem', background: 'rgba(59, 130, 246, 0.1)', borderRadius: '8px', textAlign: 'center', border: '1px solid rgba(59, 130, 246, 0.3)' }}>
                                            <div style={{ fontSize: '0.7rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Train Acc</div>
                                            <div style={{ fontSize: '1.2rem', fontWeight: 700, fontFamily: 'monospace', color: '#3b82f6' }}>{(trainAcc * 100).toFixed(1)}%</div>
                                        </div>
                                        <div style={{ padding: '0.5rem', background: 'rgba(245, 158, 11, 0.1)', borderRadius: '8px', textAlign: 'center', border: '1px solid rgba(245, 158, 11, 0.3)' }}>
                                            <div style={{ fontSize: '0.7rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Val Acc</div>
                                            <div style={{ fontSize: '1.2rem', fontWeight: 700, fontFamily: 'monospace', color: '#f59e0b' }}>
                                                {revealValidation ? `${(valAcc * 100).toFixed(1)}%` : '???'}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            );
                        })}
                    </div>

                    {/* STEP 3 PLOT - Learning Curve */}
                    <div style={{ display: step === 3 ? 'block' : 'none', flex: 1 }}>
                        <Plot
                            data={[
                                {
                                    x: nnLossHistory.map(h => h.epoch),
                                    y: nnLossHistory.map(h => h.loss),
                                    mode: 'lines',
                                    type: 'scatter',
                                    line: { color: '#10b981', width: 2 },
                                    name: 'Cross Entropy'
                                }
                            ]}
                            layout={{
                                showlegend: false,
                                autosize: true,
                                paper_bgcolor: 'transparent',
                                plot_bgcolor: 'transparent',
                                font: { color: pTheme.fontColor },
                                margin: { t: 30, r: 20, l: 80, b: 80 },
                                xaxis: { title: { text: 'Epochs', standoff: 15 }, gridcolor: pTheme.gridColor, automargin: true },
                                yaxis: { title: { text: 'Log Loss', standoff: 15 }, gridcolor: pTheme.gridColor, automargin: true, rangemode: 'tozero' }
                            }}
                            useResizeHandler={true}
                            style={{ width: "100%", height: "350px" }}
                        />
                    </div>

                    {step === 4 ? (
                        <div style={{ display: 'flex', gap: '1rem', marginTop: 'auto' }}>
                            <button className="btn" onClick={() => setRevealValidation(v => !v)}>
                                {revealValidation ? <EyeOff size={18} /> : <Eye size={18} />}
                                {revealValidation ? 'Hide Validation' : 'Reveal Validation Data'}
                            </button>
                        </div>
                    ) : (
                        <>
                            <div style={{ display: 'flex', gap: '1rem', marginTop: 'auto' }}>
                                <button className="btn" onClick={() => { setIsPlaying(!isPlaying); setPendingGradient(null); }}>
                                    {isPlaying ? <Pause size={18} /> : <Play size={18} />}
                                    {isPlaying ? "Pause" : "Play"}
                                </button>
                                {step !== 3 && (
                                    <button className="btn btn-secondary" onClick={handleManualStep} disabled={isPlaying}>
                                        <FastForward size={18} /> {pendingGradient ? "Apply Step" : "Show Slope"}
                                    </button>
                                )}
                                <button className="btn btn-secondary" onClick={resetModel}>
                                    <RotateCcw size={18} /> Reset
                                </button>
                            </div>

                            <div className="slider-container" style={{ marginTop: '1rem' }}>
                                <div className="slider-header">
                                    <span>Learning Rate (Log Scale): {Number(lr).toFixed(4)}</span>
                                </div>
                                <input
                                    type="range"
                                    min="-3"
                                    max={Math.log10(0.5)}
                                    step="0.01"
                                    value={Math.log10(lr || 0.001)}
                                    onChange={(e) => setLr(parseFloat(Math.pow(10, parseFloat(e.target.value)).toFixed(4)))}
                                />
                            </div>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}
