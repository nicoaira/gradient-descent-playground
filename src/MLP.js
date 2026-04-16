export class MLP {
    constructor(hiddenSize) {
        this.H = hiddenSize;
        this.W1 = [
            Array(this.H).fill(0).map(() => (Math.random() - 0.5) * 2),
            Array(this.H).fill(0).map(() => (Math.random() - 0.5) * 2)
        ];
        this.b1 = Array(this.H).fill(0);
        this.W2 = Array(this.H).fill(0).map(() => (Math.random() - 0.5) * 2);
        this.b2 = 0;
    }

    forward(pts) {
        this.cache = [];
        const preds = [];
        for (let pt of pts) {
            const X = [pt.x, pt.y];
            const A1 = Array(this.H).fill(0);
            for (let i = 0; i < this.H; i++) {
                let z = X[0] * this.W1[0][i] + X[1] * this.W1[1][i] + this.b1[i];
                A1[i] = Math.tanh(z);
            }
            let Z2 = this.b2;
            for (let i = 0; i < this.H; i++) {
                Z2 += A1[i] * this.W2[i];
            }
            const A2 = 1 / (1 + Math.exp(-Z2));
            preds.push(A2);
            this.cache.push({ X, A1, Z2, A2 });
        }
        return preds;
    }

    predict(x, y) {
        const A1 = Array(this.H).fill(0);
        for (let i = 0; i < this.H; i++) {
            let z = x * this.W1[0][i] + y * this.W1[1][i] + this.b1[i];
            A1[i] = Math.tanh(z);
        }
        let Z2 = this.b2;
        for (let i = 0; i < this.H; i++) {
            Z2 += A1[i] * this.W2[i];
        }
        return 1 / (1 + Math.exp(-Z2));
    }

    backward(pts, lr) {
        let dW1 = [Array(this.H).fill(0), Array(this.H).fill(0)];
        let db1 = Array(this.H).fill(0);
        let dW2 = Array(this.H).fill(0);
        let db2 = 0;

        const N = pts.length;
        let loss = 0;

        for (let i = 0; i < N; i++) {
            const c = this.cache[i];
            const yMain = pts[i].label;

            const p = Math.max(1e-7, Math.min(1 - 1e-7, c.A2));
            loss += - (yMain * Math.log(p) + (1 - yMain) * Math.log(1 - p));

            const dZ2 = c.A2 - yMain;
            db2 += dZ2;
            for (let j = 0; j < this.H; j++) {
                dW2[j] += dZ2 * c.A1[j];

                let dA1_j = dZ2 * this.W2[j];
                let dZ1_j = dA1_j * (1 - c.A1[j] * c.A1[j]); // tanh deriv

                db1[j] += dZ1_j;
                dW1[0][j] += dZ1_j * c.X[0];
                dW1[1][j] += dZ1_j * c.X[1];
            }
        }

        // Apply gradients
        this.b2 -= lr * (db2 / N);
        for (let j = 0; j < this.H; j++) {
            this.W2[j] -= lr * (dW2[j] / N);
            this.b1[j] -= lr * (db1[j] / N);
            this.W1[0][j] -= lr * (dW1[0][j] / N);
            this.W1[1][j] -= lr * (dW1[1][j] / N);
        }

        return loss / N;
    }
}

export const generateClassificationData = (type = 'circles') => {
    const pts = [];
    const nPoints = 200;

    if (type === 'spirals') {
        for (let i = 0; i < nPoints / 2; i++) {
            const n = i / (nPoints / 2); // 0 to 1
            const r = 5 * n;
            const t = 1.75 * Math.PI * n; // 1.75 turns

            // Class 0
            pts.push({
                x: r * Math.sin(t) + (Math.random() - 0.5) * 0.5,
                y: r * Math.cos(t) + (Math.random() - 0.5) * 0.5,
                label: 0
            });
            // Class 1 (offset by PI)
            pts.push({
                x: -r * Math.sin(t) + (Math.random() - 0.5) * 0.5,
                y: -r * Math.cos(t) + (Math.random() - 0.5) * 0.5,
                label: 1
            });
        }
    } else if (type === 'moons') {
        for (let i = 0; i < nPoints / 2; i++) {
            const t = Math.PI * (i / (nPoints / 2));
            pts.push({
                x: 4 * Math.cos(t) - 2 + (Math.random() - 0.5) * 0.5,
                y: 4 * Math.sin(t) + (Math.random() - 0.5) * 0.5,
                label: 0
            });
            pts.push({
                x: 4 * (1 - Math.cos(t)) - 2 + (Math.random() - 0.5) * 0.5,
                y: 4 * (0.5 - Math.sin(t)) + (Math.random() - 0.5) * 0.5,
                label: 1
            });
        }
    } else {
        // default: circles
        for (let i = 0; i < nPoints / 2; i++) {
            const r1 = Math.random() * 2;
            const a1 = Math.random() * 2 * Math.PI;
            pts.push({ x: r1 * Math.cos(a1), y: r1 * Math.sin(a1), label: 0 });

            const r2 = 3.5 + Math.random() * 1.5;
            const a2 = Math.random() * 2 * Math.PI;
            pts.push({ x: r2 * Math.cos(a2), y: r2 * Math.sin(a2), label: 1 });
        }
    }
    return pts;
};
