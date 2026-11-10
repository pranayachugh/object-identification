# ===== Rocky sonar, NO-SERVO version: record while a shape is turned by hand, then identify it (single script) =====
# Run in a terminal:  python rocky_rotate.py      Needs rocky_rotate.ino uploaded, Serial Monitor CLOSED,
# and: pip install pyserial scikit-learn pandas matplotlib numpy
import csv, os, time, datetime
import numpy as np, pandas as pd, matplotlib.pyplot as plt, serial, serial.tools.list_ports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix

PORT, BAUD, CSV = "COM9", 115200, "rocky_rot_scans.csv"   # PORT: the one shown in the Arduino IDE (Tools > Port)
REC_S = 10            # seconds per recording; must equal REC_MS/1000 in rocky_rotate.ino
NB = 60               # a recording is cut into NB equal time slices (about 1/NB of the turn each)
OBJ_DEPTH = 12.0      # cm: echoes farther than this behind the nearest ones are background, not the shape
MIN_CONF, MAX_RANGE = 0.5, 60.0

# ---------------------------------------------------------------- recording
def record(ser):
    """Countdown, then one recording while you turn the shape ONE full turn. Returns time (s) and distance (cm, NaN = no echo)."""
    for s in (3, 2, 1): print(f"  {s}...", end="", flush=True); time.sleep(1)
    ser.reset_input_buffer(); ser.write(b"R"); t, d = [], []
    print(f"\n  GO! turn the shape ONE full turn, slowly and steadily, taking about {REC_S} seconds")
    while True:
        ln = ser.readline().decode(errors="ignore").strip()
        if ln == "END": break
        if ln == "": raise TimeoutError("no data from the Arduino: check the port, the wiring and that rocky_rotate.ino is uploaded")
        p = ln.split(",")
        if len(p) == 2 and p[0].isdigit():
            try: v = float(p[1])
            except ValueError: continue
            t.append(int(p[0]) / 1000); d.append(v if v > 0 else np.nan)
    t, d = np.array(t), np.array(d); print(f"  STOP. {len(t)} readings, {int(np.isfinite(d).sum())} with an echo"); return t, d

# ---------------------------------------------------------------- features
def prep(t, d):
    """Cut one recording into NB time slices. Returns (v = slice had an object echo, db = median distance per slice, near) or None."""
    ok = np.isfinite(d)
    if ok.sum() < 5 or t.max() <= 0: return None
    near = np.percentile(d[ok], 10); ok &= d < near + OBJ_DEPTH                      # keep the shape, drop far background
    b = np.minimum((t / t.max() * NB).astype(int), NB - 1)
    v = np.array([ok[b == k].any() for k in range(NB)])
    v = v | (np.roll(v, 1) & np.roll(v, -1)); v = v & (np.roll(v, 1) | np.roll(v, -1))   # fill 1-slice gaps, drop 1-slice blips
    if not v.any(): return None
    db = np.array([np.median(d[(b == k) & ok]) if v[k] and ((b == k) & ok).any() else np.nan for k in range(NB)])
    return v, db, near

def features(t, d):
    """Count the echo bursts (flat faces turning past the sensor) and describe their rhythm. A full turn is treated as a circle."""
    r = prep(t, d)
    if r is None: return None
    v, db, near = r; runs = 0 if v.all() else int(np.sum(v & ~np.roll(v, 1)))        # circular count: start and end are the same direction
    x = v - v.mean(); ac = np.fft.irfft(np.abs(np.fft.rfft(x)) ** 2, NB); lo, hi = int(NB * .12), int(NB * .6)
    lag = lo + int(np.argmax(ac[lo:hi])) if ac[0] > 1e-9 else 0
    dv = db[np.isfinite(db)]
    return dict(near=near, n_bursts=runs, valid_frac=v.mean(), duty=v.mean() / max(runs, 1), d_range=np.ptp(dv), d_std=dv.std(),
                ac_peak=ac[lag] / ac[0] if lag else 0, period_frac=lag / NB if lag else 1.0)

def load_features():
    if not os.path.exists(CSV): print("no recordings yet: choose 1 and record some first"); return None
    df = pd.read_csv(CSV); df["dist_cm"] = df.dist_cm.where(df.dist_cm > 0); rows = []
    for rid, g in df.groupby("rec_id"):
        f = features(g.t_ms.values / 1000, g.dist_cm.values)
        if f is not None: rows.append((g.label.iloc[0], f))
    if not rows: print("all recordings were empty (nothing detected). Check the sensor with option 3 first"); return None
    X = pd.DataFrame([f for _, f in rows]); y = np.array([l for l, _ in rows])
    print(f"{len(y)} usable recordings ({df.rec_id.nunique() - len(y)} empty ones skipped): {dict(pd.Series(y).value_counts())}"); return X, y

def train():
    r = load_features()
    if r is None: return None
    X, y = r; counts = pd.Series(y).value_counts()
    if len(counts) < 2 or counts.min() < 3: print("need at least 2 shapes with 3+ recordings each (about 15-20 each is better): choose 1 and record more"); return None
    k = min(5, counts.min()); rf = RandomForestClassifier(300, random_state=0).fit(X, y)
    print(f"expected accuracy ({k}-fold cross-validation): {cross_val_score(RandomForestClassifier(300, random_state=0), X, y, cv=StratifiedKFold(k, shuffle=True, random_state=0)).mean():.0%}")
    return rf, list(X.columns)

# ---------------------------------------------------------------- display
def show(t, d, p=None, save="rocky_rot_view.png"):
    """Left: distance over time. Middle: 'rotation view' (a dot in each direction where a face answered). Right: model probabilities."""
    ok = np.isfinite(d); r = prep(t, d); fig = plt.figure(figsize=(16, 4.8))
    a0, a1, a2 = fig.add_subplot(131), fig.add_subplot(132, projection="polar"), fig.add_subplot(133)
    a0.plot(t, d, ".", c="tab:blue", ms=4, label="echo"); a0.scatter(t[~ok], np.zeros((~ok).sum()), marker="x", c="r", s=10, label="no echo")
    verdict = "NOTHING DETECTED"
    if r is None: print("almost no echoes: nothing detected (shape too far, too small, or the sensor is not pointing at it)")
    else:
        v, db, near = r; f = features(t, d); T = t.max()
        for k in np.flatnonzero(v): a0.axvspan(k * T / NB, (k + 1) * T / NB, color="tab:red", alpha=.15)
        a1.scatter(2 * np.pi * np.flatnonzero(v) / NB, np.nanmax(db) + 3 - db[v], c="tab:red", s=22)
        verdict = "(recording only, no model)"
        if p is not None:
            best = max(p, key=p.get)
            verdict = "NO OBJECT IN RANGE" if near > MAX_RANGE else f"{best.upper()}  ({p[best]:.0%})" if p[best] >= MIN_CONF else f"UNCERTAIN (best guess {best}, {p[best]:.0%})"
            a2.barh(list(p), list(p.values()), color=["tab:red" if k == best else "gray" for k in p]); a2.set(xlim=(0, 1), title="class probabilities")
        print(f"echo bursts per turn: {f['n_bursts']} | echoes in {f['valid_frac']:.0%} of the turn | distance changes by {f['d_range']:.1f} cm | nearest {near:.0f} cm\n=> {verdict}")
    a0.set(title=f"distance over time  ->  {verdict}", xlabel="time (s)", ylabel="distance (cm)"); a0.legend(fontsize=8)
    a1.set_title("rotation view: dots = directions where a face\nreturned an echo", fontsize=9); a1.set_yticklabels([])
    plt.tight_layout(); plt.savefig(save, dpi=100); print(f"(picture saved as {save}; close the window to continue)"); plt.show()

# ---------------------------------------------------------------- the modes
def collect(ser):
    session = datetime.datetime.now().strftime("%Y%m%d_%H%M%S"); new = not os.path.exists(CSV)
    rec_id = 0 if new else 1 + max(int(r["rec_id"]) for r in csv.DictReader(open(CSV)))
    f = open(CSV, "a", newline=""); w = csv.writer(f)
    if new: w.writerow(["rec_id", "session", "label", "t_ms", "dist_cm"])
    while True:
        label = input("\nshape name (e.g. circle / box / triangle; blank = back to menu): ").strip().lower()
        if not label: break
        n = int(input("how many recordings? [15] ") or 15)
        for i in range(n):
            input(f"  [{i + 1}/{n}] put the {label} centred on the turntable at a NEW starting angle, hand on it, press Enter to start the countdown ")
            t, d = record(ser)
            for tt, v in zip(t, d): w.writerow([rec_id, session, label, int(round(tt * 1000)), -1 if np.isnan(v) else round(float(v), 1)])
            f.flush(); rec_id += 1
    f.close(); print(f"saved to {os.path.abspath(CSV)}")

def identify_loop(ser):
    m = train()
    if m is None: return
    rf, cols = m
    while input("\nput a shape on the turntable, press Enter to start (q = back to menu): ").strip().lower() != "q":
        t, d = record(ser); f = features(t, d)
        show(t, d, None if f is None else dict(zip(rf.classes_, rf.predict_proba(pd.DataFrame([f], columns=cols))[0])))

def analyse():
    r = load_features()
    if r is None: return
    X, y = r; counts = pd.Series(y).value_counts(); classes = sorted(counts.index); col = dict(zip(classes, plt.cm.tab10.colors))
    print("\nfeature averages per shape:\n" + X.groupby(y).mean().round(2).T.to_string())
    fig, ax = plt.subplots(2, 2, figsize=(12, 9)); rng = np.random.default_rng(0)
    counts[classes].plot.bar(ax=ax[0, 0], color=[col[c] for c in classes]); ax[0, 0].set_title("recordings per shape")
    for i, c in enumerate(classes):
        m = y == c; ax[0, 1].scatter(i + rng.uniform(-.15, .15, m.sum()), X.n_bursts[m], color=col[c]); ax[1, 0].scatter(X.valid_frac[m], X.d_range[m], color=col[c], label=c)
    ax[0, 1].set(xticks=range(len(classes)), xticklabels=classes, title="echo bursts per turn (box ~4, triangle ~3, circle ~0-1)", ylabel="bursts")
    ax[1, 0].set(title="how much of the turn had an echo vs distance change", xlabel="fraction of turn with echo", ylabel="distance change (cm)"); ax[1, 0].legend()
    k = min(5, counts.min())
    if k >= 2:
        pred = cross_val_predict(RandomForestClassifier(300, random_state=0), X, y, cv=StratifiedKFold(k, shuffle=True, random_state=0)); cm = confusion_matrix(y, pred, labels=classes)
        print(f"\n{k}-fold cross-validation accuracy {np.mean(pred == y):.0%}\nconfusion matrix (rows = true, columns = predicted) {classes}\n{cm}")
        ax[1, 1].imshow(cm, cmap="Blues"); ax[1, 1].set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes, yticklabels=classes, title="confusion matrix (cross-validated)")
        for i in range(len(classes)):
            for j in range(len(classes)): ax[1, 1].text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout(); plt.savefig("rocky_rot_analysis.png", dpi=100); print("(picture saved as rocky_rot_analysis.png; close the window to continue)"); plt.show()

def main():
    print("serial ports found:", [p.device for p in serial.tools.list_ports.comports()])
    try: ser = serial.Serial(PORT, BAUD, timeout=3)
    except serial.SerialException as e: print(f"could not open {PORT}: {e}\nClose the Arduino Serial Monitor, check PORT at the top of this file, and try again."); return
    with ser:
        time.sleep(2.5)                                                              # the Arduino resets when the port opens
        while True:
            c = input("\n1 = record training data | 2 = identify a shape | 3 = test recording | 4 = analyse saved data | q = quit\n> ").strip().lower()
            if c == "q": break
            elif c == "1": collect(ser)
            elif c == "2": identify_loop(ser)
            elif c == "3": t, d = record(ser); show(t, d)
            elif c == "4": analyse()

if __name__ == "__main__":
    main()
