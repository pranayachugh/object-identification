# ===== Rocky sonar ALL-IN-ONE: collect scans, test the sensor, identify shapes (single script) =====
# Run in a terminal:  python rocky_all_in_one.py     (VS Code notebooks hide input() prompts in a box at the top of the window)
# Needs: rocky_scan.ino uploaded to the Arduino, Serial Monitor CLOSED, and: pip install pyserial scikit-learn pandas matplotlib
import csv, os, time, datetime
import numpy as np, pandas as pd, matplotlib.pyplot as plt, serial, serial.tools.list_ports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix

PORT, BAUD, CSV = "COM9", 115200, "rocky_scans.csv"     # PORT: the one shown in the Arduino IDE (Tools > Port)
A_MIN, A_MAX, STEP = 30, 150, 2                         # must match rocky_scan.ino
OBJ_DEPTH = 25.0    # cm: echoes within this distance of the nearest echo count as "the object"
MIN_CONF = 0.5      # below this the answer is shown as UNCERTAIN
MAX_RANGE = 60.0    # cm: if the nearest echo is farther than this, nothing is "in front of Rocky"
ANG = np.arange(A_MIN, A_MAX + 1, STEP); TH = np.radians(ANG - 90)

# ---------------------------------------------------------------- talking to the Arduino
def read_scan(ser):
    """Send 'S', read one sweep. Returns distances aligned with ANG (NaN = no echo). Prints a dot per reading so you can see it working."""
    ser.reset_input_buffer(); ser.write(b"S"); got = {}; print("scanning ", end="", flush=True)
    while True:
        ln = ser.readline().decode(errors="ignore").strip()
        if ln == "END": break
        if ln == "": raise TimeoutError("\nno data from the Arduino: check the port, the wiring and that rocky_scan.ino is uploaded")
        if "," in ln and ln.replace(",", "").replace(".", "").replace("-", "").isdigit():
            a, v = ln.split(","); got[int(a)] = float(v) if float(v) > 0 else np.nan; print(".", end="", flush=True)
    d = np.array([got.get(int(a), np.nan) for a in ANG]); print(f" done: {np.isfinite(d).sum()}/{len(d)} angles had an echo"); return d

# ---------------------------------------------------------------- features + model
def features(d):
    """Distance-invariant descriptors of the object's top-down cross-section."""
    ok = np.isfinite(d)
    if ok.sum() < 3: return None
    near = np.nanmin(d); m = ok & (d < near + OBJ_DEPTH); i = np.flatnonzero(m)
    x, y = d[m] * np.sin(TH[m]), d[m] * np.cos(TH[m]); n, span = m.sum(), i[-1] - i[0] + 1
    ok2 = n >= 4 and np.ptp(x) > 0
    q = np.polyfit(x, y, 2) if ok2 else [0, 0, y.mean()]; l = np.polyfit(x, y, 1) if ok2 else [0, y.mean()]
    return dict(n_obj=n, width=np.ptp(x), depth=np.ptp(y), ang_width=span * STEP, gap_frac=1 - n / span, curvature=q[0],
                slope=abs(l[0]), lin_rms=np.sqrt(np.mean((y - np.polyval(l, x)) ** 2)), quad_rms=np.sqrt(np.mean((y - np.polyval(q, x)) ** 2)),
                asym=abs(y[x > x.mean()].mean() - y[x < x.mean()].mean()) if (x > x.mean()).any() and (x < x.mean()).any() else 0, valid_frac=ok.mean())

def load_features():
    """Features for every saved scan. Returns (X, labels, sessions) or None."""
    if not os.path.exists(CSV): print("no rocky_scans.csv yet: choose 1 and collect some scans first"); return None
    df = pd.read_csv(CSV); df["dist_cm"] = df.dist_cm.where(df.dist_cm > 0)
    Dm = df.pivot(index="scan_id", columns="angle", values="dist_cm").reindex(columns=ANG); meta = df.groupby("scan_id")[["label", "session"]].first()
    rows = [(s, features(r.values.astype(float))) for s, r in Dm.iterrows()]; rows = [(s, f) for s, f in rows if f is not None]
    if not rows: print("all saved scans were empty: nothing was detected. Check the sensor first (choose 3)"); return None
    X = pd.DataFrame([f for _, f in rows]); ids = [s for s, _ in rows]
    print(f"{len(ids)} usable scans ({len(Dm) - len(ids)} empty ones skipped): {dict(pd.Series(meta.loc[ids, 'label'].values).value_counts())}")
    return X, meta.loc[ids, "label"].values, meta.loc[ids, "session"].values

def train():
    """Train on rocky_scans.csv. Returns (model, feature columns) or None if there is not enough data yet."""
    r = load_features()
    if r is None: return None
    X, y, _ = r; counts = pd.Series(y).value_counts()
    if len(counts) < 2 or counts.min() < 3: print("need at least 2 shapes with 3+ scans each (about 30 each is better): choose 1 and collect more"); return None
    rf = RandomForestClassifier(300, random_state=0).fit(X, y); k = min(5, counts.min())
    print(f"expected accuracy ({k}-fold cross-validation): {cross_val_score(RandomForestClassifier(300, random_state=0), X, y, cv=StratifiedKFold(k, shuffle=True, random_state=0)).mean():.0%}")
    return rf, list(X.columns)

def analyse():
    """Data analysis: scans per shape, feature averages, cross-validated accuracy, accuracy on a NEW sitting, confusion matrix, plots."""
    r = load_features()
    if r is None: return
    X, y, grp = r; counts = pd.Series(y).value_counts(); classes = sorted(counts.index); col = dict(zip(classes, plt.cm.tab10.colors))
    print("\nfeature averages per shape:\n" + X.groupby(y).mean().round(2).T.to_string())
    k = min(5, counts.min()); fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    counts[classes].plot.bar(ax=ax[0, 0], color=[col[c] for c in classes]); ax[0, 0].set_title("scans per shape")
    for c in classes: m = y == c; ax[0, 1].scatter(X.width[m], X.curvature[m], color=col[c], label=c, s=18)
    ax[0, 1].set(title="object width vs outline curvature", xlabel="width (cm)", ylabel="curvature"); ax[0, 1].legend()
    if k >= 2:
        rf = RandomForestClassifier(300, random_state=0); pred = cross_val_predict(rf, X, y, cv=StratifiedKFold(k, shuffle=True, random_state=0)); cm = confusion_matrix(y, pred, labels=classes)
        print(f"\n{k}-fold cross-validation accuracy {np.mean(pred == y):.0%}\nconfusion matrix (rows = true, columns = predicted) {classes}\n{cm}")
        if len(set(grp)) >= 2:
            pg = cross_val_predict(rf, X, y, cv=GroupKFold(min(5, len(set(grp)))), groups=grp); print(f"accuracy on sittings never seen in training: {np.mean(pg == y):.0%}   <- the honest number")
        else: print("only one recording sitting so far: record again on another day or after a break to get the honest 'new sitting' accuracy")
        ax[1, 0].imshow(cm, cmap="Blues"); ax[1, 0].set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes, yticklabels=classes, title="confusion matrix (cross-validated)")
        for i in range(len(classes)):
            for j in range(len(classes)): ax[1, 0].text(j, i, cm[i, j], ha="center", va="center")
    pd.Series(RandomForestClassifier(300, random_state=0).fit(X, y).feature_importances_, X.columns).sort_values().plot.barh(ax=ax[1, 1]); ax[1, 1].set_title("which features the model relies on")
    plt.tight_layout(); plt.savefig("rocky_analysis.png", dpi=100); print("(picture saved as rocky_analysis.png; close the window to continue)"); plt.show()

def show(d, p=None, save="rocky_view.png"):
    """Plot what the sonar sees. p = {shape: probability} from the model, or None to only show the scan."""
    ok = np.isfinite(d); x, yy = d * np.sin(TH), d * np.cos(TH); f = features(d); fig, ax = plt.subplots(1, 3, figsize=(16, 4.8)); near = np.nanmin(d) if ok.any() else np.nan
    ax[0].scatter(x[ok], yy[ok], c="lightgray", s=22, label="echoes"); ax[0].plot(0, 0, "k^", ms=10, label="sensor")
    verdict = "NOTHING DETECTED"
    if f is None: print("fewer than 3 echoes: nothing detected (object too far, too small, or tilted so the sound bounced away)")
    else:
        m = ok & (d < near + OBJ_DEPTH); ax[0].scatter(x[m], yy[m], c="tab:red", s=34, label="object echoes")
        if m.sum() >= 4 and np.ptp(x[m]) > 0:
            q = np.polyfit(x[m], yy[m], 2); xs = np.linspace(x[m].min(), x[m].max(), 50); ax[0].plot(xs, np.polyval(q, xs), "r--", lw=1, label="fitted outline")
        verdict = "(scan only, no model)"
        if p is not None:
            best = max(p, key=p.get)
            verdict = "NO OBJECT IN RANGE" if near > MAX_RANGE else f"{best.upper()}  ({p[best]:.0%})" if p[best] >= MIN_CONF else f"UNCERTAIN (best guess {best}, {p[best]:.0%})"
            ax[2].barh(list(p), list(p.values()), color=["tab:red" if k == best else "gray" for k in p]); ax[2].set(xlim=(0, 1), title="class probabilities")
        print(f"object: {int(m.sum())} echoes, width {f['width']:.1f} cm, depth {f['depth']:.1f} cm, nearest {near:.0f} cm, curvature {f['curvature']:.4f}\n=> {verdict}")
    ax[0].set(title=f"top-down view of what the sonar sees  ->  {verdict}", xlabel="x (cm)", ylabel="forward (cm)", aspect="equal"); ax[0].legend(fontsize=8)
    ax[1].plot(ANG, d, ".-", c="tab:blue"); ax[1].scatter(ANG[~ok], np.zeros((~ok).sum()), marker="x", c="r", s=14, label="no echo")
    ax[1].set(title="raw scan: distance vs servo angle", xlabel="servo angle (deg)", ylabel="distance (cm)"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(save, dpi=100); print(f"(picture saved as {save}; close the window to continue)"); plt.show()

# ---------------------------------------------------------------- the three modes
def collect(ser):
    session = datetime.datetime.now().strftime("%Y%m%d_%H%M%S"); new = not os.path.exists(CSV)
    scan_id = 0 if new else 1 + max(int(r["scan_id"]) for r in csv.DictReader(open(CSV)))
    f = open(CSV, "a", newline=""); w = csv.writer(f)
    if new: w.writerow(["scan_id", "session", "label", "angle", "dist_cm"])
    while True:
        label = input("\nshape name (e.g. circle / box / triangle; blank = back to menu): ").strip().lower()
        if not label: break
        n = int(input("how many scans? [20] ") or 20)
        for i in range(n):
            input(f"  [{i + 1}/{n}] move or rotate the {label} a little (20-35 cm away), then press Enter ")
            d = read_scan(ser)
            for a, v in zip(ANG, d): w.writerow([scan_id, session, label, int(a), -1 if np.isnan(v) else round(float(v), 1)])
            f.flush(); scan_id += 1
    f.close(); print(f"saved to {os.path.abspath(CSV)}")

def identify_loop(ser):
    t = train()
    if t is None: return
    rf, cols = t
    while input("\nplace a shape 20-35 cm in front of the sensor, press Enter to scan (q = back to menu): ").strip().lower() != "q":
        d = read_scan(ser); f = features(d)
        show(d, None if f is None else dict(zip(rf.classes_, rf.predict_proba(pd.DataFrame([f], columns=cols))[0])))

def main():
    print("serial ports found:", [p.device for p in serial.tools.list_ports.comports()])
    try: ser = serial.Serial(PORT, BAUD, timeout=3)
    except serial.SerialException as e: print(f"could not open {PORT}: {e}\nClose the Arduino Serial Monitor, check PORT at the top of this file, and try again."); return
    with ser:
        time.sleep(2.5)                                            # the Arduino resets when the port opens
        while True:
            c = input("\n1 = collect training scans | 2 = identify a shape | 3 = test scan (just show what the sonar sees) | 4 = analyse saved data | q = quit\n> ").strip().lower()
            if c == "q": break
            elif c == "1": collect(ser)
            elif c == "2": identify_loop(ser)
            elif c == "3": show(read_scan(ser))
            elif c == "4": analyse()

if __name__ == "__main__":
    main()
