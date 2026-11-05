"""
patch_index.py - applies the small web-page changes to YOUR existing frontend/index.html.
Run ONCE:  python patch_index.py        (a backup index.html.bak is made first)

Changes:
  * loads libraries from /vendor (works with no internet; run get_vendor.py first)
  * "Confidence" shows the real value (it was the fixed text 98.4%) + hand state, decision time, data rate
  * adds a letters strip (Backspace / Clear / Speak) fed by the server's "letter" messages
  * Training Config panel matches the new report (Model / Features / Smoothing)
Each change is skipped, with a message, if its target text is not found - nothing is ever half-written.
"""
import shutil
import sys
from pathlib import Path

path = Path(__file__).resolve().parent / "frontend" / "index.html"
if not path.exists():
    sys.exit(f"Not found: {path}")
raw = path.read_text(encoding="utf-8")
text = raw.replace("\r\n", "\n")
bak = path.with_suffix(".html.bak")
if not bak.exists():
    shutil.copyfile(path, bak)

STRIP = r'''<div className="mt-4 flex items-center justify-center gap-2 flex-wrap relative z-10">
                                            <div className="font-mono text-2xl tracking-widest min-h-[2rem] mr-2">{letters.join('')}</div>
                                            <button onClick={() => setLetters(p => p.slice(0, -1))} className="px-3 py-1 rounded border border-gray-500/50 text-xs hover:opacity-70">Backspace</button>
                                            <button onClick={() => setLetters([])} className="px-3 py-1 rounded border border-gray-500/50 text-xs hover:opacity-70">Clear</button>
                                            <button onClick={() => { if (window.speechSynthesis && letters.length) window.speechSynthesis.speak(new SpeechSynthesisUtterance(letters.join(''))); }} className="px-3 py-1 rounded border border-gray-500/50 text-xs hover:opacity-70">Speak</button>
                                        </div>'''

CONF = (r"{sensorData.confidence != null ? `Confidence: ${Math.round(sensorData.confidence * 100)}%` : 'Confidence: --'}"
        r"{sensorData.motion ? ` · hand ${sensorData.motion.state}` : ''}"
        r"{sensorData.decisionMs != null ? ` · decided in ${sensorData.decisionMs} ms` : ''}"
        r"{sensorData.telemetry ? ` · ${sensorData.telemetry.rate_hz} Hz · gaps ${sensorData.telemetry.seq_gaps}` : ''}</div>")

CHANGES = [
    # (what, old, new)
    ("tailwind local", '<script src="https://cdn.tailwindcss.com"></script>', '<script src="/vendor/tailwind.js"></script>'),
    ("react local", '<script crossorigin src="https://unpkg.com/react@18/umd/react.development.js"></script>',
     '<script src="/vendor/react.production.min.js"></script>'),
    ("react-dom local", '<script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>',
     '<script src="/vendor/react-dom.production.min.js"></script>'),
    ("babel local", '<script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>',
     '<script src="/vendor/babel.min.js"></script>'),
    ("framer-motion local", '<script src="https://unpkg.com/framer-motion@10.16.4/dist/framer-motion.js"></script>',
     '<script src="/vendor/framer-motion.js"></script>'),
    ("lucide local", '<script src="https://unpkg.com/lucide@latest"></script>', '<script src="/vendor/lucide.js"></script>'),
    ("letters state", "const [dataHistory, setDataHistory] = useState([]);",
     "const [dataHistory, setDataHistory] = useState([]);\n            const [letters, setLetters] = useState([]);"),
    ("live fields", "setSensorData(prev => ({ ...prev, ...msg.data.raw_data, prediction: msg.data.prediction }));",
     "setSensorData(prev => ({ ...prev, ...msg.data.raw_data, prediction: msg.data.prediction, confidence: msg.data.confidence, motion: msg.data.motion, decisionMs: msg.data.decision_ms, telemetry: msg.data.telemetry }));"),
    ("letter message", '} else if (msg.type === "train_log") {',
     '} else if (msg.type === "letter") {\n                                setLetters(prev => [...prev.slice(-79), msg.data.label]);\n                            } else if (msg.type === "train_log") {'),
    ("real confidence + letters strip", "Confidence: 98.4%</div>", CONF + "\n                                        " + STRIP),
    ("config: model", '<span>Epochs</span><span className="font-mono font-bold">{trainingReport.training_config.epochs}</span>',
     '<span>Model</span><span className="font-mono font-bold">{trainingReport.training_config.model}</span>'),
    ("config: features", '<span>Batch Size</span><span className="font-mono font-bold">{trainingReport.training_config.batch_size}</span>',
     '<span>Features</span><span className="font-mono font-bold">{trainingReport.training_config.num_features}</span>'),
    ("config: smoothing", '<span>Window Size</span><span className="font-mono font-bold">{trainingReport.training_config.window_size}</span>',
     '<span>Smoothing (ms)</span><span className="font-mono font-bold">{trainingReport.training_config.smooth_ms}</span>'),
]

done = skipped = 0
for what, old, new in CHANGES:
    n = text.count(old)
    if new in text:
        print(f"already  {what}")
    elif n == 1:
        text = text.replace(old, new)
        print(f"applied  {what}")
        done += 1
    else:
        print(f"SKIPPED  {what}  (found {n} matches; expected 1)")
        skipped += 1

path.write_text(text, encoding="utf-8")
print(f"\n{done} applied, {skipped} skipped. Backup: {bak.name}")
