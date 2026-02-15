# AI-Powered Smart Patient Triage (Frontend)

Production-ready hackathon demo: **40% faster triage · 95% accuracy · Explainable AI**.

- **This folder is frontend only.** Backend and models live in the **SOL** folder (sibling folder in the workspace). Do not put a SOL folder inside this repo.
- **Stack:** Vite + React 18 + TypeScript + Tailwind CSS + React Router + Framer Motion
- **Run:** See **HACKATHON_RUN.md** for full flow (start SOL backend first, then `npm run dev` here).

---

## Jury demo script (5–7 min)

1. **Landing (30 s)**  
   - Open app. Point out: hero title, gradient, particle background, dual CTAs.  
   - Click **Quick Demo Patient #1** → jumps to Triage with chest-pain patient pre-filled.

2. **Triage wizard (1.5 min)**  
   - **Step 1:** Age slider, gender, symptom search (type "chest"), Voice button (demo/real), vitals (BP/HR/Temp), drag-drop zone (mock OCR).  
   - **Step 2:** Conditions, allergies, medications.  
   - **Step 3:** Review summary → **ANALYZE WITH AI** (2 s spinner) → Results.

3. **Results (1 min)**  
   - Confetti entry, **HIGH RISK** badge, donut (confidence), department card.  
   - Explainable AI: bar chart (top factors), decision path, innovation badges.  
   - **PDF Report** (download), **Add to Queue** → Dashboard.

4. **Live dashboard (1.5 min)**  
   - Kanban: Low | Medium | High. Cards auto-populate every 5 s.  
   - Drag card between columns, Queue on left, Add Random, Speed slider.  
   - Department pie chart, KPI cards.

5. **About (30 s)**  
   - Mermaid architecture, 12 feature cards, metrics table (Accuracy 95.2%, F1 0.92, &lt;500 ms, &lt;2% bias).

6. **Polish**  
   - Dark/light toggle, EN/HI/TA selector, keyboard nav, PWA install prompt (if supported).

---

## Routes

| Path       | Page              |
|-----------|-------------------|
| `/`       | Hero landing      |
| `/triage` | 3-step triage     |
| `/results`| AI results        |
| `/dashboard` | Live Kanban   |
| `/about`  | Tech & metrics    |

---

## Quick start

```bash
npm install
npm run dev
```

Open **http://localhost:5173**.

### With backend (SOL)

**Two folders in workspace:** **AI traige** (this frontend) and **SOL** (backend). See **HACKATHON_RUN.md** for step-by-step run instructions.

1. Start **SOL** first (the backend folder): `cd SOL && pip3 install -r requirements.txt && python3 app.py` → runs at **http://127.0.0.1:8000**.
2. Start this frontend: `npm run dev` → open http://localhost:5173.

The app calls `POST http://127.0.0.1:8000/triage/predict`. Override with `VITE_API_URL` in a `.env` file if needed. If SOL is not running, the app falls back to demo data and shows a toast.

---

## Project structure

```
src/
  data/dummyData.ts   # 50+ synthetic patients, presets, symptoms, conditions
  context/DemoContext.tsx  # useDemo(), loadPatient, queue, live mode
  components/         # Layout, Nav, Footer
  pages/              # Hero, Triage, Results, Dashboard, About
  types/index.ts      # DemoPatient, AIOutput, Vitals, etc.
```

---

## Evaluation checklist

- **UI/UX:** Medical theme (blue → teal), risk colors (green/yellow/red), mobile-first, WCAG focus rings, ARIA.
- **Routes:** 5 routes with Framer Motion transitions.
- **Dummy data:** 50+ patients, all risk levels and departments, presets.
- **Demo flow:** Quick presets, auto-demo loop (30 s), live dashboard simulation.
- **Triage:** 3 steps, Fuse.js symptoms, voice (Web Speech API), vitals, drag-drop EHR mock, localStorage draft.
- **Results:** Confetti, risk badge, Chart.js donut/bar, explainability (factors, decision path), PDF export, Add to Queue.
- **Dashboard:** Kanban columns, auto-add every 5 s, drag between columns, queue, KPIs, department pie.
- **About:** Mermaid diagram, feature cards, metrics table.
- **Polish:** PWA manifest, dark mode, language selector, README with jury script.
