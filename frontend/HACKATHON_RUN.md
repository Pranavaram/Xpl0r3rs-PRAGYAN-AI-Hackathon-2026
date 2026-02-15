# Hackathon — How to Run (Frontend + Backend)

Your workspace has **two separate folders**:

| Folder       | What it is              | Port |
|-------------|--------------------------|------|
| **AI traige** | Frontend only (this repo) | 5173 |
| **SOL**       | Backend + ML models       | 8000 |

**AI traige must NOT contain a SOL folder.** All backend code lives in the **SOL** folder at the workspace root.

---

## 1. Start the backend (SOL) first

Open a terminal:

```bash
cd SOL
pip3 install -r requirements.txt
python3 app.py
```

Wait until you see the server running (e.g. `Uvicorn running on http://0.0.0.0:8000`).

- Backend API: **http://127.0.0.1:8000**
- Docs: http://127.0.0.1:8000/docs

---

## 2. Start the frontend (AI traige)

Open a **second** terminal:

```bash
cd "AI traige"
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## 3. Test the flow

1. Go to **Triage** (or use a Quick Demo patient from the home page).
2. Fill the form and click **Analyze with AI**.
3. The frontend calls `POST http://127.0.0.1:8000/triage/predict` and shows the result on the Results page.

If the SOL backend is not running, you’ll see a toast “Backend unavailable — using demo result” and the app will still show a demo result so you can present.

---

## Optional: different backend URL

If SOL runs on another host/port, create a `.env` file in **AI traige**:

```
VITE_API_URL=http://localhost:8000
```

(Use your actual SOL URL.)

---

## Quick checklist

- [ ] SOL folder is **at workspace root**, not inside AI traige
- [ ] Backend (SOL) started first on port 8000
- [ ] Frontend (AI traige) started on port 5173
- [ ] Browser at http://localhost:5173
