# Predictive AI System for Healthcare Surge Management — `context.md`

> Purpose: a single, **hackathon-ready** reference document you can hand to any AI or teammate. It describes the scope we will build in 24 hours, how to extend afterward, datasets, training scripts, API contracts, debugging tips, roles & auth, self-critique loop, fallback behaviors (AQI settable by hospitals), resource-allocation logic, push alerts, UI components and Figma guidance, and deployment notes (Supabase + Next.js + Python ML).
> File name suggestion: `context.md`

---

## Table of contents

1. Project summary (MVP)
2. High-level goals & constraints (24-hour hack)
3. Data: available & synthetic (location of generated CSV)
4. Database / Auth (Supabase) — schema + roles
5. Models: chosen approach (Prophet) + optional residual LSTM
6. Training & inference workflows (step-by-step commands & code)
7. API contracts (REST) — request/response examples
8. Resource allocation — minimal algorithm for hack + future GNN plan
9. Agentic orchestration & Self-Critique Loop (how LLM fits)
10. Push notification design & triggers
11. Frontend (Next.js) structure, pages, components, and UI theme
12. Figma layout guide (full layout directions)
13. Dev environment, repo layout, CI, and deployment
14. Evaluation metrics, test cases & debugging tips for AI-driven dev
15. Roadmap post-hack (GNN, real-time streaming, scaling)
16. Appendix: useful code snippets, SQL table DDL, mermaid architecture diagram

---

# 1. Project summary (MVP)

**Title:** Predictive AI System for Healthcare Surge Management and Resource Optimization — Hackathon MVP

**MVP scope (must-have for 24-hour hack):**

* Single-copy dataset (synthetic) for **8 Mumbai hospitals** — CSV at `/mnt/data/mumbai_hospitals_synthetic_2023_2024.csv`.
* Backend ML service (Python) that trains **Prophet** per hospital to forecast daily patient inflow for next 7 days. (Prophet chosen by team.)
* Inference endpoint to return forecasts for a hospital or city-level aggregate.
* Simple **resource allocation engine** (greedy or min-cost flow) that recommends patient routing or staff redistribution for shortfall/overflow.
* Rule-based **patient advisory** (for patients) with simple triage rules + LLM-assisted natural language explanation (Gemini/GPT optional).
* Web dashboard (Next.js) where admins see hospital cards, forecasts, and allocation suggestions.
* Push alerts: proactive notifications (web/push) when predicted surge > threshold.
* Supabase for Postgres + auth (roles: `admin`, `hospital_user`, `patient`).
* Fallback AQI input: if no AQI API, each hospital can update AQI via their dashboard.

**Stretch but optional (only if time allows):**

* Residual LSTM to improve Prophet forecasts.
* Minimal LLM agent that converts admin prompts into API calls and explains reasoning (Gemini/GPT).
* Dashboard map visualization and small animations.

**Deliverable constraints:** deliver a functioning demo using the synthetic data, clear documentation, and a UI that shows predictions + allocation + one notification flow.

---

# 2. High-level goals & constraints (24-hour hack)

* **Keep models light**: Prophet models per hospital can be trained quickly and give credible results. Training 8 Prophet models on 2 years of daily data takes minutes on a laptop.
* **Avoid heavy ML infra**: no distributed training, no large GNN training in hack. Implement simple deterministic allocation with a clear "future work: GNN" note.
* **Agentic AI angle**: build an LLM-based orchestrator (Gemini or GPT) that *calls* your Python endpoints, critiques its allocation (self-critique loop), and produces human-readable plans/alerts.
* **Transparent synthetic data**: be explicit the dataset is synthetic and how it was generated in the README.

---

# 3. Data — available & synthetic

**Generated dataset path (from earlier step):**

```
/mnt/data/mumbai_hospitals_synthetic_2023_2024.csv
```

**CSV columns:**

* `date` (YYYY-MM-DD)
* `hospital_id` (int)
* `hospital_name`
* `latitude`, `longitude`
* `AQI` (int)
* `is_festival` (0/1)
* `outbreak_level` (0/1/2)
* `inflow` (int) — daily patient arrivals
* `beds_capacity` (int)
* `doctors_on_duty` (int)

**How synthetic values were generated (short):**

* Baseline inflow per hospital with weekday effects, monsoon/festival/outbreak multipliers, an AQI-dependent additive effect, and Gaussian noise.
* Festivals and outbreak events are sparse and increase inflow.
* Usage: train Prophet models per hospital using `date` and `inflow` with external regressors `AQI`, `is_festival`, `outbreak_level`.

**If you obtain real hospital data later:**

* Keep same schema (date-hospital granularity).
* Ensure PII is stripped, only aggregated counts used for forecasting.

---

# 4. Database / Auth (Supabase)

**Why Supabase:** fast setup (Postgres), built-in auth, row-level security, file storage, and realtime features for push.

**Roles:**

* `admin` — full access, system-wide config and alerts.
* `hospital_user` — update hospital-level inputs (AQI override, current capacity, doctors on duty), view forecasts for own hospital.
* `patient` — limited advisory; no direct access to hospital internals.

**Suggested Postgres schema (Supabase):**

```sql
-- hospitals
CREATE TABLE hospitals (
  id integer PRIMARY KEY,
  name text NOT NULL,
  latitude double precision,
  longitude double precision,
  beds_capacity integer,
  created_at timestamptz DEFAULT now()
);

-- hospital_daily (observations & manual overrides)
CREATE TABLE hospital_daily (
  id serial PRIMARY KEY,
  hospital_id integer REFERENCES hospitals(id),
  date date NOT NULL,
  inflow integer,               -- observed (if available)
  aqi integer,
  is_festival boolean DEFAULT false,
  outbreak_level integer DEFAULT 0,
  doctors_on_duty integer,
  aqi_override boolean DEFAULT false,
  created_at timestamptz DEFAULT now()
);

-- forecasts
CREATE TABLE forecasts (
  id serial PRIMARY KEY,
  hospital_id integer REFERENCES hospitals(id),
  forecast_date date NOT NULL,  -- date being forecasted
  created_at timestamptz DEFAULT now(),
  predicted_inflow integer,
  lower_bound integer,
  upper_bound integer,
  source text -- e.g., prophet_v1
);

-- allocations
CREATE TABLE allocations (
  id serial PRIMARY KEY,
  created_at timestamptz DEFAULT now(),
  source_hospital integer REFERENCES hospitals(id),
  target_hospital integer REFERENCES hospitals(id),
  patients_to_move integer,
  reason text,
  agent_version text
);

-- users handled by supabase auth; map role to metadata claim `app_role`
```

**Row-level security (RLS):**

* Hospital users can only write/read rows where `hospital_id = their_hospital_id` (store mapping in auth metadata).
* Admin role can read/write everything.

**Realtime notifications:**

* Use Supabase Realtime or Edge Functions to push webhooks when `forecasts` or `allocations` are inserted that exceed thresholds.

---

# 5. Models — chosen approach

**Primary model (MVP): Prophet (by Facebook/Meta)**

* Train one Prophet model per hospital on daily `inflow`.
* Add external regressors: `AQI`, `is_festival`, `outbreak_level`.
* Forecast horizon: 7 days (daily granularity).

**Optional improvement (if time remains):**

* Fit an LSTM on **residuals** produced by Prophet to capture short non-linear patterns.
* Or combine Prophet seasonal/trend + small MLP on exogenous features (faster than LSTM).

**Why Prophet?**

* Fast to train, robust to missing dates, supports external regressors, interpretable components (trend, weekly/seasonal), suitable for hackday demo.

---

# 6. Training & inference workflows (concrete)

### Dev environment

* Python 3.10+; create virtualenv: `python -m venv .venv && source .venv/bin/activate`
* Requirements (minimal):

  ```
  pip install prophet pandas numpy fastapi uvicorn pydantic scikit-learn sqlalchemy psycopg2-binary supabase
  ```

  (Note: package name `prophet` or `cmdstanpy` variant depending on system; for quick hack, `prophet` pip is OK.)

### Training script (prophet) — `train_prophet.py` (simplified):

```python
# train_prophet.py
import pandas as pd
from prophet import Prophet
import joblib

df = pd.read_csv("mumbai_hospitals_synthetic_2023_2024.csv", parse_dates=["date"])
HOSP_ID = 1  # change per hospital or loop
hdf = df[df.hospital_id == HOSP_ID].sort_values("date")

# Prepare prophet dataframe
m_df = hdf[["date", "inflow"]].rename(columns={"date": "ds", "inflow": "y"})
# add regressors
m_df["AQI"] = hdf["AQI"].values
m_df["is_festival"] = hdf["is_festival"].values
m_df["outbreak_level"] = hdf["outbreak_level"].values

m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
m.add_regressor("AQI")
m.add_regressor("is_festival")
m.add_regressor("outbreak_level")

m.fit(m_df)

# Forecast
future = m.make_future_dataframe(periods=7)
# fill future regressors using last-known values or hospital_daily overrides
# naive: copy last row values
last = m_df.iloc[-1]
future["AQI"] = last["AQI"]
future["is_festival"] = 0
future["outbreak_level"] = 0

fcst = m.predict(future)
# Save model & predictions
joblib.dump(m, f"models/prophet_hosp_{HOSP_ID}.pkl")
fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(f"outputs/forecast_hosp_{HOSP_ID}.csv", index=False)
```

### Inference endpoint (FastAPI)

* Endpoint: `GET /api/forecast?hospital_id=1`
* Implementation loads `models/prophet_hosp_1.pkl` (or refit on-demand) and returns next 7-day `yhat`.

### Training time estimate (realistic)

* On a modern dev laptop: ~10–60 seconds per Prophet model on 2 years daily data. 8 models ≈ 2–10 minutes. Reserve additional time for debugging/data prep — safe to say **train all 8 in under 30 minutes** in most cases.

---

# 7. API contracts (REST) — examples

### Auth

* Supabase handles auth; include `Authorization: Bearer <supabase_jwt>` in headers.

### Forecast

**GET** `/api/forecast?hospital_id=1&horizon=7`
Response:

```json
{
  "hospital_id": 1,
  "forecasts": [
    {"date":"2025-11-25","predicted_inflow":123,"lower":101,"upper":145},
    ...
  ],
  "model":"prophet_v1",
  "generated_at":"2025-11-25T01:05:00Z"
}
```

### Allocation suggestion

**POST** `/api/allocate`
Request:

```json
{
  "source_hospital_id": 1,
  "forecast_window_days": 3,
  "threshold_percent": 0.9
}
```

Response:

```json
{
  "allocations": [
    {"source":1, "target":2, "patients_to_move":10, "reason":"predicted overflow"}
  ],
  "explanation":"Redistribution to nearest hospitals with spare capacity."
}
```

### Patient advisory (rule-based)

**POST** `/api/advice`
Request:

```json
{
  "symptoms_severity":"medium",
  "age_group":"adult",
  "location":{"lat":19.04,"lon":72.85},
  "preferred_time":"now"  // or "later"
}
```

Response:

```json
{
  "advice":"Go to Lilavati Hospital now (nearest, predicted load 70%). If symptoms worsen, call ambulance.",
  "confidence":"high",
  "explanation":"Severity medium + nearest facility load acceptable"
}
```

---

# 8. Resource allocation — use & hack-friendly algorithm

**Purpose of resource allocation:**

* Ensure patients receive timely care by distributing demand across facilities.
* Reduce overcrowding at any single ED.
* Suggest staff redistribution and patient routing to maintain quality.
* Provide administrators a data-driven plan for short-term actions (transfer patients, call in staff, open surge beds, redirect ambulances).

**What we implement in the hack: Minimal greedy allocation**

1. Inputs:

   * `forecasted_inflow` for next `k` days for each hospital.
   * `current_beds_capacity`, `doctors_on_duty`, `current_load` (if available).
   * `travel_time_matrix` or haversine distances to estimate nearby hospitals.
2. For each hospital `H` with `predicted_load > capacity * threshold`:

   * Compute overflow = `predicted_load - capacity`.
   * Find candidate hospitals `C` with `spare_capacity = capacity - predicted_load`.
   * Rank candidates by `travel_time` and `spare_capacity`.
   * Assign `min(overflow, spare_capacity)` patients to nearest candidates until overflow resolved.
3. Produce `allocations` table rows and an `explain` text.

**Algorithm complexity:** O(N * M log M) where N = hospitals overloaded, M = candidate hospitals — trivial for 8 nodes.

**Why not GNN in hack:** training a GNN to *learn* allocation policies needs simulated scenarios and training time. For hack, implement the greedy algorithm and mark GNN as future improvement. Provide a stubbed module `gnn_allocator.py` that can be filled later.

**GNN future plan (post-hack):**

* Simulate many surge scenarios (Monte Carlo).
* Train a GNN (PyTorch Geometric) to output allocation decisions minimizing city-cost (transfer risk + patient wait).
* Reward function: minimize number of severely overloaded hospitals + total travel + number of redirected critical patients.

---

# 9. Agentic orchestration & Self-Critique Loop

**Agent role (Gemini/GPT):** The agent acts as an orchestrator that:

* Receives user/admin natural-language queries.
* Calls forecast & allocation APIs (tool-calls).
* Produces human-readable recommendations.
* Runs a **Self-Critique Loop**: before finalizing recommendations, the agent generates an internal critique based on extra constraints (capacity percentages, traffic conditions if available, recent override flags) and proposes a revised plan. The final output contains both the plan and the critique summary.

**Implementation pattern (safe & hackable):**

1. Agent prompt template includes:

   * The forecasts (structured JSON)
   * Current hospital capacities and overrides
   * Traffic & AQI (if available)
   * A list of allowed actions (move X patients, call in N staff, open surge beds)
2. Agent proposes `plan_v1`.
3. Agent runs `self_critique()` — queries the same state with constraints and asks:

   * “Is any target hospital >80% after move?”
   * “Is travel time > threshold (e.g., 60 mins)?”
   * “Does hospital have `aqi_override` flagged?”
4. If critique finds problems, agent adjusts `plan_v1` → `plan_v2`.
5. Agent returns both:

   * `final_plan` (human readable)
   * `critique_summary` (short)
   * `confidence_score` (low/medium/high)

**Simple local implementation (hack):**

* The orchestration backend contains a small function:

```python
def self_critique_and_revise(plan, state):
    issues = []
    for alloc in plan:
        target_load_after = state[target].predicted + alloc.patients_to_move
        if target_load_after / state[target].capacity > 0.85:
            issues.append(("target_overloaded", alloc))
    # Consider traffic flag or hospital overrides
    if issues:
        revise_plan(...)  # reduce moves or find alternative
    return revised_plan, issues
```

* Optionally wrap with LLM: send `plan` and `state` to LLM to produce textual critique + revised plan decisions. Agent should *call* your `self_critique_and_revise` service or use LLM to *propose* critiques and then validate them.

**Example (from your requested scenario):**

* Plan: Move 20 patients from KEM → Sion.
* Critique: “Sion at 80%, heavy traffic, risk of failure.”
* Revised: Move only 10 to Nair (which has spare capacity and is reachable).

**Why this hits the Agentic theme:** the agent makes decisions, introspects, and corrects/justifies them.

---

# 10. Push notification design & triggers

**Push triggers (examples):**

* Forecasted % increase > `X%` (e.g., 15%) for a hospital in next 48 hours.
* Forecasted load > capacity * threshold (e.g., 0.9).
* AQI > 200 and predicted respiratory case bump > 10% city-wide.
* Manual override by hospital: marks immediate 'alert'.

**Push channels:**

* In-app notifications (Next.js + Supabase Realtime)
* Browser push (via Web Push API / Supabase Edge Functions)
* Optional: Email or SMS via Twilio (if time allows)

**Payload example:**

```json
{
  "type":"surge_alert",
  "title":"Predicted surge: KEM Hospital (48h)",
  "message":"AQI 300 predicted in Chembur; expected respiratory case spike 15% in 48h. Click for suggested resource plan.",
  "links":{"open_allocation":"/admin/allocations/1"}
}
```

**Push UI behavior (MVP):**

* Bell icon shows unread alerts.
* Click opens details modal with forecast graphs and proposed allocation + self-critique summary.

---

# 11. Frontend (Next.js) — pages, components, and theme

**Stack**:

* Next.js (App Router or Pages router — choose what you’re familiar with)
* Tailwind CSS for quick styling (you already use it)
* Charts: `recharts` or `chart.js` (`react-chartjs-2`)
* Map: optional `react-leaflet` for small map view

**Pages / routes:**

* `/` — Landing + summary (city-level)
* `/login` — Supabase auth (email links)
* `/dashboard` (admin) — cards for hospitals, quick alerts, aggregated forecast
* `/hospital/[id]` — hospital detail: observed inflow, forecast graph, capacity, AQI override input (editable by hospital_user), allocation actions
* `/allocations` — suggested allocations + approve/reject buttons
* `/advice` — patient advisory UI (form: symptoms, location) → result
* `/settings` — thresholds, push notification prefs, festival calendar

**Components:**

* `HospitalCard` — summary metrics (predicted load %, traffic/warning badge)
* `ForecastChart` — historical + forecast lines with uncertainty ribbon
* `AllocationPanel` — list of suggested patient/staff moves with approve buttons
* `NotificationBell` — shows push alerts
* `AQIOverride` — input + timestamp + reason

**UI Theme suggestion (simple & professional):**

* Palette: Deep teal/navy primary (#0f4c81), soft cyan accents, warm orange for alerts (#ff7a59), neutral greys for background.
* Typography: Inter or Poppins (clean, modern).
* Visual style: cards with subtle shadows, 2xl radius, compact spacing (p-3 to p-4), small badges for statuses.
* Charts: light background; forecast band uses translucent fill.
* Accessibility: color-contrast checks; badges with icons.

---

# 12. Figma layout guide (full layout directions)

*(Provide to designer/AI) — create frames sized 1440×1024; use a 12-column grid.*

**Pages to create in Figma (each as frames):**

1. **Login** — email input + SSO button + small project description.
2. **Landing / Overview** — top header with nav; big KPI row (city predicted surge %, hospitals overloaded count), central map (small) or grid of `HospitalCard`s.
3. **Dashboard (Admin)** — left sidebar with nav, main canvas:

   * Top: Notification strip (pushes)
   * Left column: Hospital list (vertical) with small sparkline
   * Right column: Large `ForecastChart` + `AllocationPanel` for selected hospital
4. **Hospital Detail** — forecast chart (7-day) with table for predictions, capacity controls editable, AQI override panel (input + save), "Request Allocation" button.
5. **Allocation Approval Modal** — shows suggested moves, critique box, Approve/Reject buttons, reason textfield.
6. **Patient Advisory** — form (symptom severity, age group, location), output card with advice and nearby facility suggestion.
7. **Settings** — thresholds, event calendar editor (festival multipliers).

**Components & styles:**

* `HospitalCard` component with hospital name, small map pin, predicted load %, capacity ring (donut), small status badge.
* `ForecastChart` with 3 lines: historical, predicted, recommended capacity line; shaded confidence interval.
* `AlertToast` component for push messages (auto-dismiss and link).
* `CritiqueBox` — small red/orange border for agent self-critique statements.

**Export instructions:**

* Export assets as SVG where possible.
* Provide color tokens and text styles in a small style guide page inside Figma.

---

# 13. Dev environment, repo layout & deployment

**Repo skeleton:**

```
/project-root
  /frontend (Next.js + Tailwind)
  /backend
    /app (FastAPI)
    /models (trained Prophet models)
    /scripts (train_prophet.py, generate_data.py)
  /infra
    supabase-config.md
  README.md
  context.md   <-- this file
```

**Local run**

* Frontend: `cd frontend && npm install && npm run dev`
* Backend: `cd backend && pip install -r requirements.txt && uvicorn app.main:app --reload`
* Supabase: use `supabase start` or use cloud project with env vars.

**Deployment**

* Frontend: Vercel (Next.js) or Netlify
* Backend: Vercel Serverless Functions or Render / Railway / Supabase Edge Functions (if Python support). Alternatively, deploy FastAPI on Render/Heroku.
* Supabase: hosted project for production.

**CI**

* GitHub Actions: run tests and `lint` on push; build frontend preview on PR.

---

# 14. Evaluation metrics & debugging tips (for AI + humans)

**Forecasting metrics:**

* MAE, RMSE, MAPE for validation period.
* Coverage of predicted intervals: check % of actuals within `yhat_lower`–`yhat_upper`.

**Allocation metrics:**

* Number of hospitals overloaded after allocation (should decrease).
* Average % capacity used across hospitals.
* Total patient transfer distance (minimize).

**Agent critique checks:**

* Ensure the agent does not recommend moving patients into hospitals flagged `aqi_override` or `on_strike`.
* Rate-limit agent actions (no more than X moves without admin approval).

**Debugging tips:**

* If Prophet predictions are flat: verify external regressors passed to `future` dataframe.
* If forecasts error: check date parsing (ensure `ds` is datetime) and no duplicate dates.
* If allocation picks same hospital as target repeatedly: validate spare_capacity calculation uses predicted load **after** other planned moves; process in descending overflow order.
* If push notifications not sent: inspect supabase Realtime logs and edge function logs.

---

# 15. Roadmap post-hack (extensions)

* Replace greedy allocator with **GNN** trained on simulated scenarios to learn optimal flows and staff allocation.
* Real-time streaming of ED arrivals via Kafka or Supabase Realtime for hourly forecasts.
* Mobile apps and SMS integration for patient advisories.
* Add ambulance dispatch optimization.
* Integrate official AQI APIs when available (CPCB / SAFAR). Add traffic APIs (Google Maps / Here) for travel-time-aware allocation.
* Add RL-based policy learning for worst-case surges.

---

# 16. Appendix

## A. Minimal mermaid architecture diagram

```mermaid
flowchart LR
  subgraph Frontend
    A[Next.js App]
  end

  subgraph Backend
    B[FastAPI Forecast Service]
    C[Allocation Engine (greedy)]
    D[Agent Orchestrator (LLM)]
  end

  subgraph DB
    S[(Supabase Postgres)]
  end

  subgraph MLModels
    M1[Prophet Models (per hospital)]
  end

  A -->|API| B
  A -->|Auth| S
  B -->|reads/writes| S
  C -->|writes allocations| S
  B --> M1
  D -->|calls| B
  D -->|reads state| S
  D -->|notifies| A
  A -->|push| Users[Users]
```

## B. Sample Prophet training command (bash)

```bash
python backend/scripts/train_all_prophet.py \
  --data /mnt/data/mumbai_hospitals_synthetic_2023_2024.csv \
  --out_dir backend/models \
  --hospitals 1 2 3 4 5 6 7 8
```

## C. Example SQL DDL (short)

(See section 4 above — copy/paste into Supabase SQL editor.)

## D. Notes about AQI fallback & hospital override

* On each hospital page, provide an `AQIOverride` control:

  * input value, reason, timestamp. Set `aqi_override = true`.
* Propagate `aqi_override` into forecasts: if override exists for recent date, use it; for future, use last-known override value unless a real API is present.
* Validate overrides via admin: require quick confirmation (checkbox) to avoid accidental bad data.

## E. Authentication mapping

* On signup, hospital users must be associated with a `hospital_id`. Store this in Supabase user metadata (`app_role = hospital_user`, `hospital_id = X`).
* Store admin privileges in metadata `app_role = admin`.

## F. Self-Critique implementation pseudo-flow (detailed)

1. Agent receives `admin: "What should we do for KEM next 48h?"`
2. Agent calls `GET /api/forecast?hospital_id=1&horizon=2` → gets predicted loads.
3. Agent calls `GET /api/nearby?hospital_id=1` → gets neighbors & travel times.
4. Agent proposes allocation: move 20 patients to Sion.
5. Agent runs `self_critique_and_revise(plan)`:

   * fetch Sion current load & predicted
   * check travel_time(Sion) vs acceptable threshold
   * check if Sion `aqi_override` or `on_maintenance` flag set
6. If critique issues exist, agent finds next best candidate (e.g., Nair) and modifies plan.
7. Agent writes `allocations` to DB as `pending` and returns message:

   * Plan summary
   * Critique explanation
   * Approve button for admin (if interactive) or auto-execute if `auto_approve` flagged.
