-- DANGER: This script drops existing tables to ensure schema compatibility.
-- We use INTEGER for hospital_id to match the Python backend's mock data (IDs 1-8).
-- If you have important data, back it up first.

-- Drop tables with CASCADE to handle foreign key dependencies
DROP TABLE IF EXISTS allocations CASCADE;
DROP TABLE IF EXISTS forecasts CASCADE;
DROP TABLE IF EXISTS hospital_daily CASCADE;
DROP TABLE IF EXISTS hospitals CASCADE;

-- Also drop old/conflicting tables from previous schema versions if they exist
DROP TABLE IF EXISTS hospital_daily_stats CASCADE;
DROP TABLE IF EXISTS forecasts_inflow CASCADE;
DROP TABLE IF EXISTS forecasts_capacity CASCADE;
DROP TABLE IF EXISTS alerts CASCADE;

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

-- Seed initial hospitals (matching Python mock data)
INSERT INTO hospitals (id, name, latitude, longitude, beds_capacity) VALUES
(1, 'KEM Hospital', 19.002, 72.842, 500),
(2, 'Sion Hospital', 19.045, 72.865, 600),
(3, 'Nair Hospital', 18.975, 72.825, 400),
(4, 'Cooper Hospital', 19.101, 72.837, 300),
(5, 'Lilavati Hospital', 19.051, 72.829, 350),
(6, 'Hinduja Hospital', 19.033, 72.838, 320),
(7, 'Breach Candy', 18.971, 72.809, 200),
(8, 'Saifee Hospital', 18.953, 72.821, 250);
