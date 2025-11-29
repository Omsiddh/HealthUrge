-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    role TEXT CHECK (role IN ('admin', 'hospital', 'patient')) NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create hospitals table
CREATE TABLE IF NOT EXISTS hospitals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hospital_id TEXT UNIQUE NOT NULL, -- External ID if needed
    name TEXT NOT NULL,
    lat DOUBLE PRECISION NOT NULL,
    lon DOUBLE PRECISION NOT NULL,
    capacity INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create hospital_daily_stats table
CREATE TABLE IF NOT EXISTS hospital_daily_stats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hospital_id UUID REFERENCES hospitals(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    inflow INTEGER,
    available_capacity INTEGER,
    staff INTEGER,
    outbreak_level TEXT,
    aqi_manual INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(hospital_id, date)
);

-- Create forecasts_inflow table
CREATE TABLE IF NOT EXISTS forecasts_inflow (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hospital_id UUID REFERENCES hospitals(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    predicted_inflow INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(hospital_id, date)
);

-- Create forecasts_capacity table
CREATE TABLE IF NOT EXISTS forecasts_capacity (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hospital_id UUID REFERENCES hospitals(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    predicted_capacity INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(hospital_id, date)
);

-- Create allocations table
CREATE TABLE IF NOT EXISTS allocations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plan_json JSONB NOT NULL,
    critique_json JSONB,
    date DATE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create alerts table
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message TEXT NOT NULL,
    hospital_id UUID REFERENCES hospitals(id) ON DELETE CASCADE,
    type TEXT NOT NULL,
    date DATE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable Row Level Security (RLS)
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE hospitals ENABLE ROW LEVEL SECURITY;
ALTER TABLE hospital_daily_stats ENABLE ROW LEVEL SECURITY;
ALTER TABLE forecasts_inflow ENABLE ROW LEVEL SECURITY;
ALTER TABLE forecasts_capacity ENABLE ROW LEVEL SECURITY;
ALTER TABLE allocations ENABLE ROW LEVEL SECURITY;
ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;

-- Create policies (Example: Allow read access to everyone for now, restrict write)
-- In production, you would have stricter policies based on user roles.
CREATE POLICY "Allow public read access" ON hospitals FOR SELECT USING (true);
