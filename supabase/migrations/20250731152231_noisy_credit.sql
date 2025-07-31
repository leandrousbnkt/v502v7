/*
  # Create analyses table for ARQV30 Enhanced v2.0

  1. New Tables
    - `analyses`
      - `id` (bigint, primary key)
      - `nicho` (text, market segment)
      - `produto` (text, product/service)
      - `descricao` (text, description)
      - `preco` (numeric, price)
      - `publico` (text, target audience)
      - `concorrentes` (text, competitors)
      - `dados_adicionais` (text, additional data)
      - `objetivo_receita` (numeric, revenue goal)
      - `orcamento_marketing` (numeric, marketing budget)
      - `prazo_lancamento` (text, launch timeline)
      - `avatar_data` (jsonb, avatar analysis)
      - `positioning_data` (jsonb, positioning strategy)
      - `competition_data` (jsonb, competition analysis)
      - `marketing_data` (jsonb, marketing strategy)
      - `metrics_data` (jsonb, performance metrics)
      - `funnel_data` (jsonb, sales funnel)
      - `market_intelligence` (jsonb, market research)
      - `action_plan` (jsonb, action plan)
      - `comprehensive_analysis` (jsonb, full analysis)
      - `status` (text, analysis status)
      - `created_at` (timestamptz)
      - `updated_at` (timestamptz)

  2. Security
    - Enable RLS on `analyses` table
    - Add policy for public read access (since no auth system)
*/

CREATE TABLE IF NOT EXISTS analyses (
  id bigserial PRIMARY KEY,
  nicho text NOT NULL,
  produto text,
  descricao text,
  preco numeric(10,2),
  publico text,
  concorrentes text,
  dados_adicionais text,
  objetivo_receita numeric(15,2),
  orcamento_marketing numeric(15,2),
  prazo_lancamento text,
  avatar_data jsonb,
  positioning_data jsonb,
  competition_data jsonb,
  marketing_data jsonb,
  metrics_data jsonb,
  funnel_data jsonb,
  market_intelligence jsonb,
  action_plan jsonb,
  comprehensive_analysis jsonb,
  status text DEFAULT 'pending',
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_analyses_nicho ON analyses(nicho);
CREATE INDEX IF NOT EXISTS idx_analyses_status ON analyses(status);
CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at DESC);

-- Create sessions table for tracking user sessions
CREATE TABLE IF NOT EXISTS sessions (
  id bigserial PRIMARY KEY,
  session_id text UNIQUE NOT NULL,
  user_agent text,
  ip_address inet,
  metadata jsonb,
  created_at timestamptz DEFAULT now(),
  last_activity timestamptz DEFAULT now()
);

-- Create indexes for sessions
CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at DESC);

-- Create attachments table for file uploads
CREATE TABLE IF NOT EXISTS attachments (
  id bigserial PRIMARY KEY,
  session_id text NOT NULL,
  filename text NOT NULL,
  original_filename text NOT NULL,
  file_size integer,
  mime_type text,
  content_type text,
  extracted_content text,
  metadata jsonb,
  created_at timestamptz DEFAULT now()
);

-- Create indexes for attachments
CREATE INDEX IF NOT EXISTS idx_attachments_session_id ON attachments(session_id);
CREATE INDEX IF NOT EXISTS idx_attachments_content_type ON attachments(content_type);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for analyses table
DROP TRIGGER IF EXISTS update_analyses_updated_at ON analyses;
CREATE TRIGGER update_analyses_updated_at
    BEFORE UPDATE ON analyses
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Enable Row Level Security
ALTER TABLE analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE attachments ENABLE ROW LEVEL SECURITY;

-- Create policies for public access (no authentication required)
CREATE POLICY "Allow public read access to analyses"
  ON analyses
  FOR SELECT
  TO anon, authenticated
  USING (true);

CREATE POLICY "Allow public insert to analyses"
  ON analyses
  FOR INSERT
  TO anon, authenticated
  WITH CHECK (true);

CREATE POLICY "Allow public read access to sessions"
  ON sessions
  FOR ALL
  TO anon, authenticated
  USING (true);

CREATE POLICY "Allow public read access to attachments"
  ON attachments
  FOR ALL
  TO anon, authenticated
  USING (true);