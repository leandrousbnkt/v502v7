-- ARQV30 Enhanced v2.0 - Database Migration
-- Create analyses table for storing market analysis data

-- Create analyses table
CREATE TABLE IF NOT EXISTS analyses (
    id SERIAL PRIMARY KEY,
    nicho VARCHAR(255) NOT NULL,
    produto VARCHAR(255),
    descricao TEXT,
    preco DECIMAL(10,2),
    publico VARCHAR(500),
    concorrentes TEXT,
    dados_adicionais TEXT,
    objetivo_receita DECIMAL(15,2),
    orcamento_marketing DECIMAL(15,2),
    prazo_lancamento VARCHAR(100),
    
    -- JSON fields for analysis results
    avatar_data JSONB,
    positioning_data JSONB,
    competition_data JSONB,
    marketing_data JSONB,
    metrics_data JSONB,
    funnel_data JSONB,
    market_intelligence JSONB,
    action_plan JSONB,
    comprehensive_analysis JSONB,
    
    -- Status and metadata
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_analyses_nicho ON analyses(nicho);
CREATE INDEX IF NOT EXISTS idx_analyses_status ON analyses(status);
CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at);

-- Create sessions table for tracking user sessions
CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_agent TEXT,
    ip_address INET,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for sessions
CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at);

-- Create attachments table for file uploads
CREATE TABLE IF NOT EXISTS attachments (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    file_size INTEGER,
    mime_type VARCHAR(100),
    content_type VARCHAR(50),
    extracted_content TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for attachments
CREATE INDEX IF NOT EXISTS idx_attachments_session_id ON attachments(session_id);
CREATE INDEX IF NOT EXISTS idx_attachments_content_type ON attachments(content_type);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for analyses table
DROP TRIGGER IF EXISTS update_analyses_updated_at ON analyses;
CREATE TRIGGER update_analyses_updated_at
    BEFORE UPDATE ON analyses
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create trigger for sessions table
DROP TRIGGER IF EXISTS update_sessions_last_activity ON sessions;
CREATE TRIGGER update_sessions_last_activity
    BEFORE UPDATE ON sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Insert sample data for testing (optional)
-- INSERT INTO analyses (nicho, produto, preco, publico, status) VALUES
-- ('Produtos Digitais', 'Curso Online', 997.00, 'Empreendedores digitais', 'completed'),
-- ('E-commerce', 'Loja Virtual', 2500.00, 'Pequenos comerciantes', 'pending');

-- Grant permissions (adjust as needed for your Supabase setup)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO authenticated;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO authenticated;

