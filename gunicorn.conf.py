#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v2.0 - Gunicorn Configuration for Production
Configuração robusta para produção com Gunicorn
"""

import os
import multiprocessing

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', 5000)}"
backlog = 2048

# Worker processes
workers = int(os.getenv('GUNICORN_WORKERS', multiprocessing.cpu_count() * 2 + 1))
worker_class = "sync"
worker_connections = 1000
timeout = 60
keepalive = 2

# Restart workers after this many requests, to help prevent memory leaks
max_requests = 1000
max_requests_jitter = 100

# Preload app for better performance
preload_app = True

# Logging
accesslog = "logs/gunicorn_access.log" if os.getenv('LOG_FILE_ENABLED', 'true').lower() == 'true' else "-"
errorlog = "logs/gunicorn_error.log" if os.getenv('LOG_FILE_ENABLED', 'true').lower() == 'true' else "-"
loglevel = os.getenv('LOG_LEVEL', 'info').lower()
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'arqv30_enhanced'

# Server mechanics
daemon = False
pidfile = 'logs/gunicorn.pid'
user = None
group = None
tmp_upload_dir = None

# SSL (if needed)
keyfile = os.getenv('SSL_KEYFILE')
certfile = os.getenv('SSL_CERTFILE')

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Performance tuning
worker_tmp_dir = '/dev/shm' if os.path.exists('/dev/shm') else None

def when_ready(server):
    """Called just after the server is started"""
    server.log.info("🚀 ARQV30 Enhanced v2.0 server is ready. Listening on: %s", server.address)

def worker_int(worker):
    """Called just after a worker exited on SIGINT or SIGQUIT"""
    worker.log.info("🛑 Worker received INT or QUIT signal")

def pre_fork(server, worker):
    """Called just before a worker is forked"""
    server.log.info("🔄 Worker %s is being forked", worker.pid)

def post_fork(server, worker):
    """Called just after a worker has been forked"""
    server.log.info("✅ Worker %s forked successfully", worker.pid)

def worker_abort(worker):
    """Called when a worker received the SIGABRT signal"""
    worker.log.info("💥 Worker %s aborted", worker.pid)

def pre_exec(server):
    """Called just before a new master process is forked"""
    server.log.info("🔄 Forked child, re-executing")

def pre_request(worker, req):
    """Called just before a worker processes the request"""
    worker.log.debug("🔍 Processing request: %s %s", req.method, req.path)

def post_request(worker, req, environ, resp):
    """Called after a worker processes the request"""
    worker.log.debug("✅ Request processed: %s %s - %s", req.method, req.path, resp.status_code)

# Environment variables
raw_env = [
    f'FLASK_ENV={os.getenv("FLASK_ENV", "production")}',
    f'PYTHONPATH={os.getcwd()}/src',
]

# Graceful timeout
graceful_timeout = 30

# Enable stdio inheritance
enable_stdio_inheritance = True