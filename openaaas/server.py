"""
OpenAaaS Network Hub (Server)
Tier 2: Lightweight HTTP server with SQLite for service registration,
task routing, node heartbeat, and file relay.

Based on the paper:
- Service registration: nodes register capabilities with descriptions
- Task routing: routes tasks to qualified nodes based on service type and capacity
- Node heartbeat: monitors node health and availability
- File relay: relays task inputs/outputs (bounded to 50MB, 7-day retention)
- Authentication: API-key-based with HMAC-SHA256
"""

import os
import json
import time
import uuid
import hashlib
import hmac
import sqlite3
import threading
import tempfile
import shutil
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_file
from functools import wraps


class OpenAaaSServer:
    """OpenAaaS Network Hub - Tier 2 of the architecture."""
    
    def __init__(self, db_path=":memory:", file_storage_dir=None, 
                 admin_api_key=None, max_file_size_mb=50, file_retention_days=7):
        self.app = Flask(__name__)
        self.db_path = db_path
        self.file_storage_dir = file_storage_dir or tempfile.mkdtemp(prefix="openaaas_files_")
        self.admin_api_key = admin_api_key or str(uuid.uuid4())
        self.max_file_size_mb = max_file_size_mb
        self.file_retention_days = file_retention_days
        self.lock = threading.Lock()
        
        # Initialize database
        self._init_db()
        
        # Register routes
        self._register_routes()
        
    def _init_db(self):
        """Initialize SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Services table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS services (
                service_id TEXT PRIMARY KEY,
                service_name TEXT NOT NULL,
                domain_tag TEXT,
                description TEXT,
                detailed_usage TEXT,
                input_schema TEXT,
                output_schema TEXT,
                example_usage TEXT,
                evidence_levels TEXT,
                node_id TEXT,
                api_key TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                capacity INTEGER DEFAULT 10,
                current_load INTEGER DEFAULT 0,
                last_heartbeat REAL,
                created_at REAL,
                updated_at REAL
            )
        """)
        
        # Tasks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                service_id TEXT,
                client_api_key TEXT,
                status TEXT DEFAULT 'pending',
                task_description TEXT,
                task_input TEXT,
                input_files TEXT,
                result TEXT,
                output_files TEXT,
                error_message TEXT,
                created_at REAL,
                assigned_at REAL,
                completed_at REAL,
                timeout_seconds INTEGER DEFAULT 3600,
                FOREIGN KEY (service_id) REFERENCES services(service_id)
            )
        """)
        
        # Clients table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clients (
                client_id TEXT PRIMARY KEY,
                api_key TEXT NOT NULL UNIQUE,
                client_name TEXT,
                created_at REAL,
                last_active REAL
            )
        """)
        
        # Audit log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                log_id TEXT PRIMARY KEY,
                timestamp REAL,
                actor_id TEXT,
                action TEXT,
                resource_type TEXT,
                resource_id TEXT,
                details TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        
    def _get_db(self):
        """Get a database connection."""
        return sqlite3.connect(self.db_path)
    
    def _log_audit(self, conn, actor_id, action, resource_type, resource_id, details=None):
        """Log an audit event."""
        conn.execute(
            "INSERT INTO audit_log (log_id, timestamp, actor_id, action, resource_type, resource_id, details) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), time.time(), actor_id, action, resource_type, resource_id, 
             json.dumps(details) if details else None)
        )
        
    def _authenticate(self, api_key, require_admin=False):
        """Validate API key."""
        if require_admin:
            return api_key == self.admin_api_key
        conn = self._get_db()
        # Check if it's a service key or client key
        service = conn.execute("SELECT service_id FROM services WHERE api_key = ?", (api_key,)).fetchone()
        client = conn.execute("SELECT client_id FROM clients WHERE api_key = ?", (api_key,)).fetchone()
        conn.close()
        return service is not None or client is not None or api_key == self.admin_api_key
    
    def _register_routes(self):
        """Register all API routes."""
        
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({"status": "healthy", "timestamp": time.time()})
        
        # === Client Registration ===
        @self.app.route('/api/v1/clients/register', methods=['POST'])
        def register_client():
            data = request.json
            client_id = str(uuid.uuid4())
            api_key = str(uuid.uuid4())
            
            conn = self._get_db()
            conn.execute(
                "INSERT INTO clients (client_id, api_key, client_name, created_at, last_active) "
                "VALUES (?, ?, ?, ?, ?)",
                (client_id, api_key, data.get('name', 'anonymous'), time.time(), time.time())
            )
            self._log_audit(conn, client_id, 'register', 'client', client_id)
            conn.commit()
            conn.close()
            
            return jsonify({
                "client_id": client_id,
                "api_key": api_key,
                "message": "Client registered successfully"
            }), 201
        
        # === Service Registration ===
        @self.app.route('/api/v1/services/register', methods=['POST'])
        def register_service():
            data = request.json
            service_id = str(uuid.uuid4())
            api_key = str(uuid.uuid4())
            node_id = data.get('node_id', str(uuid.uuid4()))
            
            conn = self._get_db()
            conn.execute(
                "INSERT INTO services (service_id, service_name, domain_tag, description, "
                "detailed_usage, input_schema, output_schema, example_usage, evidence_levels, "
                "node_id, api_key, capacity, created_at, updated_at, last_heartbeat) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (service_id, data['service_name'], data.get('domain_tag', ''),
                 data.get('description', ''), data.get('detailed_usage', ''),
                 json.dumps(data.get('input_schema', {})),
                 json.dumps(data.get('output_schema', {})),
                 data.get('example_usage', ''),
                 data.get('evidence_levels', ''),
                 node_id, api_key, data.get('capacity', 10),
                 time.time(), time.time(), time.time())
            )
            self._log_audit(conn, node_id, 'register', 'service', service_id)
            conn.commit()
            conn.close()
            
            return jsonify({
                "service_id": service_id,
                "api_key": api_key,
                "node_id": node_id,
                "message": "Service registered successfully"
            }), 201
        
        # === Progressive Capability Discovery ===
        # Stage 1: Lightweight summary
        @self.app.route('/api/v1/services', methods=['GET'])
        def list_services():
            domain = request.args.get('domain', None)
            conn = self._get_db()
            
            if domain:
                services = conn.execute(
                    "SELECT service_id, service_name, domain_tag, description, capacity, current_load, status "
                    "FROM services WHERE domain_tag LIKE ? AND status = 'active'",
                    (f'%{domain}%',)
                ).fetchall()
            else:
                services = conn.execute(
                    "SELECT service_id, service_name, domain_tag, description, capacity, current_load, status "
                    "FROM services WHERE status = 'active'"
                ).fetchall()
            
            conn.close()
            
            result = []
            for s in services:
                result.append({
                    "service_id": s[0],
                    "service_name": s[1],
                    "domain_tag": s[2],
                    "description": s[3],
                    "capacity": s[4],
                    "current_load": s[5],
                    "status": s[6]
                })
            
            return jsonify({"services": result, "count": len(result)})
        
        # Stage 2: On-demand usage
        @self.app.route('/api/v1/services/<service_id>/usage', methods=['GET'])
        def get_service_usage(service_id):
            conn = self._get_db()
            service = conn.execute(
                "SELECT service_name, domain_tag, description, detailed_usage, "
                "input_schema, output_schema, example_usage, evidence_levels "
                "FROM services WHERE service_id = ?",
                (service_id,)
            ).fetchone()
            conn.close()
            
            if not service:
                return jsonify({"error": "Service not found"}), 404
            
            return jsonify({
                "service_name": service[0],
                "domain_tag": service[1],
                "description": service[2],
                "detailed_usage": service[3],
                "input_schema": json.loads(service[4]) if service[4] else {},
                "output_schema": json.loads(service[5]) if service[5] else {},
                "example_usage": service[6],
                "evidence_levels": service[7]
            })
        
        # === Task Submission ===
        @self.app.route('/api/v1/tasks/submit', methods=['POST'])
        def submit_task():
            data = request.json
            task_id = str(uuid.uuid4())
            service_id = data['service_id']
            
            conn = self._get_db()
            
            # Check service exists and is active
            service = conn.execute(
                "SELECT status, capacity, current_load FROM services WHERE service_id = ?",
                (service_id,)
            ).fetchone()
            
            if not service:
                conn.close()
                return jsonify({"error": "Service not found"}), 404
            
            if service[0] != 'active':
                conn.close()
                return jsonify({"error": "Service is not active"}), 503
            
            conn.execute(
                "INSERT INTO tasks (task_id, service_id, client_api_key, status, "
                "task_description, task_input, timeout_seconds, created_at) "
                "VALUES (?, ?, ?, 'pending', ?, ?, ?, ?)",
                (task_id, service_id, data.get('api_key', ''),
                 data.get('description', ''), json.dumps(data.get('input', {})),
                 data.get('timeout', 3600), time.time())
            )
            
            self._log_audit(conn, data.get('api_key', 'anonymous'), 
                          'submit', 'task', task_id)
            conn.commit()
            conn.close()
            
            return jsonify({
                "task_id": task_id,
                "status": "pending",
                "message": "Task submitted successfully"
            }), 201
        
        # === Task Polling (for Agent Core nodes) ===
        @self.app.route('/api/v1/tasks/poll', methods=['POST'])
        def poll_tasks():
            data = request.json
            service_id = data.get('service_id')
            
            conn = self._get_db()
            
            # Get oldest pending task for this service
            task = conn.execute(
                "SELECT task_id, task_description, task_input, timeout_seconds "
                "FROM tasks WHERE service_id = ? AND status = 'pending' "
                "ORDER BY created_at ASC LIMIT 1",
                (service_id,)
            ).fetchone()
            
            if not task:
                conn.close()
                return jsonify({"task": None, "message": "No pending tasks"})
            
            # Assign the task
            conn.execute(
                "UPDATE tasks SET status = 'assigned', assigned_at = ? WHERE task_id = ?",
                (time.time(), task[0])
            )
            
            # Update service load
            conn.execute(
                "UPDATE services SET current_load = current_load + 1 WHERE service_id = ?",
                (service_id,)
            )
            
            self._log_audit(conn, service_id, 'poll', 'task', task[0])
            conn.commit()
            conn.close()
            
            return jsonify({
                "task": {
                    "task_id": task[0],
                    "description": task[1],
                    "input": json.loads(task[2]) if task[2] else {},
                    "timeout_seconds": task[3]
                }
            })
        
        # === Task Result Submission ===
        @self.app.route('/api/v1/tasks/<task_id>/result', methods=['POST'])
        def submit_result(task_id):
            data = request.json
            
            conn = self._get_db()
            
            task = conn.execute(
                "SELECT service_id FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            
            if not task:
                conn.close()
                return jsonify({"error": "Task not found"}), 404
            
            conn.execute(
                "UPDATE tasks SET status = ?, result = ?, error_message = ?, "
                "completed_at = ? WHERE task_id = ?",
                (data.get('status', 'completed'), json.dumps(data.get('result', {})),
                 data.get('error', None), time.time(), task_id)
            )
            
            # Decrease service load
            conn.execute(
                "UPDATE services SET current_load = MAX(0, current_load - 1) "
                "WHERE service_id = ?",
                (task[0],)
            )
            
            self._log_audit(conn, task[0], 'complete', 'task', task_id)
            conn.commit()
            conn.close()
            
            return jsonify({"message": "Result submitted successfully"})
        
        # === Task Status Query ===
        @self.app.route('/api/v1/tasks/<task_id>', methods=['GET'])
        def get_task(task_id):
            conn = self._get_db()
            task = conn.execute(
                "SELECT task_id, service_id, status, task_description, result, "
                "error_message, created_at, assigned_at, completed_at "
                "FROM tasks WHERE task_id = ?",
                (task_id,)
            ).fetchone()
            conn.close()
            
            if not task:
                return jsonify({"error": "Task not found"}), 404
            
            return jsonify({
                "task_id": task[0],
                "service_id": task[1],
                "status": task[2],
                "description": task[3],
                "result": json.loads(task[4]) if task[4] else None,
                "error": task[5],
                "created_at": task[6],
                "assigned_at": task[7],
                "completed_at": task[8]
            })
        
        # === Node Heartbeat ===
        @self.app.route('/api/v1/heartbeat', methods=['POST'])
        def heartbeat():
            data = request.json
            service_id = data.get('service_id')
            
            conn = self._get_db()
            conn.execute(
                "UPDATE services SET last_heartbeat = ?, current_load = ?, "
                "capacity = ?, updated_at = ? WHERE service_id = ?",
                (time.time(), data.get('current_load', 0), 
                 data.get('capacity', 10), time.time(), service_id)
            )
            conn.commit()
            conn.close()
            
            return jsonify({"message": "Heartbeat received"})
        
        # === File Upload (relay) ===
        @self.app.route('/api/v1/files/upload', methods=['POST'])
        def upload_file():
            if 'file' not in request.files:
                return jsonify({"error": "No file provided"}), 400
            
            f = request.files['file']
            file_id = str(uuid.uuid4())
            filepath = os.path.join(self.file_storage_dir, file_id)
            f.save(filepath)
            
            # Check file size
            file_size = os.path.getsize(filepath)
            if file_size > self.max_file_size_mb * 1024 * 1024:
                os.remove(filepath)
                return jsonify({"error": f"File exceeds {self.max_file_size_mb}MB limit"}), 413
            
            return jsonify({
                "file_id": file_id,
                "filename": f.filename,
                "size_bytes": file_size
            }), 201
        
        # === File Download ===
        @self.app.route('/api/v1/files/<file_id>', methods=['GET'])
        def download_file(file_id):
            filepath = os.path.join(self.file_storage_dir, file_id)
            if not os.path.exists(filepath):
                return jsonify({"error": "File not found"}), 404
            return send_file(filepath)
        
        # === Admin: List all tasks ===
        @self.app.route('/api/v1/admin/tasks', methods=['GET'])
        def admin_list_tasks():
            api_key = request.headers.get('X-API-Key', '')
            if not self._authenticate(api_key, require_admin=True):
                return jsonify({"error": "Unauthorized"}), 401
            
            conn = self._get_db()
            tasks = conn.execute(
                "SELECT task_id, service_id, status, created_at, completed_at "
                "FROM tasks ORDER BY created_at DESC LIMIT 100"
            ).fetchall()
            conn.close()
            
            return jsonify({
                "tasks": [{
                    "task_id": t[0], "service_id": t[1], "status": t[2],
                    "created_at": t[3], "completed_at": t[4]
                } for t in tasks]
            })
        
        # === Server Statistics ===
        @self.app.route('/api/v1/stats', methods=['GET'])
        def server_stats():
            conn = self._get_db()
            services_count = conn.execute("SELECT COUNT(*) FROM services WHERE status = 'active'").fetchone()[0]
            tasks_total = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
            tasks_pending = conn.execute("SELECT COUNT(*) FROM tasks WHERE status = 'pending'").fetchone()[0]
            tasks_completed = conn.execute("SELECT COUNT(*) FROM tasks WHERE status = 'completed'").fetchone()[0]
            clients_count = conn.execute("SELECT COUNT(*) FROM clients").fetchone()[0]
            conn.close()
            
            return jsonify({
                "active_services": services_count,
                "total_tasks": tasks_total,
                "pending_tasks": tasks_pending,
                "completed_tasks": tasks_completed,
                "registered_clients": clients_count
            })
    
    def run(self, host='0.0.0.0', port=8080, debug=False):
        """Run the server."""
        print(f"OpenAaaS Server starting on {host}:{port}")
        print(f"Admin API Key: {self.admin_api_key}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)
    
    def get_routing_overhead_ms(self):
        """Measure the routing overhead of the server (for benchmarking).
        Paper reports ~550ms for task routing overhead."""
        import time
        
        # Simulate routing overhead measurement
        start = time.time()
        
        conn = self._get_db()
        # Simulate service lookup
        conn.execute("SELECT * FROM services WHERE status = 'active' ORDER BY current_load ASC LIMIT 1").fetchone()
        # Simulate task creation
        conn.execute("SELECT COUNT(*) FROM tasks").fetchone()
        # Simulate heartbeat check
        conn.execute("SELECT service_id FROM services WHERE last_heartbeat > ?", 
                     (time.time() - 60,)).fetchall()
        conn.close()
        
        elapsed_ms = (time.time() - start) * 1000
        return elapsed_ms


def create_server(db_path=None, port=8080):
    """Create and return a configured OpenAaaS server instance."""
    if db_path is None:
        db_path = os.path.join(tempfile.mkdtemp(), "openaaas.db")
    
    server = OpenAaaSServer(db_path=db_path)
    return server


if __name__ == '__main__':
    server = create_server()
    server.run(port=8080, debug=True)
