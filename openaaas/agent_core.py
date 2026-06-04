"""
OpenAaaS Agent Core (Network Node)
Tier 3: Domain-specific execution engine deployed at data sites.

Handles:
- Registration with the Server
- Polling for tasks
- Executing tasks in isolated environment
- Reporting results back to the Server
- Heartbeat sending
"""

import os
import json
import time
import uuid
import threading
import requests
import traceback
from typing import Callable, Dict, Any, Optional


class AgentCore:
    """OpenAaaS Agent Core - Tier 3 execution node."""
    
    def __init__(self, server_url: str, service_name: str, 
                 domain_tag: str = "", description: str = "",
                 detailed_usage: str = "", capacity: int = 10,
                 executor: Callable = None, poll_interval: int = 5,
                 heartbeat_interval: int = 30):
        self.server_url = server_url.rstrip('/')
        self.service_name = service_name
        self.domain_tag = domain_tag
        self.description = description
        self.detailed_usage = detailed_usage
        self.capacity = capacity
        self.executor = executor
        self.poll_interval = poll_interval
        self.heartbeat_interval = heartbeat_interval
        
        self.service_id = None
        self.api_key = None
        self.node_id = None
        self.running = False
        self.current_load = 0
        
    def register(self):
        """Register this node with the OpenAaaS Server."""
        response = requests.post(
            f"{self.server_url}/api/v1/services/register",
            json={
                "service_name": self.service_name,
                "domain_tag": self.domain_tag,
                "description": self.description,
                "detailed_usage": self.detailed_usage,
                "capacity": self.capacity
            }
        )
        
        if response.status_code == 201:
            data = response.json()
            self.service_id = data['service_id']
            self.api_key = data['api_key']
            self.node_id = data['node_id']
            print(f"Registered service '{self.service_name}' "
                  f"(ID: {self.service_id}, Node: {self.node_id})")
            return True
        else:
            print(f"Registration failed: {response.text}")
            return False
    
    def poll_and_execute(self):
        """Poll for a task and execute it."""
        try:
            response = requests.post(
                f"{self.server_url}/api/v1/tasks/poll",
                json={"service_id": self.service_id}
            )
            
            if response.status_code != 200:
                return False
            
            data = response.json()
            task = data.get('task')
            
            if not task:
                return False
            
            task_id = task['task_id']
            print(f"Received task {task_id}: {task.get('description', '')[:100]}")
            self.current_load += 1
            
            try:
                # Execute the task using the registered executor
                if self.executor:
                    result = self.executor(task)
                else:
                    result = {"message": "No executor configured", "status": "error"}
                
                # Submit result
                requests.post(
                    f"{self.server_url}/api/v1/tasks/{task_id}/result",
                    json={
                        "status": "completed",
                        "result": result
                    }
                )
                print(f"Task {task_id} completed successfully")
                
            except Exception as e:
                # Report error
                requests.post(
                    f"{self.server_url}/api/v1/tasks/{task_id}/result",
                    json={
                        "status": "failed",
                        "error": str(e),
                        "result": {"traceback": traceback.format_exc()}
                    }
                )
                print(f"Task {task_id} failed: {e}")
            
            finally:
                self.current_load = max(0, self.current_load - 1)
            
            return True
            
        except requests.exceptions.ConnectionError:
            return False
    
    def send_heartbeat(self):
        """Send heartbeat to the server."""
        try:
            requests.post(
                f"{self.server_url}/api/v1/heartbeat",
                json={
                    "service_id": self.service_id,
                    "current_load": self.current_load,
                    "capacity": self.capacity
                }
            )
        except requests.exceptions.ConnectionError:
            pass
    
    def _heartbeat_loop(self):
        """Background heartbeat loop."""
        while self.running:
            self.send_heartbeat()
            time.sleep(self.heartbeat_interval)
    
    def _poll_loop(self):
        """Background polling loop."""
        while self.running:
            self.poll_and_execute()
            time.sleep(self.poll_interval)
    
    def start(self, background=True):
        """Start the agent core."""
        if not self.service_id:
            raise RuntimeError("Must register before starting")
        
        self.running = True
        
        if background:
            self.heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop, daemon=True)
            self.poll_thread = threading.Thread(
                target=self._poll_loop, daemon=True)
            self.heartbeat_thread.start()
            self.poll_thread.start()
            print(f"Agent Core started in background (poll={self.poll_interval}s, "
                  f"heartbeat={self.heartbeat_interval}s)")
        else:
            # Foreground mode
            heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop, daemon=True)
            heartbeat_thread.start()
            
            print(f"Agent Core running (press Ctrl+C to stop)")
            try:
                while self.running:
                    self.poll_and_execute()
                    time.sleep(self.poll_interval)
            except KeyboardInterrupt:
                self.running = False
    
    def stop(self):
        """Stop the agent core."""
        self.running = False


class MasterAgent:
    """
    OpenAaaS Master Agent - Tier 1.
    General-purpose agent that decomposes tasks and orchestrates sub-agents.
    """
    
    def __init__(self, server_url: str, name: str = "MasterAgent"):
        self.server_url = server_url.rstrip('/')
        self.name = name
        self.client_id = None
        self.api_key = None
    
    def register(self):
        """Register as a client with the server."""
        response = requests.post(
            f"{self.server_url}/api/v1/clients/register",
            json={"name": self.name}
        )
        if response.status_code == 201:
            data = response.json()
            self.client_id = data['client_id']
            self.api_key = data['api_key']
            print(f"Master Agent '{self.name}' registered (ID: {self.client_id})")
            return True
        return False
    
    def discover_services(self, domain: str = None):
        """Stage 1: Lightweight service discovery."""
        params = {}
        if domain:
            params['domain'] = domain
        
        response = requests.get(
            f"{self.server_url}/api/v1/services",
            params=params
        )
        
        if response.status_code == 200:
            return response.json()['services']
        return []
    
    def get_service_usage(self, service_id: str):
        """Stage 2: On-demand detailed usage."""
        response = requests.get(
            f"{self.server_url}/api/v1/services/{service_id}/usage"
        )
        
        if response.status_code == 200:
            return response.json()
        return None
    
    def submit_task(self, service_id: str, description: str, 
                    task_input: dict = None, timeout: int = 3600):
        """Submit a task to a specific service."""
        response = requests.post(
            f"{self.server_url}/api/v1/tasks/submit",
            json={
                "service_id": service_id,
                "description": description,
                "input": task_input or {},
                "api_key": self.api_key,
                "timeout": timeout
            }
        )
        
        if response.status_code == 201:
            return response.json()['task_id']
        return None
    
    def get_task_result(self, task_id: str, poll_interval: int = 2, 
                        max_wait: int = 3600):
        """Poll for task result with timeout."""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            response = requests.get(
                f"{self.server_url}/api/v1/tasks/{task_id}"
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['status'] in ('completed', 'failed'):
                    return data
            
            time.sleep(poll_interval)
        
        return {"status": "timeout", "error": "Task did not complete within timeout"}
    
    def execute_workflow(self, service_id: str, description: str, 
                        task_input: dict = None, timeout: int = 3600):
        """Execute a complete workflow: submit task and wait for result."""
        task_id = self.submit_task(service_id, description, task_input, timeout)
        if task_id:
            return self.get_task_result(task_id, max_wait=timeout)
        return {"status": "error", "error": "Failed to submit task"}
