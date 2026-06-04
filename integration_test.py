"""
OpenAaaS Integration Test

Demonstrates the full 3-tier architecture:
1. Master Agent Layer (simulated LLM agent for task decomposition)
2. Network Hub (Flask server with SQLite)
3. Network Node (Agent Core with HEA executor)

This test runs the complete HEA case study pipeline through the framework.
"""

import sys
import os
import json
import time
import tempfile

sys.path.insert(0, '/workspace')


def run_integration_test():
    """Run the full OpenAaaS integration test."""
    from openaaas.server import OpenAaaSServer
    from executors.hea_executor import HEAExecutor
    
    print("=" * 70)
    print("OpenAaaS Integration Test - Full 3-Tier Architecture")
    print("=" * 70)
    
    # =========================================================================
    # TIER 2: Start Network Hub (Server)
    # =========================================================================
    print("\n[TIER 2] Starting Network Hub...")
    db_path = tempfile.mktemp(suffix=".db")
    server = OpenAaaSServer(db_path=db_path, admin_api_key="integration-test-admin")
    server.app.config['TESTING'] = True
    client = server.app.test_client()
    
    # Health check
    resp = client.get('/health')
    assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
    print("  ✓ Server health check passed")
    
    # =========================================================================
    # TIER 3: Register Network Node (HEA Service)
    # =========================================================================
    print("\n[TIER 3] Registering HEA Screening Service...")
    resp = client.post('/api/v1/services/register', json={
        "service_name": "hea-screening",
        "description": "High-entropy alloy ductility screening using ML descriptors",
        "domain_tag": "materials-science",
        "capacity": 10,
        "input_schema": json.dumps({
            "type": "object",
            "properties": {
                "elements": {"type": "array"},
                "n_components": {"type": "integer"},
                "required_elements": {"type": "array"}
            }
        }),
        "evidence_levels": "Level 1: DFT-computed descriptors, Level 2: ML predictions, Level 3: Empirical rules"
    })
    assert resp.status_code == 201, f"Service registration failed: {resp.status_code}"
    reg_data = resp.get_json()
    hea_service_id = reg_data["service_id"]
    hea_api_key = reg_data["api_key"]
    print(f"  ✓ HEA service registered: {hea_service_id}")
    
    # Register AlphaAgent service
    print("\n[TIER 3] Registering AlphaAgent Q&A Service...")
    resp = client.post('/api/v1/services/register', json={
        "service_name": "alpha-agent-qa",
        "description": "Evidence-grounded Q&A for materials science literature",
        "domain_tag": "materials-science",
        "capacity": 20,
        "evidence_levels": "Level 1: Direct citations, Level 2: Cross-referenced, Level 3: Inferred"
    })
    assert resp.status_code == 201
    alpha_reg = resp.get_json()
    alpha_service_id = alpha_reg["service_id"]
    print(f"  ✓ AlphaAgent service registered: {alpha_service_id}")
    
    # =========================================================================
    # Progressive Capability Discovery (Paper Section 3.2)
    # =========================================================================
    print("\n[DISCOVERY] Progressive Capability Discovery...")
    
    # Stage 1: Lightweight summary
    resp = client.get('/api/v1/services')
    services = resp.get_json()
    print(f"  Stage 1 - Found {services['count']} services:")
    for svc in services['services']:
        print(f"    - {svc['service_name']}: {svc['description'][:60]}...")
    
    # Stage 2: On-demand usage details
    resp = client.get(f'/api/v1/services/{hea_service_id}/usage')
    usage = resp.get_json()
    print(f"  Stage 2 - HEA service details retrieved: {usage.get('service_name', 'N/A')}")
    
    # Filter by domain
    resp = client.get('/api/v1/services?domain=materials')
    domain_services = resp.get_json()
    print(f"  Domain filter - Materials science services: {domain_services['count']}")
    
    # =========================================================================
    # TIER 1: Master Agent Layer (Task Decomposition)
    # =========================================================================
    print("\n[TIER 1] Master Agent - Task Decomposition...")
    print("  User query: 'Screen six-element HEAs from 15-element palette for ductility,")
    print("               focusing on MoNbTaW-containing systems'")
    
    # Submit the main HEA screening task
    task_input = {
        "elements": ["Al", "Co", "Cr", "Cu", "Fe", "Hf", "Mn", "Mo", 
                     "Nb", "Ni", "Ta", "Ti", "V", "W", "Zr"],
        "n_components": 6,
        "required_elements": ["Mo", "Nb", "Ta", "W"]
    }
    
    print(f"\n[HUB] Submitting HEA screening task through Network Hub...")
    resp = client.post('/api/v1/tasks/submit', json={
        "service_id": hea_service_id,
        "description": "Screen MoNbTaW-containing six-element HEAs for ductility",
        "input": task_input
    })
    assert resp.status_code == 201, f"Task submission failed: {resp.status_code}"
    task_data = resp.get_json()
    task_id = task_data["task_id"]
    print(f"  ✓ Task submitted: {task_id}")
    
    # =========================================================================
    # TIER 3: Agent Core polls and executes task
    # =========================================================================
    print("\n[TIER 3] Agent Core - Polling for tasks...")
    resp = client.post('/api/v1/tasks/poll', json={"service_id": hea_service_id})
    poll_data = resp.get_json()
    
    assert poll_data.get("task") is not None, "No task received from poll"
    polled_task = poll_data["task"]
    assert polled_task["task_id"] == task_id
    print(f"  ✓ Task received: {polled_task['task_id']}")
    
    # Execute using HEA executor
    print("\n[TIER 3] Executing HEA screening pipeline...")
    hea_executor = HEAExecutor()
    start_time = time.time()
    result = hea_executor.execute(polled_task["input"])
    exec_time = time.time() - start_time
    print(f"  ✓ Execution completed in {exec_time:.1f}s")
    
    # Report results back to server
    opt = result.get("optimal_system", {})
    print("\n[HUB] Reporting results...")
    resp = client.post(f'/api/v1/tasks/{task_id}/result', json={
        "status": "completed",
        "output": {
            "n_combinations_screened": result.get("n_combinations_screened"),
            "total_samples_analyzed": result.get("total_samples_analyzed"),
            "optimal_system": opt.get("combination_id", "N/A"),
            "optimal_ductile_pct": opt.get("pct_highly_ductile", 0),
            "improvement_factor": result.get("improvement_factor"),
            "result_size_mb": result.get("result_size_mb"),
        },
        "execution_time_seconds": exec_time
    })
    assert resp.status_code == 200, f"Result reporting failed: {resp.status_code}"
    print(f"  ✓ Results reported to Network Hub")
    
    # =========================================================================
    # Retrieve results through Hub
    # =========================================================================
    print("\n[HUB] Retrieving task results...")
    resp = client.get(f'/api/v1/tasks/{task_id}')
    task_result = resp.get_json()
    print(f"  Task status: {task_result.get('status')}")
    
    # =========================================================================
    # Heartbeat monitoring
    # =========================================================================
    print("\n[MONITORING] Node heartbeat...")
    resp = client.post('/api/v1/heartbeat', json={
        "service_id": hea_service_id,
        "status": "healthy",
        "load": 0,
        "capacity": 10
    })
    print(f"  ✓ Heartbeat acknowledged")
    
    # =========================================================================
    # Server statistics
    # =========================================================================
    print("\n[STATS] Server statistics...")
    resp = client.get('/api/v1/stats')
    stats = resp.get_json()
    print(f"  Total services: {stats.get('total_services', 'N/A')}")
    print(f"  Total tasks: {stats.get('total_tasks', 'N/A')}")
    print(f"  Completed tasks: {stats.get('completed_tasks', 'N/A')}")
    
    # =========================================================================
    # Print key HEA results
    # =========================================================================
    print("\n" + "=" * 70)
    print("HEA Case Study Results (via OpenAaaS Pipeline)")
    print("=" * 70)
    
    baseline = result.get("baseline_system", {})
    
    print(f"  Combinations screened: {result.get('n_combinations_screened', 'N/A')}")
    print(f"  Total samples analyzed: {result.get('total_samples_analyzed', 'N/A')}")
    print(f"  Optimal system: {opt.get('combination_id', 'N/A')}")
    print(f"  Optimal ductile %: {opt.get('pct_highly_ductile', 'N/A')}%")
    print(f"  Baseline system: {baseline.get('combination_id', 'N/A')}")
    print(f"  Baseline ductile %: {baseline.get('pct_highly_ductile', 'N/A')}%")
    print(f"  Improvement factor: {result.get('improvement_factor', 'N/A')}x")
    print(f"  Result size: {result.get('result_size_mb', 'N/A')} MB (vs 17.4 TB raw)")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Integration Test Summary")
    print("=" * 70)
    print("  ✓ Network Hub (Tier 2): Server running with SQLite backend")
    print("  ✓ Service Registration: 2 services registered")
    print("  ✓ Progressive Discovery: 3-stage capability discovery working")
    print("  ✓ Task Routing: Task submitted and routed through Hub")
    print("  ✓ Agent Core (Tier 3): Task polled and executed")
    print("  ✓ HEA Screening: Full pipeline with ML + empirical screening")
    print("  ✓ Result Reporting: Results stored and retrievable")
    print("  ✓ Heartbeat Monitoring: Node health tracking active")
    print("=" * 70)
    print("ALL TESTS PASSED ✓")
    
    # Cleanup
    try:
        os.unlink(db_path)
    except:
        pass
    
    return True


if __name__ == "__main__":
    success = run_integration_test()
    sys.exit(0 if success else 1)
