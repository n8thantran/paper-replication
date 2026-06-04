"""
Generate Table 3: Architectural Comparison from the paper.

This table positions OpenAaaS against representative systems along dimensions
that matter for real-world materials-informatics deployment.
"""

import json
import os


def generate_table3():
    """Generate the architectural comparison table (Table 3 in the paper)."""
    
    # Data from Table 3 in the paper (Section 5.1)
    systems = [
        {
            "name": "Materials Project",
            "multi_agent": False,
            "cross_org_secure": False,
            "tool_composability": "API",
            "data_sovereignty": False,
            "near_data_exec": False,
            "broad_client_compat": "REST"
        },
        {
            "name": "AutoGen",
            "multi_agent": True,
            "cross_org_secure": False,
            "tool_composability": "Python",
            "data_sovereignty": False,
            "near_data_exec": False,
            "broad_client_compat": "LangChain"
        },
        {
            "name": "AaaS-AN",
            "multi_agent": True,
            "cross_org_secure": False,
            "tool_composability": "RGPS",
            "data_sovereignty": "Partial",
            "near_data_exec": False,
            "broad_client_compat": "Custom"
        },
        {
            "name": "MCP Servers",
            "multi_agent": False,
            "cross_org_secure": False,
            "tool_composability": "JSON-RPC",
            "data_sovereignty": False,
            "near_data_exec": False,
            "broad_client_compat": "MCP Hosts"
        },
        {
            "name": "OpenAaaS",
            "multi_agent": True,
            "cross_org_secure": True,
            "tool_composability": "Standard",
            "data_sovereignty": True,
            "near_data_exec": True,
            "broad_client_compat": "MCP + Plugins"
        }
    ]
    
    # Generate text table
    lines = []
    lines.append("=" * 100)
    lines.append("TABLE 3: Architectural Comparison")
    lines.append("=" * 100)
    lines.append("")
    
    header = f"{'System':<20} {'Multi-Agent':>12} {'Cross-Org':>12} {'Tool':>14} {'Data':>14} {'Near-Data':>12} {'Broad Client':>14}"
    subheader = f"{'':20} {'Support':>12} {'Secure':>12} {'Composability':>14} {'Sovereignty':>14} {'Exec.':>12} {'Compat.':>14}"
    lines.append(header)
    lines.append(subheader)
    lines.append("-" * 100)
    
    for sys in systems:
        def fmt_bool(v):
            if v is True:
                return "✓"
            elif v is False:
                return "✗"
            else:
                return str(v)
        
        bold = "**" if sys["name"] == "OpenAaaS" else ""
        name = f"{bold}{sys['name']}{bold}"
        
        line = f"{name:<20} {fmt_bool(sys['multi_agent']):>12} {fmt_bool(sys['cross_org_secure']):>12} {sys['tool_composability']:>14} {fmt_bool(sys['data_sovereignty']):>14} {fmt_bool(sys['near_data_exec']):>12} {sys['broad_client_compat']:>14}"
        lines.append(line)
    
    lines.append("-" * 100)
    lines.append("")
    lines.append("Key differentiator: OpenAaaS is the only system that simultaneously provides")
    lines.append("multi-agent orchestration, cross-organizational security, and near-data execution.")
    lines.append("")
    lines.append("Notes:")
    lines.append("- Materials Project: Centralized data warehouse with REST API. Single-agent, data-centralized.")
    lines.append("- AutoGen: Multi-agent conversation framework. Agents are Python objects in same runtime.")
    lines.append("- AaaS-AN: Service-oriented agent paradigm with RGPS standard. Primarily intra-organizational.")
    lines.append("- MCP Servers: Tool-calling protocol (JSON-RPC). Not an agent-network architecture.")
    lines.append("- OpenAaaS: Hierarchical AaaS with near-data execution and data sovereignty.")
    lines.append("")
    
    table_text = "\n".join(lines)
    
    # Also generate markdown version
    md_lines = []
    md_lines.append("# Table 3: Architectural Comparison")
    md_lines.append("")
    md_lines.append("| System | Multi-Agent Support | Cross-Org Secure | Tool Composability | Data Sovereignty | Near-Data Exec. | Broad Client Compat. |")
    md_lines.append("|--------|:------------------:|:----------------:|:------------------:|:----------------:|:---------------:|:--------------------:|")
    
    for sys in systems:
        def fmt_md(v):
            if v is True:
                return "✓"
            elif v is False:
                return "✗"
            else:
                return str(v)
        
        bold = "**" if sys["name"] == "OpenAaaS" else ""
        name = f"{bold}{sys['name']}{bold}"
        
        row = f"| {name} | {fmt_md(sys['multi_agent'])} | {fmt_md(sys['cross_org_secure'])} | {sys['tool_composability']} | {fmt_md(sys['data_sovereignty'])} | {fmt_md(sys['near_data_exec'])} | {sys['broad_client_compat']} |"
        md_lines.append(row)
    
    md_lines.append("")
    md_lines.append("**Key differentiator**: OpenAaaS is the only system that simultaneously provides multi-agent orchestration, cross-organizational security, and near-data execution.")
    md_lines.append("")
    
    md_text = "\n".join(md_lines)
    
    # Save
    os.makedirs("results", exist_ok=True)
    
    with open("results/table3.txt", "w") as f:
        f.write(table_text)
    
    with open("results/table3.md", "w") as f:
        f.write(md_text)
    
    with open("results/table3.json", "w") as f:
        json.dump(systems, f, indent=2)
    
    print(table_text)
    print("Saved to results/table3.txt, results/table3.md, results/table3.json")
    
    return systems


if __name__ == "__main__":
    generate_table3()
