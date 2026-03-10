import os
import sys
import json
import subprocess
from pathlib import Path

def get_mcp_capabilities():
    try:
        script_dir = Path(__file__).parent
        mcp_client_path = script_dir / "mcp_client.py"
        
        # Get servers
        servers_res = subprocess.run([sys.executable, str(mcp_client_path), "servers"], capture_output=True, text=True)
        if servers_res.returncode != 0:
            return ""
            
        servers = json.loads(servers_res.stdout)
        if not servers:
            return "No MCP servers configured."
            
        output = "Currently Configured MCP Servers & Tools:\n"
        for server in servers:
            name = server.get("name")
            output += f"- Server: {name}\n"
            
            # Get tools for this server
            tools_res = subprocess.run([sys.executable, str(mcp_client_path), "tools", name], capture_output=True, text=True)
            if tools_res.returncode == 0:
                tools = json.loads(tools_res.stdout)
                if not tools:
                    output += "  * (No tools found)\n"
                for tool in tools:
                    desc = tool.get('description', 'No description')
                    tool_name = tool.get('name')
                    output += f"  * Tool '{tool_name}': {desc}\n"
            else:
                output += "  * (Could not fetch tools)\n"
                
        return output
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    print(get_mcp_capabilities())
