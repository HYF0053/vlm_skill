import os
import sys
import json
import subprocess
import re
from pathlib import Path

def get_mcp_capabilities():
    try:
        script_dir = Path(__file__).parent
        mcp_client_path = script_dir / "mcp_client.py"
        
        # Get servers
        servers_res = subprocess.run([sys.executable, str(mcp_client_path), "servers"], capture_output=True, text=True)
        if servers_res.returncode != 0:
            return "Error: Could not list servers."
            
        servers = json.loads(servers_res.stdout)
        if not servers:
            return "No MCP servers configured."
            
        output = ""
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

def update_skill_md():
    capabilities = get_mcp_capabilities()
    if not capabilities or capabilities.startswith("Error"):
        print(f"Aborting update: {capabilities}")
        return

    script_dir = Path(__file__).parent
    skill_md_path = script_dir.parent / "SKILL.md"
    
    if not skill_md_path.exists():
        print(f"Error: {skill_md_path} not found.")
        return

    content = skill_md_path.read_text(encoding='utf-8')
    
    section_title = "## Currently Available MCP Tools"
    marker = "> [!IMPORTANT]\n> The list below shows the tools available at the last check. Since MCP servers can be dynamic, ALWAYS run `python skills/mcp_client/scripts/mcp_client.py tools <server>` to get the most up-to-date schema before using a tool.\n"
    
    new_section_content = f"\n{section_title}\n\n{marker}\n{capabilities}"
    
    # Check if section already exists
    if section_title in content:
        # Replace existing section
        # We look for the section title and replace everything until the next header or end of file
        pattern = re.compile(rf"{re.escape(section_title)}.*?(?=\n##|$)", re.DOTALL)
        content = pattern.sub(new_section_content.strip() + "\n", content)
    else:
        # Append to end
        content = content.rstrip() + "\n\n" + new_section_content.strip() + "\n"
        
    skill_md_path.write_text(content, encoding='utf-8')
    print(f"Successfully updated {skill_md_path}")

if __name__ == "__main__":
    if "--update" in sys.argv:
        update_skill_md()
    else:
        print("Currently Configured MCP Servers & Tools:")
        print(get_mcp_capabilities())
