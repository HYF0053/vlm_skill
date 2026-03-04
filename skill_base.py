import uuid
import os
from typing import TypedDict, NotRequired, Callable
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, AgentMiddleware
from langchain.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

# Import the new Skill Library
from skill_library import FileSystemSkillRepository

# Initialize Repository
# Assuming scripts are run from /home/ubuntu/ocr_test or similar root
# Using absolute path for safety based on previous tool outputs
SKILL_REPO_PATH = "/home/ubuntu/ocr_test/skills"
skill_repo = FileSystemSkillRepository(SKILL_REPO_PATH)


# Create skill loading tools
@tool
def load_skill_overview(skill_name: str) -> str:
    """Load the overview of a skill to understand its capabilities and usage.

    Args:
        skill_name: The name of the skill to load (e.g., "form_ocr_skill").
    """
    content = skill_repo.get_skill_overview(skill_name)
    if content:
        files = skill_repo.list_skill_files(skill_name)
        file_list_str = "\n".join([f"- {f}" for f in files])
        return (
            f"Loaded overview for skill: {skill_name}\n\n"
            f"{content}\n\n"
            f"Available files to read in this skill:\n{file_list_str}\n\n"
            f"Use 'read_skill_file' to read specific files."
        )
    else:
        # Suggest available skills
        skills = skill_repo.get_all_skills()
        available = ", ".join(s.name for s in skills)
        return f"Skill '{skill_name}' not found. Available skills: {available}"

@tool
def read_skill_file(skill_name: str, file_path: str) -> str:
    """Read a specific file from a skill's directory.
    
    Use this to read reference documents, examples, or specific instructions
    mentioned in the skill overview.

    Args:
        skill_name: The name of the skill.
        file_path: The relative path to the file within the skill directory (e.g., "references/accident_report.md").
    """
    content = skill_repo.get_skill_details(skill_name, file_path)
    return content

# Create skill middleware
class SkillMiddleware(AgentMiddleware):
    """Middleware that injects skill descriptions into the system prompt."""

    # Register the tools as class variables
    tools = [load_skill_overview, read_skill_file]

    def __init__(self):
        """Initialize and generate the skills prompt from the repository."""
        # Build skills prompt from the repository
        # We explicitly re-fetch here to ensure freshness if init happens at startup
        self.refresh_skills_prompt()

    def refresh_skills_prompt(self):
        skills = skill_repo.get_all_skills()
        skills_list = []
        for skill in skills:
            # We strip newlines from description to keep the prompt clean
            desc = skill.description.replace('\n', ' ').strip()
            skills_list.append(f"- **{skill.name}**: {desc}")
        
        if not skills_list:
            self.skills_prompt = "No skills found."
        else:
            self.skills_prompt = "\n".join(skills_list)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Sync: Inject skill descriptions into system prompt."""
        # Build the skills addendum
        skills_addendum = (
            f"\n\n## Available Skills\n\n{self.skills_prompt}\n\n"
            "Use 'load_skill_overview' to view details of a skill, and "
            "'read_skill_file' to read specific reference documents within a skill."
        )

        # Append to system message content blocks
        # Handle cases where content might be string or list of blocks
        original_content = request.system_message.content
        if isinstance(original_content, str):
            new_content = original_content + skills_addendum
        else:
             # It's likely a list of blocks
             new_content = list(original_content) + [
                {"type": "text", "text": skills_addendum}
             ]
             
        new_system_message = SystemMessage(content=new_content)
        modified_request = request.override(system_message=new_system_message)
        return handler(modified_request)

# Initialize Chat Model
# Disable LangSmith tracing to avoid errors if API key is missing/invalid
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

print("Initializing ChatOpenAI...")
llm = ChatOpenAI(
    model="Qwen/Qwen3-VL-32B-Instruct",
    api_key="",
    base_url="http://10.1.1.7:9000/v1",
    max_tokens=4096,
    temperature=0,
)

# Create the agent with skill support
agent = create_agent(
    llm,
    system_prompt=(
        "You are an intelligent assistant with access to a library of capabilities (skills). "
        "Use them to help the user."
    ),
    middleware=[SkillMiddleware()],
    checkpointer=InMemorySaver(),
)

# Example usage for verific# Example usage
if __name__ == "__main__":
    import argparse
    import base64
    from io import BytesIO

    def encode_image(image_path):
        """Encodes an image to base64."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return None
        except Exception as e:
            print(f"Error reading image: {e}")
            return None

    parser = argparse.ArgumentParser(description="Run the Skill Agent with optional image input.")
    parser.add_argument("--query", type=str, default="請幫我提取這張單據的資料，請自行判斷是哪一類單據，然後依照其種類去提取相對應的欄位資料", help="The user query.")
    parser.add_argument("--image", type=str, default="/home/ubuntu/Documents/2026-0120_新安東京 - 表單 AI OCR/A06現場圖/手寫/A06-H-N-260116-011.jpg")
 
    args = parser.parse_args()

    # Configuration for this conversation thread
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"Running query: {args.query}")
    
    message_content = []
    
    # Add text content
    message_content.append({"type": "text", "text": args.query})
    
    # Add image content if provided
    if args.image:
        print(f"Loading image from: {args.image}")
        base64_image = encode_image(args.image)
        if base64_image:
            # LangChain/Ollama generic image format
            # Note: The exact format might depend on the specific ChatOllama implementation version,
            # but usually passing 'image_url' with a data URI or just base64 works for many.
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        else:
            print("Failed to load image. Proceeding with text only.")

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": message_content,
                }
            ]
        },
        config
    )

    # Print the conversation
    for message in result["messages"]:
        if hasattr(message, 'pretty_print'):
            message.pretty_print()
        else:
            print(f"\n{message.type.upper()}: {message.content}")