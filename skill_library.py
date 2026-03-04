import os
import yaml
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

# --- Domain Layer ---

@dataclass
class Skill:
    """Domain model representing a Skill."""
    name: str
    description: str
    path: str  # Absolute path to the skill directory
    # Potentially other metadata like version, author etc.

class SkillRepository(ABC):
    """Interface for accessing skills."""
    
    @abstractmethod
    def get_all_skills(self) -> List[Skill]:
        """Retrieve all available skills."""
        pass
    
    @abstractmethod
    def get_skill_overview(self, skill_name: str) -> Optional[str]:
        """Retrieve the overview (content of SKILL.md) for a skill."""
        pass
    
    @abstractmethod
    def get_skill_details(self, skill_name: str, file_path: str) -> Optional[str]:
        """Retrieve specific file content within a skill directory."""
        pass
    
    @abstractmethod
    def list_skill_files(self, skill_name: str) -> List[str]:
        """List files available in a skill directory."""
        pass

# --- Infrastructure Layer ---

class FileSystemSkillRepository(SkillRepository):
    """File-system based implementation of SkillRepository."""
    
    def __init__(self, root_directory: str):
        self.root_directory = os.path.abspath(root_directory)
        self._cache_skills: Optional[List[Skill]] = None

    def get_all_skills(self) -> List[Skill]:
        if self._cache_skills is not None:
            return self._cache_skills
            
        skills = []
        if not os.path.exists(self.root_directory):
            print(f"Warning: Skills directory not found at {self.root_directory}")
            return []

        for item in os.listdir(self.root_directory):
            item_path = os.path.join(self.root_directory, item)
            if os.path.isdir(item_path):
                skill_md_path = os.path.join(item_path, "SKILL.md")
                if os.path.exists(skill_md_path):
                    try:
                        name, description = self._parse_frontmatter(skill_md_path)
                        # Fallback if name is not in frontmatter
                        if not name:
                            name = item 
                        
                        skills.append(Skill(
                            name=name,
                            description=description or "No description provided.",
                            path=item_path
                        ))
                    except Exception as e:
                        print(f"Error loading skill at {item_path}: {e}")
        
        self._cache_skills = skills
        return skills

    def _parse_frontmatter(self, file_path: str) -> tuple[Optional[str], Optional[str]]:
        """Parses simple YAML frontmatter from a markdown file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if content.startswith('---'):
            try:
                # Find the second '---'
                end_idx = content.find('---', 3)
                if end_idx != -1:
                    frontmatter_str = content[3:end_idx]
                    data = yaml.safe_load(frontmatter_str)
                    return data.get('name'), data.get('description')
            except Exception as e:
                print(f"Failed to parse frontmatter for {file_path}: {e}")
        
        return None, None

    def _find_skill_by_name(self, skill_name: str) -> Optional[Skill]:
        skills = self.get_all_skills()
        for skill in skills:
            if skill.name == skill_name:
                return skill
        
        # Try finding by directory name if display name doesn't match
        # This is important if user asks for "form_ocr_skill" but name is "Form OCR Extraction"
        # Ideally we map both, but for now let's prioritize exact match on name, then fallback?
        # A better approach is to rely on the Repo to have a map.
        # Let's search by directory name (basename) as well.
        for skill in skills:
            if os.path.basename(skill.path) == skill_name:
                 return skill
                 
        return None

    def get_skill_overview(self, skill_name: str) -> Optional[str]:
        skill = self._find_skill_by_name(skill_name)
        if not skill:
            return None
            
        try:
            with open(os.path.join(skill.path, "SKILL.md"), 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading SKILL.md for {skill_name}: {e}")
            return None

    def get_skill_details(self, skill_name: str, file_path: str) -> Optional[str]:
        skill = self._find_skill_by_name(skill_name)
        if not skill:
            return None
        
        # Sanitize path to prevent directory traversal
        # We want to allow reading files inside the skill directory relative to it.
        # file_path should be relative to skill root.
        
        target_path = os.path.join(skill.path, file_path)
        common_prefix = os.path.commonpath([os.path.abspath(skill.path), os.path.abspath(target_path)])
        if common_prefix != os.path.abspath(skill.path):
            return f"Error: Access denied. Path {file_path} is outside skill directory."
            
        if not os.path.exists(target_path):
             return f"Error: File {file_path} not found in skill {skill_name}."
             
        if os.path.isdir(target_path):
             return f"Error: {file_path} is a directory."

        try:
            with open(target_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file {file_path}: {e}"
            
    def list_skill_files(self, skill_name: str) -> List[str]:
        skill = self._find_skill_by_name(skill_name)
        if not skill:
            return []
            
        files_list = []
        for root, dirs, files in os.walk(skill.path):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, skill.path)
                files_list.append(rel_path)
        return sorted(files_list)

# --- Test ---
if __name__ == "__main__":
    # Point this to where the user said skills are: /home/ubuntu/ocr_test/skills
    repo = FileSystemSkillRepository("/home/ubuntu/ocr_test/skills")
    print("Loading skills...")
    skills = repo.get_all_skills()
    for s in skills:
        print(f"- {s.name} ({os.path.basename(s.path)}): {s.description[:50]}...")
    
    if skills:
        print("\nTest reading overview of first skill:")
        print(repo.get_skill_overview(skills[0].name)[:100] + "...")
        
        # files = repo.list_skill_files(skills[0].name)
        # print(f"\nFiles in {skills[0].name}: {files}")
