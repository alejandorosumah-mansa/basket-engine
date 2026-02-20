"""Taxonomy loader and helpers for theme classification."""

import yaml
from pathlib import Path
from typing import Optional


TAXONOMY_PATH = Path(__file__).parent.parent.parent / "config" / "taxonomy.yaml"


def load_taxonomy(path: Optional[Path] = None) -> dict:
    """Load and validate taxonomy from YAML."""
    path = path or TAXONOMY_PATH
    with open(path) as f:
        data = yaml.safe_load(f)
    
    themes = data.get("themes", {})
    # Validate required fields
    for theme_id, theme in themes.items():
        assert "name" in theme, f"Theme {theme_id} missing 'name'"
        assert "description" in theme, f"Theme {theme_id} missing 'description'"
        assert "examples" in theme, f"Theme {theme_id} missing 'examples'"
        assert "anti_examples" in theme, f"Theme {theme_id} missing 'anti_examples'"
    
    return themes


def list_themes(themes: Optional[dict] = None) -> list[str]:
    """Return sorted list of theme IDs."""
    themes = themes or load_taxonomy()
    return sorted(themes.keys())


def get_theme_description(theme_id: str, themes: Optional[dict] = None) -> str:
    """Get full description for a theme."""
    themes = themes or load_taxonomy()
    return themes[theme_id]["description"]


def format_taxonomy_for_prompt(themes: Optional[dict] = None) -> str:
    """Format the full taxonomy for inclusion in LLM prompts."""
    themes = themes or load_taxonomy()
    lines = []
    for theme_id in sorted(themes.keys()):
        t = themes[theme_id]
        lines.append(f"## {theme_id}: {t['name']}")
        lines.append(f"Description: {t['description']}")
        lines.append("Examples:")
        for ex in t["examples"]:
            lines.append(f"  - {ex}")
        lines.append("Anti-examples (do NOT classify these here):")
        for ax in t["anti_examples"]:
            lines.append(f"  - {ax}")
        lines.append("")
    return "\n".join(lines)


def get_all_anti_examples(themes: Optional[dict] = None) -> list[dict]:
    """Get all anti-examples with their forbidden theme."""
    themes = themes or load_taxonomy()
    results = []
    for theme_id, t in themes.items():
        for ax in t["anti_examples"]:
            # Strip comments
            text = ax.split("#")[0].strip().strip('"').strip("'")
            results.append({"text": text, "forbidden_theme": theme_id})
    return results
