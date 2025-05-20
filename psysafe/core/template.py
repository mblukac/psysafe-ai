# psysafe/core/template.py
from pathlib import Path
from typing import Any, Dict, Union, Optional

# Assuming PromptRenderCtx will be imported from psysafe.core.models
from psysafe.core.models import PromptRenderCtx

# Jinja2 will be a dependency. For now, we assume it's installed.
# If not, this will raise an ImportError, which is fine for now.
try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
except ImportError:
    # Fallback or raise a more specific error if Jinja2 is a hard requirement at this stage
    # For now, let's make it clear that it's expected.
    raise ImportError("Jinja2 is required for PromptTemplate. Please install it.")


class PromptTemplate:
    """
    Manages prompt templates, allowing loading from strings or files,
    and rendering them with provided context using Jinja2.
    """
    def __init__(self, template_string: str, template_path: Optional[Path] = None):
        """
        Initializes a PromptTemplate.

        Args:
            template_string: The raw template string.
            template_path: Optional path to the original template file (for context/debugging).
        """
        self.template_string = template_string
        self.template_path = template_path
        # Initialize a Jinja environment.
        # For from_string, the loader isn't strictly necessary for the string itself,
        # but a consistent environment is good.
        # For from_file, a FileSystemLoader would be more appropriate if we were loading by name.
        # Here, we load content first, then create template.
        self.jinja_env = Environment(
            loader=FileSystemLoader("."), # Dummy loader, not used if template_string is directly used
            autoescape=False, # Disable autoescaping by default for prompts
            keep_trailing_newline=True,
            trim_blocks=True, # Useful for template readability
            lstrip_blocks=True # Useful for template readability
        )
        # Render None as empty string
        self.jinja_env.finalize = lambda x: x if x is not None else ''
        self._compiled_template = self.jinja_env.from_string(self.template_string)

    @classmethod
    def from_string(cls, prompt_text: str) -> "PromptTemplate":
        """
        Creates a PromptTemplate instance from a raw string.

        Args:
            prompt_text: The string containing the prompt template.

        Returns:
            A PromptTemplate instance.
        """
        return cls(template_string=prompt_text)

    @classmethod
    def from_file(cls, template_file_path: Union[str, Path]) -> "PromptTemplate":
        """
        Creates a PromptTemplate instance by loading content from a file.

        Args:
            template_file_path: Path to the template file (e.g., a .md file).

        Returns:
            A PromptTemplate instance.
        
        Raises:
            FileNotFoundError: If the template file does not exist.
        """
        path = Path(template_file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Template file not found: {path}")
        
        template_content = path.read_text(encoding="utf-8")
        return cls(template_string=template_content, template_path=path)

    def render(self, context: PromptRenderCtx) -> str:
        """
        Renders the prompt template with the given context.

        The context includes variables defined in PromptRenderCtx,
        such as driver_type, model_name, request_type, and a 'variables' dictionary.

        Args:
            context: A PromptRenderCtx object containing data for rendering.

        Returns:
            The rendered prompt string.
        """
        # Prepare the rendering context by flattening PromptRenderCtx attributes
        # and merging its 'variables' dict.
        render_vars = {
            "driver_type": context.driver_type,
            "model_name": context.model_name,
            "request_type": context.request_type,
            **(context.variables or {}) # Merge the variables from PromptRenderCtx
        }
        return self._compiled_template.render(render_vars)

    def __repr__(self) -> str:
        if self.template_path:
            return f"<PromptTemplate source='{self.template_path}'>"
        return f"<PromptTemplate source='(string)', length={len(self.template_string)}>"