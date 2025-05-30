# psysafe/catalog/registry.py
from typing import Dict, List, Type, Union, Any

from psysafe.core.base import GuardrailBase
from psysafe.core.composite import CompositeGuardrail # Needed for the compose method


class GuardrailCatalog:
    """
    A registry for discovering and loading guardrail implementations.
    Guardrails can be registered by name and then loaded or composed.
    """
    _registry: Dict[str, Type[GuardrailBase]] = {}

    @classmethod
    def register(cls, name: str, guardrail_cls: Type[GuardrailBase]) -> None:
        """
        Registers a guardrail class with a given name.

        Args:
            name: The unique name to register the guardrail under.
            guardrail_cls: The guardrail class (must be a subclass of GuardrailBase).

        Raises:
            TypeError: If guardrail_cls is not a subclass of GuardrailBase.
            ValueError: If the name is already registered.
        """
        if not issubclass(guardrail_cls, GuardrailBase):
            raise TypeError(f"Cannot register {guardrail_cls.__name__}: must be a subclass of GuardrailBase.")
        if name in cls._registry:
            # Potentially allow overwriting with a warning, or raise error.
            # For now, strict: no overwriting.
            raise ValueError(f"Guardrail name '{name}' is already registered with {cls._registry[name].__name__}.")
        cls._registry[name] = guardrail_cls

    @classmethod
    def load(cls, names: Union[str, List[str]], **kwargs: Any) -> List[GuardrailBase]:
        """
        Loads one or more guardrails by name.
        Any provided kwargs will be passed to the constructor of each loaded guardrail.

        Args:
            names: A single name or a list of names of guardrails to load.
            **kwargs: Keyword arguments to pass to the guardrail constructors.

        Returns:
            A list of instantiated guardrail objects.

        Raises:
            ValueError: If any requested name is not found in the registry.
        """
        if isinstance(names, str):
            names_list = [names]
        elif isinstance(names, list):
            names_list = names
        else:
            raise TypeError("Input 'names' must be a string or a list of strings.")

        loaded_guardrails: List[GuardrailBase] = []
        for name in names_list:
            if name not in cls._registry:
                raise ValueError(f"Unknown guardrail: '{name}'. Available: {list(cls._registry.keys())}")
            
            guardrail_cls = cls._registry[name]
            try:
                # Instantiate the guardrail, passing any provided kwargs
                guardrail_instance = guardrail_cls(**kwargs)
                loaded_guardrails.append(guardrail_instance)
            except TypeError as e:
                # Catch errors if kwargs don't match the guardrail's __init__ signature
                raise TypeError(
                    f"Error instantiating guardrail '{name}' ({guardrail_cls.__name__}): {e}. "
                    f"Ensure provided kwargs match constructor parameters."
                ) from e
        return loaded_guardrails

    @classmethod
    def compose(cls, names: Union[str, List[str]], **kwargs: Any) -> CompositeGuardrail:
        """
        Loads multiple guardrails by name and returns them wrapped in a CompositeGuardrail.
        Any provided kwargs will be passed to the constructor of each loaded guardrail.

        Args:
            names: A single name or a list of names of guardrails to compose.
            **kwargs: Keyword arguments to pass to the individual guardrail constructors.

        Returns:
            A CompositeGuardrail instance containing the loaded guardrails.
        """
        guardrails_to_compose = cls.load(names, **kwargs)
        if not guardrails_to_compose:
            # This case should ideally be caught by load if names is empty or all fail,
            # but as a safeguard for CompositeGuardrail's own requirement.
            raise ValueError("Cannot compose an empty list of guardrails.")
        return CompositeGuardrail(guardrails=guardrails_to_compose)

    @classmethod
    def list_available(cls) -> List[str]:
        """Returns a list of names of all registered guardrails."""
        return list(cls._registry.keys())

    @classmethod
    def get_guardrail_class(cls, name: str) -> Type[GuardrailBase]:
        """
        Retrieves the class for a registered guardrail.

        Args:
            name: The name of the guardrail.

        Returns:
            The guardrail class.

        Raises:
            ValueError: If the name is not found.
        """
        if name not in cls._registry:
            raise ValueError(f"Unknown guardrail: '{name}'. Available: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def clear_registry(cls) -> None:
        """Clears all registered guardrails. Useful for testing."""
        cls._registry.clear()