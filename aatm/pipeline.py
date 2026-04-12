"""
Define a minimal base class for composable pipeline components.

This module provides a lightweight base class that enables pipeline-style
composition through the bitwise OR operator. It is intended for workflows where
objects can be chained together and invoked in a uniform way, allowing each
component to pass data or itself to the next stage in the pipeline.
"""

from typing import Any


class PipelineBaseClass:
    """Provide a minimal interface for pipeline composition and invocation.

    This base class defines operator overloads that support chaining pipeline
    components with the ``|`` operator, as well as a default callable behavior
    that returns the input unchanged. Subclasses can override ``__call__`` to
    implement custom pipeline logic while preserving the composition interface.
    """

    def __or__(self, other: Any) -> Any:
        """Apply the right-hand pipeline component to this instance.

        This method enables left-to-right pipeline composition using the
        ``|`` operator. The right-hand operand is expected to be callable and
        able to accept this instance as input.

        Args:
            other: A callable pipeline component or compatible object to be
                applied to this instance.

        Returns:
            The result of calling ``other`` with this instance.

        Raises:
            TypeError: If ``other`` is not callable or cannot accept this
                instance as input.
        """

        return other(self)

    def __ror__(self, other: Any) -> Any:
        """Apply this pipeline component to the left-hand operand.

        This method enables right-hand dispatch for the ``|`` operator when
        this object appears on the right side of a pipeline expression. The
        left-hand operand is passed as input to this instance.

        Args:
            other: The input value or upstream pipeline component to be passed
                to this instance.

        Returns:
            The result of calling this instance with ``other``.

        Raises:
            TypeError: If this instance is not callable with the provided input.
        """
        return self(other)

    def __call__(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        """Return the input unchanged.

        This default implementation acts as an identity operation, making the
        base class usable even before subclasses define custom behavior.

        Args:
            input: The value to pass through the pipeline unchanged.
            *args: Additional positional arguments accepted for interface
                compatibility.
            **kwargs: Additional keyword arguments accepted for interface
                compatibility.

        Returns:
            The original input value, unchanged.
        """
        return input
