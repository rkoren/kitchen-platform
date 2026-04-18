"""Generator registry and dispatch."""
from recipes.schema import ResourceSpec


def generate_resource(spec: ResourceSpec) -> str:
    """Dispatch to the appropriate generator based on resource type."""
    from recipes.generators import iam, lambda_, s3

    registry = {
        "s3": s3.generate,
        "iam_role": iam.generate,
        "lambda": lambda_.generate,
    }
    fn = registry.get(spec.type)
    if fn is None:
        raise ValueError(f"No generator registered for resource type: {spec.type!r}")
    return fn(spec)
