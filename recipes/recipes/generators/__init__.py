"""Generator registry and dispatch."""
from recipes.schema import ResourceSpec


def generate_resource(spec: ResourceSpec, all_resources: list = None) -> str:
    """Dispatch to the appropriate generator based on resource type."""
    from recipes.generators import ecr, iam, lambda_, s3

    if spec.type == "lambda":
        return lambda_.generate(spec, all_resources or [])

    registry = {
        "s3": s3.generate,
        "iam_role": iam.generate,
        "ecr": ecr.generate,
    }
    fn = registry.get(spec.type)
    if fn is None:
        raise ValueError(f"No generator registered for resource type: {spec.type!r}")
    return fn(spec)
