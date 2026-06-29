FROM public.ecr.aws/lambda/python:3.11

WORKDIR /var/task

RUN pip install --no-cache-dir fastapi>=0.110 mangum>=0.17

# Framework serving layer — copied as an importable ``kitchen.serve`` package so the app's
# absolute imports (``from kitchen.serve.loader import …``) resolve at runtime. (Copying
# serve/ *flat* into /var/task builds fine but fails to import the handler at container start —
# "No module named 'kitchen'"; the build-only CI never ran the container, so it was masked.)
COPY kitchen/serve/ ./kitchen/serve/
RUN touch ./kitchen/__init__.py

# Project predictor — projects ADD this layer in their own Dockerfile:
#   FROM <this-image>
#   COPY predictor.py .       # resolved from /var/task at startup
# The handler is the Mangum adapter exposed by the framework serving app.
CMD ["kitchen.serve.app.handler"]
