FROM public.ecr.aws/lambda/python:3.11

WORKDIR /var/task

RUN pip install --no-cache-dir fastapi>=0.110 mangum>=0.17

# Framework serving layer
COPY kitchen/serve/ .

# Project predictor — projects ADD this layer in their own Dockerfile:
#   FROM <this-image>
#   COPY predictor.py .

CMD ["app.handler"]
