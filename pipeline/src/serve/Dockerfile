FROM public.ecr.aws/lambda/python:3.11

WORKDIR /var/task

RUN pip install --no-cache-dir fastapi>=0.110 mangum>=0.17

COPY src/serve/ .

CMD ["app.handler"]
