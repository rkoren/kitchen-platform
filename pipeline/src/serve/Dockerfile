FROM public.ecr.aws/lambda/python:3.11

WORKDIR /var/task

COPY ../../pyproject.toml .
RUN pip install --no-cache-dir ".[serve]" --target /var/task

COPY . .

CMD ["src.serve.app.handler"]
