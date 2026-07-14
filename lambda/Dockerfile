# AWS's official Lambda base image - different from the plain python:3.12-slim
# used for the Docker/local-serving Dockerfile in the project root. This one
# includes the Lambda Runtime Interface Client Lambda needs to receive
# invocations, which a generic Python image doesn't have.
FROM public.ecr.aws/lambda/python:3.12

# Lambda expects application code under ${LAMBDA_TASK_ROOT} (= /var/task)
# Note: COPY paths are relative to the BUILD CONTEXT (the "." in
# "docker build -f lambda/Dockerfile -t ... ."), which is the project
# root - NOT relative to this Dockerfile's own location inside lambda/.
COPY lambda/requirements-lambda.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r requirements-lambda.txt

COPY app/main.py ${LAMBDA_TASK_ROOT}/app/main.py
COPY app/models ${LAMBDA_TASK_ROOT}/app/models
COPY lambda/lambda_handler.py ${LAMBDA_TASK_ROOT}/

# Lambda's CMD format is "filename.function_name", not a shell command -
# the runtime interface client invokes this directly per request
CMD ["lambda_handler.handler"]
