FROM public.ecr.aws/lambda/python:3.11

COPY . .

# Create a directory for data
#RUN mkdir '/tmp/data/'
COPY ffmpeg '/usr/share/'
COPY ffmpeg '/usr/bin/'
COPY standard_scaler.pkl '/usr/share/'
COPY svm_model.pkl '/usr/share/'


# Copy requirements.txt and install the required packages
RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy app.py
COPY app.py ${LAMBDA_TASK_ROOT}

# Set the CMD for the Lambda function
CMD [ "app.handler" ]
