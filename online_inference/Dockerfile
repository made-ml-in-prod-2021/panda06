FROM python:3.8

COPY dist/ml_project-0.1.0.tar.gz /ml_project-0.1.0.tar.gz
COPY online_inference/ /online_inference
RUN pip install ml_project-0.1.0.tar.gz

WORKDIR /online_inference/src
RUN pip install -r ../requirements.txt

ENV PATH_TO_MODEL=../model.pkl
ENV PATH_TO_EXTRACTOR=../extractor.pkl
ENV PYTHONPATH "${PYTHONPATH}:.."

CMD ["python", "app.py"]