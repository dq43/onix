#Serverless worker for Onix model v1 

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /

RUN pip install --no-cache-dir \
    "transformers==4.49.0" \
    "peft==0.14.0" \
    "bitsandbytes==0.45.0" \
    "accelerate==1.3.0" \
    "runpod>=1.6.0"



COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]


