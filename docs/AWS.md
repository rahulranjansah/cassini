# AWS Setup and Usage

## Prerequisites

- AWS CLI installed and configured
- S3 bucket with public access or appropriate permissions

## Downloading Files from S3

To download a file from S3, use the following commands:

### Using `wget` or `curl`

```bash
wget https://astavak.s3.eu-north-1.amazonaws.com/home/astavak/cassini/data/level1/CDA__CAT_IID_cal_data.pkl -O CDA__CAT_IID_cal_data.pkl

curl -o CDA__CAT_IID_cal_data.pkl https://astavak.s3.eu-north-1.amazonaws.com/home/astavak/cassini/data/level1/CDA__CAT_IID_cal_data.pkl

```

### S3 Bucket Setup

```
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadGetObject",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::astavak/*"
    }
  ]
}

```
