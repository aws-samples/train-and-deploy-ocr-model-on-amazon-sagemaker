#!/bin/bash
set -v
set -e
# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.
# here we need to specify aksk

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
image=$1

if [ "$image" == "" ]
then
    echo "Use image name paddle"
    image="paddle"
fi

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi

# Get the region defined in the current configuration
#if you only want to specify one region, use region, otherwise, use regions
region=$(aws configure get region)

if [[ $region =~ ^cn.* ]]
then
d
elif [[ $region = "ap-east-1" ]]
then
    fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"
    registry_id="871362719292"
    registry_uri="${registry_id}.dkr.ecr.${region}.amazonaws.com"
else
    fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"
    registry_id="763104351884"
    registry_uri="${registry_id}.dkr.ecr.${region}.amazonaws.com"
fi

echo ${fullname}

# Get the login command from ECR and execute it directly
$(aws ecr get-login --registry-ids ${account} --region ${region} --no-include-email)
$(aws ecr get-login --registry-ids ${registry_id} --region ${region} --no-include-email)

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${image}" --region ${region} || aws ecr create-repository --repository-name "${image}" --region ${region}


# Build the docker image, tag with full name and then push it to ECR
#docker build -t ${image} -f Dockerfile . --build-arg REGISTRY_URI=${registry_uri}
docker build -t ${image} -f Dockerfile . 
docker tag ${image} ${fullname}
docker push ${fullname}