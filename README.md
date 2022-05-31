#  train-and-deploy-ocr-model-on-amazon-sagemaker

**Onboard PaddleOCR with SageMaker Projects for MLOps, illustration with optical character recognition on identity documents**

# Quick Start 

Run train_and_deploy/notebook.ipynb on Amazon SageMaker Notebook Instances

Or follow step-by-step guidance on Amazon SageMaker Studio Notebooks with our blog post (to be annoucned)

# Code Structure

* experiments-sagemaker-studio (SageMaker training and deployment SDK)
* train_and_deploy (Quick start notebook on SageMaker notebook instances)
* image-build-process (data generation container with CodeBuild template for CICD)
* image-build-train (training container with CodeBuild template for CICD )
* image-build-deploy (inference container with CodeBuild template for CICD)
* sagemaker_pipelines (SageMaker Pipelines for model build workflow)

## Features

![alt text](./dsOnboarding.png)


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

