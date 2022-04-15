##  train-and-deploy-ocr-model-on-amazon-sagemaker

**Onboard PaddleOCR with SageMaker Projects for MLOps, illustration with optical character recognition on identity documents**

# Quick Start 

run train_and_deploy/notebook.ipynb

# code repo organisation (proposition) 

* experiments-sagemaker-studio (container will be built through CICD)
* experiments-notebooks (container will be built with docker build, test with local mode)
* image-build-train
* image-build-deploy
* sagemaker-pipeline-workflow

## Features

- [x] **SageMaker Training/Deploy BYOC Mode Support**
- [x] **BYOC training support integrate with Amazon SageMaker training hyper-parameters**
- [x] **convert models from training for inference**
- [x] **BYOC inference support S3ModelURL**
- [ ] **onnx speed up**
- [ ] **integrate with Amazon SageMaker pipeline**


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
