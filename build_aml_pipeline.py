# Continuous Integration (CI) pipeline that orchestrates the training, evaluation, and registration of the diabetes_regression model.

resources:
  containers:
  - container: mlops
    image: mcr.microsoft.com/mlops/python:latest

pr: none
trigger:
  branches:
    include:
    - master
  paths:
    include:
    - aml_pipeline/


variables:
- group: devopsforai-aml-vg

pool:
  vmImage: ubuntu-latest

stages:
- stage: 'Model_CI'
  displayName: 'Model CI'
  jobs:
  - job: "Model_CI_Pipeline"
    displayName: "Model CI Pipeline"
    container: mlops
    timeoutInMinutes: 0
    steps:
    - template: code-quality-template.yml
    - task: AzureCLI@1
      inputs:
        azureSubscription: '$(WORKSPACE_SVC_CONNECTION)'
        scriptLocation: inlineScript
        workingDirectory: $(Build.SourcesDirectory)
        inlineScript: |
          set -e # fail on error
          export SUBSCRIPTION_ID=$(az account show --query id -o tsv)
          # Invoke the Python building and publishing a training pipeline
          python -m aml_pipeline.create_aml_pipeline
      displayName: 'Publish Azure Machine Learning Pipeline'