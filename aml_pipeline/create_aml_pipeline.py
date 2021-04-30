
import os
import azureml.core
from azureml.core import Workspace
from azureml.core import Dataset
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import PipelineData
from azureml.pipeline.steps import PythonScriptStep
from azureml.core import Experiment
from azureml.pipeline.core import Pipeline
from azureml.core import Model
import logging
import sys
from aml_pipeline.env_variables import Env

def main():
    e = Env()
    
    print('********************')
    print(e.source_directory)
    
    
    files = os.listdir('./aml_pipeline')
    for f in files:
        print(f)
    
    print('***************')
    
    workspace_name = e.workspace_name
    subscription_id = e.subscription_id
    resource_group = e.resource_group
    
    #Connect to AML Workspace
    print('workspace_name = ' + workspace_name)
    print('subscription_id = ' + subscription_id)
    print('resource_group = ' + resource_group)
    
    
    ws = Workspace.get(
        name= workspace_name,
        subscription_id = subscription_id,
        resource_group= resource_group,
    )
    
    print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))


    default_ds = ws.get_default_datastore()

    if 'diabetes dataset' not in ws.datasets:
        default_ds.upload_files(files=['diabetes.csv', 'diabetes2.csv'], # Upload the diabetes csv files in /data
                            target_path='diabetes-data/', # Put it in a folder path in the datastore
                            overwrite=True, # Replace existing files of the same name
                            show_progress=True)

        #Create a tabular dataset from the path on the datastore (this may take a short while)
        tab_data_set = Dataset.Tabular.from_delimited_files(path=(default_ds, 'diabetes-data/*.csv'))

        # Register the tabular dataset
        try:
            tab_data_set = tab_data_set.register(workspace=ws, 
                                    name='diabetes dataset',
                                    description='diabetes data',
                                    tags = {'format':'CSV'},
                                    create_new_version=True)
            print('Dataset registered.')
        except Exception as ex:
            print(ex)
    else:
        print('Dataset already registered.')






    # Create a folder for the pipeline step files
    experiment_folder = 'diabetes_pipeline'
    os.makedirs(experiment_folder, exist_ok=True)

    print(experiment_folder)








    cluster_name = "mmcomputecluster"

    try:
        # Check for existing compute target
        pipeline_cluster = ComputeTarget(workspace=ws, name=cluster_name)
        print('Found existing cluster, use it.')
    except ComputeTargetException:
        # If it doesn't already exist, create it
        try:
            compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS11_V2', max_nodes=2)
            pipeline_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
            pipeline_cluster.wait_for_completion(show_output=True)
        except Exception as ex:
            print(ex)
        




    # Create a Python environment for the experiment
    diabetes_env = Environment("diabetes-pipeline-env")
    diabetes_env.python.user_managed_dependencies = False # Let Azure ML manage dependencies
    diabetes_env.docker.enabled = True # Use a docker container

    # Create a set of package dependencies
    diabetes_packages = CondaDependencies.create(conda_packages=['scikit-learn','ipykernel','matplotlib','pandas','pip'],
                                                 pip_packages=['azureml-defaults','azureml-dataprep[pandas]','pyarrow'])

    # Add the dependencies to the environment
    diabetes_env.python.conda_dependencies = diabetes_packages

    # Register the environment 
    diabetes_env.register(workspace=ws)
    registered_env = Environment.get(ws, 'diabetes-pipeline-env')

    # Create a new runconfig object for the pipeline
    pipeline_run_config = RunConfiguration()

    # Use the compute you created above. 
    pipeline_run_config.target = pipeline_cluster

    # Assign the environment to the run configuration
    pipeline_run_config.environment = registered_env

    print ("Run configuration created.")







    # Get the training dataset
    diabetes_ds = ws.datasets.get("diabetes dataset")

    # Create a PipelineData (temporary Data Reference) for the model folder
    prepped_data_folder = PipelineData("prepped_data_folder", datastore=ws.get_default_datastore())

    # Step 1, Run the data prep script
    prep_step = PythonScriptStep(name = "Prepare Data",
                                    script_name = "prep_diabetes.py",
                                    source_directory = e.source_directory + '/aml_pipeline',
                                    arguments = ['--input-data', diabetes_ds.as_named_input('raw_data'),
                                                 '--prepped-data', prepped_data_folder],
                                    outputs=[prepped_data_folder],
                                    compute_target = pipeline_cluster,
                                    runconfig = pipeline_run_config,
                                    allow_reuse = True)

    # Step 2, run the training script
    train_step = PythonScriptStep(name = "Train and Register Model",
                                    source_directory =  e.source_directory + '/aml_pipeline',
                                    script_name = "train_diabetes.py",
                                    arguments = ['--training-folder', prepped_data_folder],
                                    inputs=[prepped_data_folder],
                                    compute_target = pipeline_cluster,
                                    runconfig = pipeline_run_config,
                                    allow_reuse = True)

    print("Pipeline steps defined")



    pipeline_steps = [prep_step, train_step]
    pipeline = Pipeline(workspace=ws, steps=pipeline_steps)
    print("Pipeline is built.")

    # Create an experiment and run the pipeline
    experiment = Experiment(workspace=ws, name = 'jlg-exp')
    pipeline_run = experiment.submit(pipeline, regenerate_outputs=True)
    print("Pipeline submitted for execution.")
    RunDetails(pipeline_run).show()
    pipeline_run.wait_for_completion(show_output=True)



    for run in pipeline_run.get_children():
        print(run.name, ':')
        metrics = run.get_metrics()
        for metric_name in metrics:
            print('\t',metric_name, ":", metrics[metric_name])



    for model in Model.list(ws):
        print(model.name, 'version:', model.version)
        for tag_name in model.tags:
            tag = model.tags[tag_name]
            print ('\t',tag_name, ':', tag)
        for prop_name in model.properties:
            prop = model.properties[prop_name]
            print ('\t',prop_name, ':', prop)
        print('\n')



    # Publish the pipeline from the run
    published_pipeline = pipeline_run.publish_pipeline(
        name="diabetes-training-pipeline", description="Trains diabetes model", version="1.0")

    published_pipeline



    rest_endpoint = published_pipeline.endpoint
    print(rest_endpoint)

if __name__ == "__main__":
    main()
    
