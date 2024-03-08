# VotaInformado Neural Network
Neural Network API for the [Vota Informado](https://votainformado.com.ar) project.

This Neural Network tries to predict the vote of a legislator on a particular project based on it's voting history.
 

<div style="text-align: center;">
    <img src="https://www.votainformado.com.ar/static/media/logo.b3817a447af529aca95c3d065b7c48e1.svg" alt="Vota Informado" width="200"/>
</div>


## Development

### Requirements
To run the project you need to have installed:
 - Docker
 - Docker Compose
 
### Running the project
To run the project you need to execute the following command:

```bash
# Build
docker compose -f local.yml build --build-arg CACHEBUST=$(openssl rand -base64 32)
# Run
docker compose -f local.yml run --rm --service-ports django.recoleccion
```
In order to get into the container and run commands, you can use the following command:

```bash
docker exec --user root -it $container_name bash
```
Where `$container_name` is the name of the container.

### Running the tests
Having the project running, connect to the container and run the following command:

```python
python manage.py test
```

### Fit and Train the model
#### Pre-requisites
In order to fir or train the model, first you must have running the [Recoleccion API](https://github.com/VotaInformado/recoleccion) and have run the neccessary commands collect the needed data.  
After that, run the project and connect to the container.

#### Training (recreating) the model
In order to **train** the model, you must run the following command from within the container:

```python
python manage.py train_neural_network
```
This command will **recreate** the model and train it with **all** the data collected. This process may take a while.

#### Fitting the model
In order to **fit** the model, you must run the following command from within the container:

```python
python manage.py fit_neural_network
```
This command will **fit** the model with the **new** data collected, without recreating the model. This process may take a while (but far less than training).




