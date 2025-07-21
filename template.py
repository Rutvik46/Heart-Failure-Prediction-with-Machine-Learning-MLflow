import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] : %(message)s:')

Project_name = "mlProject"

list_of_files = [ 
    ".github/workflows/.gitkeep",
    f"src/{Project_name}/__init__.py",
    f"src/{Project_name}/components/__init__.py",
    f"src/{Project_name}/utils/__init__.py",
    f"src/{Project_name}/utils/common.py",
    f"src/{Project_name}/config/__init__.py",
    f"src/{Project_name}/config/configuration.py",
    f"src/{Project_name}/pipeline/__init__.py",
    f"src/{Project_name}/entity/__init__.py",
    f"src/{Project_name}/entity/config_entity.py",
    f"src/{Project_name}/constants/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    "main.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "templates/index.html",
    "test.py"

]


for filepath in list_of_files:

    # Convert "/" path into windows os compatible path
    filepath = Path(filepath)

    # Split the file path into directory and filename
    filedir, filename = os.path.split(filepath)

    # Create directory if it does not exist
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directoy: {filedir} for the file: {filename}")

    # Create an empty file if it does not exist or is empty
    if not os.path.exists(filename) or os.path.getsize(filename):
        with open(filepath, "w") as file:
            pass
            logging.info(f"Creating empty file: {filename}")

    # If the file already exists, log that information
    else:
        logging.info(f"{filename} already exists")  