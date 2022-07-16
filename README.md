# Single Aircraft Complexity
[![Open in Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)](https://open.vscode.dev/risufaj/Single-Aircraft-Complexity)
	![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


 Code to support the paper From SIngle Aircraft to Communities: A Neutral Interpretation of Air Traffic Complexity Dynamics
 
 There are two ways to use the code:
 
 ### Docker
 
 [Install Docker first](https://docs.docker.com/get-docker/)
 
 Build the container.

```shell
docker build -t complexity . 
```

Run the app.

```shell
docker run -t -I -p 9999:9999 complexity
```

If everything goes well, you should be able to access the app in the browser at ``` localhost:9999``` 

### From command line

Install requirements 

```shell
python -m pip install -r requirements.txt
```

Run the application

```shell
python app.py
```

The application accepts csv files in this format:
| time | aircraft_id | type | lat | lon |
|------|-------------|------|-----|-----|
So you need a log file that has these columns. We have used the application alongside [the BlueSky Simulator]([https://docs.docker.com/get-docker/](https://github.com/risufaj/bluesky)) (links to our fork, with reference to the main repository). However, in principle the application only needs a proper log file.
We plan on extending the allowed formats.
