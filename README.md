# Lukas Widmers repo for postprocessing files used in PARTIES

## Usage guide

### running on linux home computer
1. source the correct venv
```zsh
source ./python_environment/venv_<name>/bin/activate
```
2. Change root level `main.py` to use the correct scripts and modify the scripts to use the correct directories
3. Run the main.py from the root with the `PYTHONPATH` enviroment variable set to the root

```zsh
PYTHONPATH=. python main.py
```

### running on anvil
<!-- TODO :-->

### Function return tpyes, dictionaries and their keys
<!-- TODO :-->

## python venv
- all venv directories must be in the root directory as to be properly git ignored.
### using venvs
- to activate on linux use
```zsh
source ./python_environment/venv_<name>/bin/activate
```

### creat new venv
- create requirements file in root directory `requirements_<name>.txt`
```txt
numpy
scipy
h5py
matplotlib
```
- linux
```zsh
cd ./python_environment
python3 venv venv_<type>
source ./venv_<type>/bin/activate
pip install -r ./pip_requirements/requirements_default.txt
pip install -r ./pip_requirements/requirements_torch_<type>.txt
```

- anvil (**!!!I think only conda works!!!**)
```zsh
cd ./python_environment
python3 venv venv_anvil
source ./venv_anvil/bin/activate
pip install -r ./pip_requirements/requirements_anvil.txt
```

## conda
<!-- TODO :-->

