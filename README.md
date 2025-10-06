# Lukas Widmers repo for postprocessing files used in PARTIES

## python venv
- all venv directories must be in the root directory as to be properly git ignored.
### using venvs
- to activate on linux use
```zsh
source ./venv_<name>/bin/activate
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
python3 venv venv_<type>
source ./venv_<type>/bin/activate
pip install -r ./pip_requirements/requirements_default.txt
pip install -r ./pip_requirements/requirements_torch_<type>.txt
```

- anvil (**!!!I think only conda works!!!**)
```zsh
python3 venv venv_anvil
source ./venv_anvil/bin/activate
pip install -r ./pip_requirements/requirements_anvil.txt
```

## conda
- TODO

## running on anvil
- TODO
