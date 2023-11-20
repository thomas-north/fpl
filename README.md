# fpl
A neural network to play FPL. Data collected from FPL and Understat using resources developed by https://github.com/vaastav/Fantasy-Premier-League.

# usage

Train your own predictive model using the `main_nn.py` script. Clone this repository:

```
git clone https://github.com/thomas-north/fpl.git
```

Navigate to `src/` folder, then run:

```
python player_nn.py
```

This script will store a saved version of the model as `model/trained_player_model.pth`. Try editing the parameters stored in the `config` dictionary of the `main()` function in `src/main_nn.py` to fine-tune the model. It is currently defaulted to examine an FPL favourite: Mohamed Salah. 

You can install all necessary dependencies using pip, run:

```
pip install -r requirements.txt
```
