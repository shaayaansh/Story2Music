First start by cloning this repository.

In order to construct the midi dataset, download the EMOPIA dataset from here:
```
wget --show-progress https://zenodo.org/records/5090631/files/EMOPIA_1.0.zip
```
Then unzip the file:

run ``` unzip EMOPIA_1.0.zip ``` in the terminal.
now the midi files should be under a folder called EMOPIA_1.0.

The ```story_midi_matched.csv``` dataframe contains stories with their corresponding midi_id based on emotions. The dataset object reads the midi_ids for each story and reads the midi file with that ID and tokenizes it and stores it in the dataset object.

Create a python environment and install the dependencies:

```
pyenv virtualenv 3.13.0 story2music
pyenv activate
pip install -r requirements.txt
```

Once you are set up and the data is downloaded, to train the model, run:

``` python train.py --model_name "bert-base-uncased" ```

To play a midifile from command line:

```
python play_midi.py <path/to/midi/file.mid>
```

For example:
```
python play_midi.py EMOPIA_1.0/midis/Q1_0vLPYiPN7qY_0.mid
```