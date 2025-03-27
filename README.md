First start by cloning this repository.

In order to construct the midi dataset, download the EMOPIA dataset from here:
https://zenodo.org/record/5090631#.YPPo-JMzZz8

Then unzip the file:
``` unzip EMOPIA_1.0.zip ```
now the midi files should be under a folder called EMOPIA_1.0.

The ```story_midi_matched.csv``` dataframe contains stories with their corresponding midi_id based on emotions. The dataset object reads the midi_ids for each story and reads the midi file with that ID and tokenizes it and stores it in the dataset object.

Once the data is downloaded, to train the model, run:

``` python train.py --model_name "bert-base-uncased" ```
