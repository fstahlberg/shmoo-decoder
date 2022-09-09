# The Shmoo Decoder

The Shmoo Decoder is the result of an [MT Marathon 2022](https://ufal.mff.cuni.cz/mtm22/) project on decoding algorithms
for machine translation. Shmoo aims to provide standard reference implementations of popular decoding algorithms that
are independent of the underlying NMT toolkit. The goal is to make decoding more comparable across toolkits, and to
extend the suite of decoding algorithms that are supported by existing NMT frameworks. Shmoo is a research-focused
framework, and thus prefers modularity and extensibility over runtime performance.

## Examples

### English-French translation with fairseq

This replicates
the [Getting Started example in fairseq](https://fairseq.readthedocs.io/en/latest/getting_started.html#evaluating-pre-trained-models):

```commandline
python3 run.py --config_path=example_configs/translate_fairseq_enfr.yaml --single_sentence='Why is it rare to discover new marine mammal species?' 
```

You will need to download the English-French WMT'14 checkpoint as described in the fairseq documentation.

## Software design

Shmoo's software design is inspired by the [SGNMT decoder](https://ucam-smt.github.io/sgnmt/html/): *Predictors* are
scoring modules that define the search space. The *Decoder* searches for high-scoring hypotheses in the space spanned by
the predictors. Inputs and outputs are represented as Python dictionaries that map feature names to arbitrary values.
The main entry point to Shmoo is `shmoo.core.api.Shmoo`. After setting up the API with `Shmoo.set_up()` you can
call `Shmoo.decode()` which implements the following main workflow:

1) Create an initial input feature dictionary mapping the special key name `input` to an `OrderedDict` mapping `raw` to
   the source sentence in plain text.
2) Run a sequence of `Preprocessor`s that read the last item in the `input` feature, modify it, and add it to
   the `input` `OrderedDict`. At the end, the final item in the `input` feature must contain the input token ID
   sequence.
3) Run the `Decoder` that takes the input feature dictionary and generates output feature dictionaries (one dict per
   returned hypothesis) using a daisy-chain of `Predictor`s for scoring. The output features are merged with the input
   feature dictionary.
4) Run a sequence of `Postprocessor`s separately on each feature dictionary returned by the decoder.

For example, the concrete workflow for replicating the
English-French [fairseq example](https://fairseq.readthedocs.io/en/latest/getting_started.html#evaluating-pre-trained-models)
in Shmoo is as follows:

1) Create the input feature
   dictionary `{'input': OrderedDict({'raw': 'Why is it rare to discover new marine mammal species?'})}`.
2) Run the `FairseqTokenizerPreprocessor` to Moses-tokenize the input, and the `FairseqBPEPreprocessor` to convert to
   BPE token IDs. Afterwards,
3) Run the `BeamDecoder` to generate BPE output token IDs.
4) Run the `FairseqBPEPostprocessor` and the `FairseqTokenizerPostprocessor` to convert the output BPE token IDs to the
   final Moses-detokenized target sentence.

## Reference

### Preprocessors

* `TrivialTokenPreprocessor`: Converts an input strings with token IDs such as `'1 2 3 4'` to an integer sequence such
  as `[1, 2, 3, 4]`.
* `FairseqTokenizerPreprocessor`: Applies a fairseq tokenizer such as `moses` to the `input` feature.
* `FairseqBPEPreprocessor`: Converts the string in `input` to integer IDs by applying a fairseq BPE model.
* `SPMPreprocessor`: Encodes the `input` feature with a sentencepiece model.

### Postprocessors

* `TrivialTokenPostprocessor`: Converts the integer sequence in `output` to a string.
* `FairseqTokenizerPostprocessor`: Runs a fairseq tokenizer such as `moses` to detokenize the string in the `output`
  feature.
* `FairseqBPEPostprocessor`: Converts the BPE integer ID sequence in `output` back to a string by applying a fairseq
  BPE model.
* `SPMPostprocessor`: Decodes the `output` feature with a sentencepiece model.
* `RemoveEOSPostprocessor`: Removes the last token in the `output` feature.
* `StdoutPostprocessor`: Writes the input and output features to stdout.

### Predictors

* `Fairseq`: Uses a left-to-right fairseq model for scoring.
* `TokenBoost`: Boosts the score of a particular token at each time step. For example, this can be used to boost/discount the end-of-sentence symbol.
* `LengthNorm`: Implements length normalization following [Wu et al. (2016)](https://arxiv.org/abs/1609.08144).
* `ScoreRecorder`: Adds an additional output feature called `partial_scores` that contains the partial hypothesis scores for each time step.

### Decoders

* `GreedyDecoder`: Greedy decoding (like beam decoding with a beam size of 1).
* `BeamDecoder`: Standard beam decoding
* `SamplingDecoder`: Implements a range of sampling strategies such as ancestral, top-k, nucleus, and typical sampling.