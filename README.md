<h1 align="center">
  <br>
  ⚡ Entropy Drafting ⚡
  <br>
</h1>

This is the code repository for the bachelor thesis "Effiziente Text-Generierung mit spekulativem Decoding".

The goal of this thesis is to further improve the efficiency of speculative decoding by using an entropy-based speculation length (or candidate length). This allows the draft model to generate tokens as long as it is confident in the prediction before the candidate is handed over to the target model for validation. Three threshold strategies were developed and evaluated. The implementation is based on the Hugging Face Transformers library.

For more information and the full thesis, please refer to: TBD

## Generate with Entropy Drafting

### Install requirements

```pip install -q -r requirements.txt```

Note: This dynamically overwrites some classes in the Transformers Library and might not work with future versions. Enforce ```transformers==4.38.2``` in requirements.txt for best compatibility. 

### Generate with Entropy Drafting

Note 1: This is a simplified example only for demo purposes. Because of the large overhead and the small sample size ($n=1$) it is not suitable for evaluation purposes. It might be necessary to adjust hyperparameters. The ones used by default were optimal for model configuration ($m_t$ = OPT-2.7B, $m_d$ = OPT-125M).

Note 2: The first run might take a while since the models have to be downloaded.


| Options                  | Description                             | Default Value                           |
|--------------------------|-----------------------------------------|-----------------------------------------|
| --target                 | Target Model |                   |
| --draft                  | Draft Model  |                                    |
| --input                  | Input text  |                                       |
| --max_new_tokens         | How many tokens to generate    | 100                                   |
| --entropy_drafting       | Include flag to enable Entropy Drafting  |                                     |
| --entropy_drafting_type  | Threshold Strategy(```static```, ```ma``` (moving average) or ```cum``` (cumulative static)) | static |


### Example

```python entropy_drafting.py --target "facebook/opt-2.7b" --draft "facebook/opt-125m" --input "Towering over Kyiv for six decades, Hotel Ukraine has witnessed" --max_new_tokens 100 --entropy_drafting --entropy_drafting_type "static"```

