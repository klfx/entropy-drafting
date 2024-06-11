### START OF .generation\candidate_generator.py IMPORTS (modified) ###

import copy
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch

"""
if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from .configuration_utils import GenerationConfig
    from .logits_process import LogitsProcessorList
"""


### END OF .generation\candidate_generator.py IMPORTS ###


from reporting_utils import DecodingMetricPoint, DecodingMetricsLogger
from transformers.generation.candidate_generator import AssistedCandidateGenerator, _crop_past_key_values, _prepare_attention_mask, _prepare_token_type_ids

class CustomAssistedCandidateGenerator(AssistedCandidateGenerator):

    def __init__(
        self,
        input_ids: torch.LongTensor,
        assistant_model: "PreTrainedModel",
        generation_config: "GenerationConfig",
        logits_processor: "LogitsProcessorList",
        model_kwargs: Dict,
        inputs_tensor: Optional[torch.Tensor] = None,
        num_k: Optional[int] = None,
    ):
        # num_k specifies the number of candidates to generate (KLF)
        self._num_k = num_k

        # Make sure all data at the same device as assistant model
        device = assistant_model.device
        input_ids = input_ids.to(device)
        inputs_tensor = inputs_tensor.to(device)

        # Prepare the assistant and the starting number of candidate tokens
        self.assistant_model = assistant_model

        # Override num_assistant_tokens if num_k is specified
        if self._num_k:
            self.num_assistant_tokens = self._num_k
        else:
            self.num_assistant_tokens = assistant_model.generation_config.num_assistant_tokens

        # Prepare the kwargs for the assistant model
        assistant_kwargs = {}
        for key, value in model_kwargs.items():  # deepcopy crashes if we attempt to copy encoder outputs with grads
            if key not in ("encoder_outputs", "assistant_encoder_outputs"):
                assistant_kwargs[key] = (
                    value.detach().to(device) if isinstance(value, torch.Tensor) else copy.deepcopy(value)
                )

        if "assistant_encoder_outputs" in model_kwargs:
            assistant_kwargs["encoder_outputs"] = model_kwargs["assistant_encoder_outputs"]
        elif assistant_model.config.is_encoder_decoder:
            inputs_tensor, model_input_name, assistant_kwargs = assistant_model._prepare_model_inputs(
                inputs_tensor, assistant_model.generation_config.bos_token_id, assistant_kwargs
            )
            assistant_kwargs = assistant_model._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, assistant_kwargs, model_input_name
            )
        elif "encoder_outputs" in model_kwargs:
            assistant_kwargs["encoder_outputs"] = model_kwargs["encoder_outputs"]
        self.assistant_kwargs = assistant_kwargs

        # Prepare assistant model's keys of inputs
        if assistant_model.config.is_encoder_decoder:
            # both are encoder-decoder
            self.input_ids_key = "decoder_input_ids"
            self.attention_key = "decoder_attention_mask"
        elif "encoder_outputs" in assistant_kwargs:
            # special case for encoder-decoder with decoder-only assistant (like DistilWhisper)
            self.input_ids_key = "input_ids"
            self.attention_key = "attention_mask"
            self.assistant_kwargs["attention_mask"] = self.assistant_kwargs.get(
                "decoder_attention_mask",
                torch.ones((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.long),
            )
        else:
            # both are decoder-only
            self.input_ids_key = "input_ids"
            self.attention_key = "attention_mask"

        # Prepare generation-related options.
        eos_token_id = generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.eos_token_id_tensor = (
            torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        )
        self.logits_processor = logits_processor
        self.generation_config = copy.deepcopy(generation_config)
        self.generation_config.return_dict_in_generate = True
        self.generation_config.output_scores = True


    def get_candidates(self,
                       input_ids: torch.LongTensor,
                       early_termination: bool = False,
                       early_termination_parameters: dict = None
        ) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        """
        Fetches the candidates to be tried for the current input.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

        Return:
            `torch.LongTensor` of shape `(batch_size, candidate_length)` containing the candidate sequences to be
            assessed by the model and a `torch.FloatTensor` of shape `(batch_size, candidate_length,
            vocabulary_size)` containing the logits associated to each candidate.
        """

        input_ids = input_ids.to(self.assistant_model.device)

        # Don't generate more than `max_length - 1` candidates since the target model generates one extra token.
        new_cur_len = input_ids.shape[-1]
        max_new_tokens = min(int(self.num_assistant_tokens), self.generation_config.max_length - new_cur_len - 1)
        if max_new_tokens == 0:
            return input_ids, None
        
        # If flag is not set, just do the normal generation
        if not early_termination:
            # 1. If it is not the first round of candidate generation, prepare the inputs based on the input_ids length
            # (which implicitly contains the number of accepted candidates from the previous round)
            has_past_key_values = self.assistant_kwargs.get("past_key_values", None) is not None
            if has_past_key_values:
                new_cache_size = new_cur_len - 1
                self.assistant_kwargs["past_key_values"] = _crop_past_key_values(
                    self.assistant_model, self.assistant_kwargs["past_key_values"], new_cache_size - 1
                )  # the assistant does not have the token after the last match, hence the -1

                self.assistant_kwargs = _prepare_attention_mask(
                    self.assistant_kwargs, new_cur_len, self.assistant_model.config.is_encoder_decoder
                )
                self.assistant_kwargs = _prepare_token_type_ids(self.assistant_kwargs, new_cur_len)

            # 2. Forecast next N tokens using the assistant model.
            assistant_generation_kwargs = {
                self.input_ids_key: input_ids,
                "max_new_tokens": max_new_tokens,
                "generation_config": self.generation_config,
                "logits_processor": self.logits_processor,
            }

            assistant_output = self.assistant_model.generate(**assistant_generation_kwargs, **self.assistant_kwargs)

            # 3. Update variables for the next round of candidate generation
            self.assistant_kwargs["past_key_values"] = assistant_output.past_key_values

            # 4. Prepare variables for output
            candidate_logits = torch.stack(assistant_output.scores, dim=1)
            candidate_ids = assistant_output.sequences
            return candidate_ids, candidate_logits

        # If flag is set, do the early termination        
        draft_iteration = 1
        candidate_logits = []
        candidate_ids = torch.tensor([])
        while draft_iteration <= max_new_tokens: 
            # Update new_cur_len (len of current input_ids)(so the attention mask is correct)
            new_cur_len = input_ids.shape[-1]

            # 1. If it is not the first round of candidate generation, prepare the inputs based on the input_ids length
            # (which implicitly contains the number of accepted candidates from the previous round)
            has_past_key_values = self.assistant_kwargs.get("past_key_values", None) is not None
            if has_past_key_values:
                new_cache_size = new_cur_len - 1
                self.assistant_kwargs["past_key_values"] = _crop_past_key_values(
                    self.assistant_model, self.assistant_kwargs["past_key_values"], new_cache_size - 1
                )  # the assistant does not have the token after the last match, hence the -1

                self.assistant_kwargs = _prepare_attention_mask(
                    self.assistant_kwargs, new_cur_len, self.assistant_model.config.is_encoder_decoder
                )
                self.assistant_kwargs = _prepare_token_type_ids(self.assistant_kwargs, new_cur_len)

            # 2. Forecast next N tokens using the assistant model.
            assistant_generation_kwargs = {
                self.input_ids_key: input_ids,
                #"max_new_tokens": max_new_tokens,
                "max_new_tokens": 1, # only one token per drafting iteration
                "generation_config": self.generation_config,
                "logits_processor": self.logits_processor,
            }

            assistant_output = self.assistant_model.generate(**assistant_generation_kwargs, **self.assistant_kwargs)

            # 3. Update variables for the next round of candidate generation
            self.assistant_kwargs["past_key_values"] = assistant_output.past_key_values

            # 4. Prepare variables for output
            #candidate_logits = torch.stack(assistant_output.scores, dim=1)
            #candidate_ids = assistant_output.sequences

            # 5. Check if the threshold for early termination is triggered
            # Do not early stop on the first iteration (=> we can make a special case for that)
            if self.check_threshold_triggered(candidate_logits, assistant_output.scores[0], early_termination_parameters):
                # Case Threshold Triggered (early termination):
                # Return candidate_ids and candidate_logits (last iteration is NOT discarded)
                candidate_ids = assistant_output.sequences

                candidate_logits.append(assistant_output.scores[0])
                candidate_logits = tuple(candidate_logits)
                candidate_logits = torch.stack(candidate_logits, dim=1)

                return candidate_ids, candidate_logits
            
            else:
                # Case Threshold Not Triggered:
                # Prepare input_ids for next iteration
                candidate_ids = assistant_output.sequences
                input_ids = candidate_ids

                # Prepare candidate_logits for next iteration
                candidate_logits.append(assistant_output.scores[0])

            draft_iteration += 1

        # Case: max_new_tokens generated without early termination
        candidate_logits = tuple(candidate_logits)
        candidate_logits = torch.stack(candidate_logits, dim=1)
        return candidate_ids, candidate_logits


        
    

    def check_threshold_triggered(self,
                                  candidate_logits: torch.FloatTensor,
                                  cur_logits: torch.FloatTensor,
                                  early_termination_parameters: dict
        ) -> bool:

        assert(early_termination_parameters['type'] in ['static', 'ma', 'cum'])

        # logits consists of all generated logits so far including the current one
        logits = candidate_logits.copy()
        logits.append(cur_logits)
        
        # Static Threshold
        if early_termination_parameters['type'] == 'static':
            assert(early_termination_parameters['static_threshold_val'] is not None)

            e_cur = torch.distributions.categorical.Categorical(logits=logits[-1][0]).entropy()
            if e_cur >= early_termination_parameters['static_threshold_val']:
                return True
            return False
            
        # Moving Average (MA) 
        if early_termination_parameters['type'] == 'ma':
            assert(early_termination_parameters['ma_m'] is not None)
            assert(early_termination_parameters['ma_last_n'] is not None)

            # For moving average, we need at least one current and one past entropy
            # If not available, the rule wont trigger
            if len(logits) < 2:
                return False
            
            # Don't exceed the number of past logits defined by ma_last_n
            # We take as many past logits as available, but at most ma_last_n
            last_n = min(early_termination_parameters['ma_last_n'], len(logits)-1)

            e_ma = 0
            for i in range(last_n):
                # Sum up the past n squared entropies. The current entropy(-1) is not considered.
                e_ma +=  torch.square(torch.distributions.categorical.Categorical(logits=logits[-2-i][0]).entropy())
            
            # Calculate mean and multiply by factor m
            e_ma = (e_ma / last_n) * early_termination_parameters['ma_m']

            # Check if current entropy is above the moving average
            e_cur = torch.square(torch.distributions.categorical.Categorical(logits=logits[-1][0]).entropy())

            if e_cur >= e_ma:
                return True
            return False

        # Cumulative (cum)
        if early_termination_parameters['type'] == 'cum':
            assert(early_termination_parameters['cum_threshold_val'] is not None)
            assert(early_termination_parameters['cum_last_n'] is not None)

            # For Cumulative, we need at least 1 entropy (always given)

            # Don't exceed the number of past logits defined by cum_last_n
            # We take as many past logits as available, but at most ma_last_n
            last_n = min(early_termination_parameters['cum_last_n'], len(logits)-1)
            
            e_cum = 0
            for i in range(last_n+1):
                # Sum up the past n entropies as well as the current one.
                # Example: for cum_last_n=2, we sum up the current entropy and the 2 past entropies.
                e_cum += torch.square(torch.distributions.categorical.Categorical(logits=logits[-1-i][0]).entropy())

            # Check if current entropy is above the given cumulative threshold
            # Note: cum_threshold_val is the entropy sum, no mean.
            if e_cum >= early_termination_parameters['cum_threshold_val']:
                return True
            return False
    
    def update_candidate_strategy(
        self,
        input_ids: torch.LongTensor,
        candidate_input_ids: torch.LongTensor,
        new_logits: torch.FloatTensor,
        candidate_logits: torch.FloatTensor,
        candidate_length: int,
        num_matches: int,
        dml: DecodingMetricsLogger,
        verbose = False,
        #TBD: remove unnecessary arguments 
    ):
        if self.assistant_model.generation_config.num_assistant_tokens_schedule in {
        "heuristic",
        "heuristic_transient",
        }:
            self.dml = dml

            # Add Prompt Tokens to list (only first time)
            if not self.dml.prompt_present():
                prompt_tokens = candidate_input_ids[0, :-candidate_length].tolist()
                if verbose:
                    print(f'prompt_tokens: {prompt_tokens}')
                for i in range(len(prompt_tokens)):
                    p = DecodingMetricPoint(
                        token_id=prompt_tokens[i],
                        is_prompt=True)
                    
                    self.dml.log(p)


            for i in range(num_matches+1):
                
                no_candidate = candidate_logits is None or candidate_logits.shape[1] == i

                cur_token_id = new_logits.argmax(dim=-1)[0][i]

                if no_candidate:
                    """
                    In this case, there is no draft/candidate distribution b.c. the token was only predicted by target
                    """
                    entropy_target = torch.distributions.categorical.Categorical(logits=new_logits[0][i]).entropy()
                    p = DecodingMetricPoint(
                        token_id=int(cur_token_id),
                        is_prompt=False,
                        entropy_target=float(entropy_target)
                        )
                    
                    self.dml.log(p)

                else:
                    """
                    In every other case, we have draft/candidate distribution. 
                    """
                    cur_token_id_draft = candidate_logits.argmax(dim=-1)[0][i]
                    entropy_draft = torch.distributions.categorical.Categorical(logits=candidate_logits[0][i]).entropy()
                    entropy_target = torch.distributions.categorical.Categorical(logits=new_logits[0][i]).entropy()
                    cross_entropy = torch.nn.functional.cross_entropy(candidate_logits[0][i], new_logits[0][i].softmax(dim=-1))

                    p = DecodingMetricPoint(
                        token_id=int(cur_token_id),
                        is_prompt=False,
                        token_id_draft=int(cur_token_id_draft),
                        cross_entropy=float(cross_entropy),
                        entropy_target=float(entropy_target),
                        entropy_draft=float(entropy_draft)
                        )
                    
                    self.dml.log(p)

            if verbose:
                print(f'num_matches: {num_matches}, self.num_assistant_tokens: {self.num_assistant_tokens}')

            """
            print("input_ids", input_ids)
            print("candidate_input_ids", candidate_input_ids)
            #print("new_logits", new_logits)
            #print("candidate_logits", candidate_logits)
            print("candidate_length", candidate_length)
            print("input_ids.shape", input_ids.shape)
            print("candidate_input_ids.shape", candidate_input_ids.shape)
            print("candidate_logits.shape", candidate_logits.shape)
            print("new_logits.shape", new_logits.shape)
            print("candidate_length", candidate_length)
            print("candidate_logits.argmax(dim=-1)", candidate_logits.argmax(dim=-1))
            print("new_logits.argmax(dim=-1)", new_logits.argmax(dim=-1))
            print(num_matches)
            print("##############################################")
            """

            # Start Value for num_assistant_tokens is 5
            if num_matches == int(self.num_assistant_tokens):
                self.num_assistant_tokens += 2.0
            else:
                self.num_assistant_tokens = max(1.0, self.num_assistant_tokens - 1.0)

            # If num_k is specified, override HF (+2/-1) rule
            if self._num_k:
                self.num_assistant_tokens = self._num_k

            if verbose:
                print(f'num_assistant_tokens set to {self.num_assistant_tokens}')

            # Return: As an update method, this method does not return anything.
            # For debugging purposes we return measurement objects.

            return self.dml 
