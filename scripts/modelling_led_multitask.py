"""copied from huggingface transformers library.
PLEASE FIND A BETTER WAY OTHER THAN COPYING THE WHOLE CODE
"""
import pdb
import random
import inspect
from tkinter.messagebox import NO
from traceback import print_tb
from typing import Optional, Iterable, Callable, List
from xxlimited import new

import torch
import torch.nn.functional as F
from torch import nn, sigmoid
from torch.distributions.categorical import Categorical
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.models.led.modeling_led import (
    BaseModelOutputWithPastAndCrossAttentions, LEDConfig, LEDDecoderLayer,
    LEDEncoder, LEDEncoderBaseModelOutput, LEDLearnedPositionalEmbedding,
    LEDPreTrainedModel, LEDSeq2SeqLMOutput, LEDSeq2SeqModelOutput,
    _expand_mask, _make_causal_mask, shift_tokens_right)
from transformers.file_utils import ModelOutput
from generation_utils import logger, BeamSearchScorer, beam_search, beam_search_switch_tokens
from util import get_length_positions_and_segment_ids
# from sinusodial_positional_embedding import SinusoidalPositionalEmbedding
from sinusoidal_positional_embedding_new import SinusoidalPositionalEmbedding, SinusoidalPositionalEmbeddingPosition


class LEDModel(LEDPreTrainedModel):
    def __init__(self, config: LEDConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = LEDEncoder(config, self.shared)
        self.decoder = LEDDecoder(config, self.shared)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        global_attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        length=None
    ):
        # print("Led input", input_ids)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a LEDEncoderBaseModelOutput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, LEDEncoderBaseModelOutput):
            encoder_outputs = LEDEncoderBaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                global_attentions=encoder_outputs[3] if len(encoder_outputs) > 3 else None,
            )
            
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            length=length,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return LEDSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            encoder_global_attentions=encoder_outputs.global_attentions,
        )


class LEDModelPosition(LEDPreTrainedModel):
    def __init__(self, config: LEDConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = LEDEncoder(config, self.shared)
        self.decoder = LEDDecoderPosition(config, self.shared)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        global_attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        length_positions=None,
        segment_ids = None,
    ):
        # print("Led input", input_ids)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a LEDEncoderBaseModelOutput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, LEDEncoderBaseModelOutput):
            encoder_outputs = LEDEncoderBaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                global_attentions=encoder_outputs[3] if len(encoder_outputs) > 3 else None,
            )
            
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            length_positions=length_positions,
            segment_ids=segment_ids,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return LEDSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            encoder_global_attentions=encoder_outputs.global_attentions,
        )


class LEDForConditionalGenerationSwitchHead(LEDPreTrainedModel):
    base_model_prefix = "led"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]

    def __init__(self, config: LEDConfig):
        super().__init__(config)
        self.led = LEDModelPosition(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.led.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.led.shared.num_embeddings, bias=False)
        length_hidden_size = 100
        # self.lm_length = nn.Linear(config.d_model, 1 , bias=True)
        self.lm_length = nn.Sequential(
            nn.Linear(config.d_model, length_hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(length_hidden_size, 1, bias=True)
        )

        self.sigma_generation = torch.nn.Parameter(torch.tensor(.1, requires_grad=True))
        self.sigma_length = torch.nn.Parameter(torch.tensor(.1, requires_grad=True))

        self.init_weights()

    def get_encoder(self):
        return self.led.get_encoder()

    def get_decoder(self):
        return self.led.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def predict_generation_length(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        global_attention_mask=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.led.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask,
                    return_dict=return_dict,
                )
        # t = encoder_outputs[0]
        # avg_encoder = (t * torch.ne(input_ids, self.led.decoder.padding_idx).int().unsqueeze(2).repeat(1, 1, self.config.d_model)).sum(1) / torch.ne(input_ids, self.led.decoder.padding_idx).int().sum(-1).unsqueeze(1).repeat(1,self.config.d_model)
        
        cls_encoder = encoder_outputs[0][:,0,:]
        lm_length_pred = self.lm_length(cls_encoder)
        if self.config.predict_log_length:
            lm_length_pred = torch.exp(lm_length_pred)
        return lm_length_pred


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        global_attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        length_positions=None,
        length = None,
        segment_ids = None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        Conditional generation example::

            >>> from transformers import LEDTokenizer, LEDForConditionalGeneration
            >>> tokenizer = LEDTokenizer.from_pretrained('allenai/led-base-16384')
            >>> TXT = "My friends are <mask> but they eat too many carbs."

            >>> model = LEDForConditionalGeneration.from_pretrained('allenai/led-base-16384')
            >>> input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
            >>> logits = model(input_ids).logits

            >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
            >>> probs = logits[0, masked_index].softmax(dim=0)
            >>> values, predictions = probs.topk(5)

            >>> tokenizer.decode(predictions).split()
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        if encoder_outputs is None:
            encoder_outputs = self.led.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
        # t = encoder_outputs[0]
        # avg_encoder = (t * torch.ne(input_ids, self.led.decoder.padding_idx).int().unsqueeze(2).repeat(1, 1, self.config.d_model)).sum(1) / torch.ne(input_ids, self.led.decoder.padding_idx).int().sum(-1).unsqueeze(1).repeat(1,self.config.d_model)
        
        cls_encoder = encoder_outputs[0][:,0,:]
        lm_length_pred = self.lm_length(cls_encoder)

        # batch size
        lm_length_pred = lm_length_pred.squeeze()
        if labels is not None:
            print("length: ", lm_length_pred if not self.config.predict_log_length else torch.exp(lm_length_pred))
        
        if self.led.decoder.embed_positions.sinpostype != self.config.sinpostype:
            print(f"Warning: model.led.decoder.embed_positions.sinpostype is being changed from {self.led.decoder.embed_positions.sinpostype} to  {self.config.sinpostype}")
            self.led.decoder.embed_positions.sinpostype = self.config.sinpostype        


        outputs = self.led(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            global_attention_mask=global_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            length_positions=length_positions,
            segment_ids = segment_ids,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        # TODO generation code pending
        # have to infer the length_positions for next timestep using current switch_token_logits
        
        masked_lm_loss, length_loss = None, None
        
        if labels is not None:
            # training time

            if self.config.predict_log_length:
                length = torch.log(length.float())

            loss_fct = CrossEntropyLoss()
            masked_lm_loss = self.config.generation_loss_coef * loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            print("Crosss entropy: ", masked_lm_loss)
            
            mse = MSELoss()
            length_loss = self.config.length_loss_coef * torch.sqrt(mse(lm_length_pred, length.float()))
            
            masked_lm_loss += length_loss
            print("length loss: ", length_loss)

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return LEDSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            encoder_global_attentions=outputs.encoder_global_attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        context_length_prev = None,
        context_length_next = None,
        **model_kwargs,
    ):
        r"""
        Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
        multinomial sampling, beam-search decoding, and beam-search multinomial sampling.

        Apart from :obj:`input_ids` and :obj:`attention_mask`, all the arguments below will default to the value of the
        attribute of the same name inside the :class:`~transformers.PretrainedConfig` of the model. The default values
        indicated are the default values of those config.

        Most of these parameters are explained in more detail in `this blog post
        <https://huggingface.co/blog/how-to-generate>`__.

        Parameters:

            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`None` the method initializes it as an empty
                :obj:`torch.LongTensor` of shape :obj:`(1,)`.
            max_length (:obj:`int`, `optional`, defaults to 20):
                The maximum length of the sequence to be generated.
            min_length (:obj:`int`, `optional`, defaults to 10):
                The minimum length of the sequence to be generated.
            do_sample (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to use sampling ; use greedy decoding otherwise.
            early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to stop the beam search when at least ``num_beams`` sentences are finished per batch or not.
            num_beams (:obj:`int`, `optional`, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            temperature (:obj:`float`, `optional`, defaults tp 1.0):
                The value used to module the next token probabilities.
            top_k (:obj:`int`, `optional`, defaults to 50):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (:obj:`float`, `optional`, defaults to 1.0):
                If set to float < 1, only the most probable tokens with probabilities that add up to :obj:`top_p` or
                higher are kept for generation.
            repetition_penalty (:obj:`float`, `optional`, defaults to 1.0):
                The parameter for repetition penalty. 1.0 means no penalty. See `this paper
                <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            bos_token_id (:obj:`int`, `optional`):
                The id of the `beginning-of-sequence` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            length_penalty (:obj:`float`, `optional`, defaults to 1.0):
                Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the
                model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer
                sequences.
            no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 0):
                If set to int > 0, all ngrams of that size can only occur once.
            bad_words_ids(:obj:`List[List[int]]`, `optional`):
                List of token ids that are not allowed to be generated. In order to get the tokens of the words that
                should not appear in the generated text, use :obj:`tokenizer(bad_word,
                add_prefix_space=True).input_ids`.
            num_return_sequences(:obj:`int`, `optional`, defaults to 1):
                The number of independently computed returned sequences for each element in the batch.
            attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values are in ``[0, 1]``, 1 for
                tokens that are not masked, and 0 for masked tokens. If not provided, will default to a tensor the same
                shape as :obj:`input_ids` that masks the pad token. `What are attention masks?
                <../glossary.html#attention-mask>`__
            decoder_start_token_id (:obj:`int`, `optional`):
                If an encoder-decoder model starts decoding with a different token than `bos`, the id of that token.
            use_cache: (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not the model should use the past last key/values attentions (if applicable to the model) to
                speed up decoding.
            num_beam_groups (:obj:`int`, `optional`, defaults to 1):
                Number of groups to divide :obj:`num_beams` into in order to ensure diversity among different groups of
                beams. `this paper <https://arxiv.org/pdf/1610.02424.pdf>`__ for more details.
            diversity_penalty (:obj:`float`, `optional`, defaults to 0.0):
                This value is subtracted from a beam's score if it generates a token same as any beam from other group
                at a particular time. Note that :obj:`diversity_penalty` is only effective if ``group beam search`` is
                enabled.
            prefix_allowed_tokens_fn: (:obj:`Callable[[int, torch.Tensor], List[int]]`, `optional`):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments :obj:`inputs_ids` and the batch ID
                :obj:`batch_id`. It has to return a list with the allowed tokens for the next generation step
                conditioned on the previously generated tokens :obj:`inputs_ids` and the batch ID :obj:`batch_id`. This
                argument is useful for constrained generation conditioned on the prefix, as described in
                `Autoregressive Entity Retrieval <https://arxiv.org/abs/2010.00904>`__.
            output_attentions (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more details.
            output_hidden_states (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return trhe hidden states of all layers. See ``hidden_states`` under returned tensors
                for more details.
            output_scores (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the prediction scores. See ``scores`` under returned tensors for more details.
            return_dict_in_generate (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.

            model_kwargs:
                Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model. If the
                model is an encoder-decoder model, encoder specific kwargs should not be prefixed and decoder specific
                kwargs should be prefixed with `decoder_`.

        Return:
            :class:`~transformers.file_utils.ModelOutput` or :obj:`torch.LongTensor`: A
            :class:`~transformers.file_utils.ModelOutput` (if ``return_dict_in_generate=True`` or when
            ``config.return_dict_in_generate=True``) or a :obj:`torch.FloatTensor`.

                If the model is `not` an encoder-decoder model (``model.config.is_encoder_decoder=False``), the
                possible :class:`~transformers.file_utils.ModelOutput` types are:

                    - :class:`~transformers.generation_utils.GreedySearchDecoderOnlyOutput`,
                    - :class:`~transformers.generation_utils.SampleDecoderOnlyOutput`,
                    - :class:`~transformers.generation_utils.BeamSearchDecoderOnlyOutput`,
                    - :class:`~transformers.generation_utils.BeamSampleDecoderOnlyOutput`

                If the model is an encoder-decoder model (``model.config.is_encoder_decoder=True``), the possible
                :class:`~transformers.file_utils.ModelOutput` types are:

                    - :class:`~transformers.generation_utils.GreedySearchEncoderDecoderOutput`,
                    - :class:`~transformers.generation_utils.SampleEncoderDecoderOutput`,
                    - :class:`~transformers.generation_utils.BeamSearchEncoderDecoderOutput`,
                    - :class:`~transformers.generation_utils.BeamSampleEncoderDecoderOutput`

        Examples::
            >>> from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

            >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
            >>> # do greedy decoding without providing a prompt
            >>> outputs = model.generate(max_length=40)
            >>> print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))

            >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
            >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
            >>> document = (
            ... "at least two people were killed in a suspected bomb attack on a passenger bus "
            ... "in the strife-torn southern philippines on monday , the military said."
            ... )
            >>> # encode input contex
            >>> input_ids = tokenizer(document, return_tensors="pt").input_ids
            >>> # generate 3 independent sequences using beam search decoding (5 beams)
            >>> # with T5 encoder-decoder model conditioned on short news article.
            >>> outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3)
            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))

            >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
            >>> input_context = "The dog"
            >>> # encode input context
            >>> input_ids = tokenizer(input_context, return_tensors="pt").input_ids
            >>> # generate 3 candidates using sampling
            >>> outputs = model.generate(input_ids=input_ids, max_length=20, num_return_sequences=3, do_sample=True)
            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))

            >>> tokenizer = AutoTokenizer.from_pretrained("ctrl")
            >>> model = AutoModelForCausalLM.from_pretrained("ctrl")
            >>> # "Legal" is one of the control codes for ctrl
            >>> input_context = "Legal My neighbor is"
            >>> # encode input context
            >>> input_ids = tokenizer(input_context, return_tensors="pt").input_ids
            >>> outputs = model.generate(input_ids=input_ids, max_length=20, repetition_penalty=1.2)
            >>> print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))

            >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
            >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
            >>> input_context = "My cute dog"
            >>> # get tokens of words that should not be generated
            >>> bad_words_ids = [tokenizer(bad_word, add_prefix_space=True).input_ids for bad_word in ["idiot", "stupid", "shut up"]]
            >>> # encode input context
            >>> input_ids = tokenizer(input_context, return_tensors="pt").input_ids
            >>> # generate sequences without allowing bad_words to be generated
            >>> outputs = model.generate(input_ids=input_ids, max_length=20, do_sample=True, bad_words_ids=bad_words_ids)
            >>> print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))
        """

        # set init values
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
        max_length = max_length if max_length is not None else self.config.max_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )

        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        model_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = output_hidden_states

        # length prediction
        length_prediction = self.predict_generation_length(
            input_ids=input_ids,
            attention_mask=model_kwargs["attention_mask"],
            global_attention_mask=model_kwargs["global_attention_mask"],
            return_dict=return_dict_in_generate,
        )

        (length_positions, _), (segment_ids, _) = get_length_positions_and_segment_ids(
            context_length_prev=context_length_prev, 
            span_length=length_prediction, 
            context_length_next=context_length_next, 
            max_len=max_length
        )
        length_positions = length_positions.to(input_ids.device)
        segment_ids = segment_ids.to(input_ids.device)

        if input_ids is None:
            # init `input_ids` with bos_token_id
            input_ids = self._prepare_input_ids_for_generation(bos_token_id)

        if model_kwargs.get("attention_mask", None) is None:
            # init `attention_mask` depending on `pad_token_id`
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                input_ids, pad_token_id, eos_token_id
            )

        # special case if pad_token_id is not defined
        if pad_token_id is None and eos_token_id is not None:
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            pad_token_id = eos_token_id

        if self.config.is_encoder_decoder:
            # add encoder_outputs to model_kwargs
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(input_ids, model_kwargs)

            # set input_ids as decoder_input_ids
            if "decoder_input_ids" in model_kwargs:
                input_ids = model_kwargs.pop("decoder_input_ids")
            else:
                input_ids = self._prepare_decoder_input_ids_for_generation(
                    input_ids, decoder_start_token_id=decoder_start_token_id, bos_token_id=bos_token_id
                )

            if "encoder_outputs" not in model_kwargs or not isinstance(model_kwargs["encoder_outputs"], ModelOutput):
                raise ValueError("Make sure that `model_kwargs` include `encoder_outputs` of type `ModelOutput`.")

        if input_ids.shape[-1] >= max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids.shape[-1]}, but ``max_length`` is set to {max_length}."
                "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or ``max_length``."
            )

        # determine generation mode
        is_beam_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is False
        is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1)
        if num_beam_groups > num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )

        # set model_kwargs
        model_kwargs["use_cache"] = use_cache

        # get distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            eos_token_id=eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
        )

        if is_beam_gen_mode:
            batch_size = input_ids.shape[0]

            length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
            early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping

            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                max_length=max_length,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            # interleave with `num_beams`
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
            )
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                length_positions=length_positions,
                segment_ids = segment_ids,
                **model_kwargs,
            )
        else:
            raise ValueError("only is_beam_gen_mode is supported")



    def prepare_inputs_for_generation(
        self, decoder_input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        result =  {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }
        kwargs = {k.replace("decoder_", ""):v for k, v in kwargs.items()}
        result.update(kwargs)

        return result

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer,
        logits_processor = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        length_positions = None,
        segment_ids = None,
        **model_kwargs
        ):
        return beam_search_switch_tokens(
            self,
            input_ids,
            beam_scorer,
            logits_processor,
            max_length,
            pad_token_id,
            eos_token_id,
            output_attentions,
            output_hidden_states,
            output_scores,
            return_dict_in_generate,
            length_positions = length_positions,
            segment_ids = segment_ids,
            **model_kwargs,
        )



class LEDForConditionalGenerationAddContextLength(LEDPreTrainedModel):
    base_model_prefix = "led"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]

    def __init__(self, config: LEDConfig):
        super().__init__(config)
        self.led = LEDModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.led.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.led.shared.num_embeddings, bias=False)
        length_hidden_size = 100
        # self.lm_length = nn.Linear(config.d_model, 1 , bias=True)
        self.lm_length = nn.Sequential(
            nn.Linear(config.d_model, length_hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(length_hidden_size, 1, bias=True)
        )

        self.sigma_generation = torch.nn.Parameter(torch.tensor(.1, requires_grad=True))
        self.sigma_length = torch.nn.Parameter(torch.tensor(.1, requires_grad=True))

        self.init_weights()

    def get_encoder(self):
        return self.led.get_encoder()

    def get_decoder(self):
        return self.led.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        global_attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        length=None,
        context_length = None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        Conditional generation example::

            >>> from transformers import LEDTokenizer, LEDForConditionalGeneration
            >>> tokenizer = LEDTokenizer.from_pretrained('allenai/led-base-16384')
            >>> TXT = "My friends are <mask> but they eat too many carbs."

            >>> model = LEDForConditionalGeneration.from_pretrained('allenai/led-base-16384')
            >>> input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
            >>> logits = model(input_ids).logits

            >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
            >>> probs = logits[0, masked_index].softmax(dim=0)
            >>> values, predictions = probs.topk(5)

            >>> tokenizer.decode(predictions).split()
        """
        # print("Cond gen input", input_ids)
        # print("Cond gen decoder input ids", decoder_input_ids)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        if encoder_outputs is None:
            encoder_outputs = self.led.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
        # t = encoder_outputs[0]
        # avg_encoder = (t * torch.ne(input_ids, self.led.decoder.padding_idx).int().unsqueeze(2).repeat(1, 1, self.config.d_model)).sum(1) / torch.ne(input_ids, self.led.decoder.padding_idx).int().sum(-1).unsqueeze(1).repeat(1,self.config.d_model)
        
        cls_encoder = encoder_outputs[0][:,0,:]
        lm_length_pred = self.lm_length(cls_encoder)

        # batch size
        lm_length_pred = lm_length_pred.squeeze()
        if labels is not None:
            print("length: ", lm_length_pred if not self.config.predict_log_length else torch.exp(lm_length_pred))
        
        if self.led.decoder.embed_positions.sinpostype != self.config.sinpostype:
            print(f"Warning: model.led.decoder.embed_positions.sinpostype is being changed from {self.led.decoder.embed_positions.sinpostype} to  {self.config.sinpostype}")
            self.led.decoder.embed_positions.sinpostype = self.config.sinpostype
        
        desired_length = lm_length_pred if not self.config.predict_log_length else torch.exp(lm_length_pred)
        
        if labels is not None:
            # training time
            rand_gen = random.random() # generate a float in (0,1)
            if self.config.length_schedule_sampling_prob > 0:
                print("length_schedule_sampling_prob: ", self.config.length_schedule_sampling_prob)
            
            if rand_gen <= self.config.length_schedule_sampling_prob:
                desired_length = length
            else:
                # desired_length is already predicted length
                pass

        new_desired_length = desired_length + context_length
        if self.config.add_perturbation:
            perturb = torch.randn(desired_length.shape) * 4
            new_desired_length = desired_length + context_length + perturb.to(device=desired_length.device)
            # print("perturbation: ", perturb)
            # print("new_desired_length: ", new_desired_length)

        outputs = self.led(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            global_attention_mask=global_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            length=new_desired_length,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss, length_loss = None, None
        if labels is not None:
            # training time

            if self.config.predict_log_length:
                length = torch.log(length.float())
            
            if self.config.learnable_multitask_learning:
                loss_fct = CrossEntropyLoss()
                masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
                lm_loss_weight = torch.exp(-2* self.sigma_generation) / 2
                
                mse = MSELoss()
                length_loss =  torch.sqrt(mse(lm_length_pred, length.float()))
                lm_length_weight = torch.exp(- 2 * self.sigma_length)
                masked_lm_loss  = lm_loss_weight * masked_lm_loss + lm_length_weight * length_loss + lm_loss_weight + lm_length_weight

            else:
                if not self.config.length_loss_only:                
                    loss_fct = CrossEntropyLoss()
                    masked_lm_loss = self.config.generation_loss_coef * loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
                    print("Crosss entropy: ", masked_lm_loss)
                if not self.config.generation_loss_only:
                    mse = MSELoss()

                    length_loss = self.config.length_loss_coef * torch.sqrt(mse(lm_length_pred, length.float()))
                    
                    
                    if masked_lm_loss is None:
                        masked_lm_loss = length_loss
                    else:
                        masked_lm_loss += length_loss

            # print("total loss: ", masked_lm_loss)
            # print("length pred: ", lm_length_pred)

            

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return LEDSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            encoder_global_attentions=outputs.encoder_global_attentions,
        )


    def prepare_inputs_for_generation(
        self, decoder_input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        result =  {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }
        kwargs = {k.replace("decoder_", ""):v for k, v in kwargs.items()}
        result.update(kwargs)

        return result

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


class LEDForConditionalGeneration(LEDPreTrainedModel):
    base_model_prefix = "led"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]

    def __init__(self, config: LEDConfig):
        super().__init__(config)
        self.led = LEDModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.led.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.led.shared.num_embeddings, bias=False)
        length_hidden_size = 100
        # self.lm_length = nn.Linear(config.d_model, 1 , bias=True)
        self.lm_length = nn.Sequential(
            nn.Linear(config.d_model, length_hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(length_hidden_size, 1, bias=True)
        )

        self.sigma_generation = torch.nn.Parameter(torch.tensor(.1, requires_grad=True))
        self.sigma_length = torch.nn.Parameter(torch.tensor(.1, requires_grad=True))

        self.init_weights()

    def get_encoder(self):
        return self.led.get_encoder()

    def get_decoder(self):
        return self.led.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        global_attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        length=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        Conditional generation example::

            >>> from transformers import LEDTokenizer, LEDForConditionalGeneration
            >>> tokenizer = LEDTokenizer.from_pretrained('allenai/led-base-16384')
            >>> TXT = "My friends are <mask> but they eat too many carbs."

            >>> model = LEDForConditionalGeneration.from_pretrained('allenai/led-base-16384')
            >>> input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
            >>> logits = model(input_ids).logits

            >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
            >>> probs = logits[0, masked_index].softmax(dim=0)
            >>> values, predictions = probs.topk(5)

            >>> tokenizer.decode(predictions).split()
        """
        # print("Cond gen input", input_ids)
        # print("Cond gen decoder input ids", decoder_input_ids)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        if encoder_outputs is None:
            encoder_outputs = self.led.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
        # t = encoder_outputs[0]
        # avg_encoder = (t * torch.ne(input_ids, self.led.decoder.padding_idx).int().unsqueeze(2).repeat(1, 1, self.config.d_model)).sum(1) / torch.ne(input_ids, self.led.decoder.padding_idx).int().sum(-1).unsqueeze(1).repeat(1,self.config.d_model)
        
        cls_encoder = encoder_outputs[0][:,0,:]
        # print(avg_encoder.shape)
        # print("Sum", avg_encoder.sum())
        lm_length_pred = self.lm_length(cls_encoder)
        # print("length", lm_length_pred[0][0])
        # print(length[0])

        # batch size
        lm_length_pred = lm_length_pred.squeeze()
        if labels is not None:
            print("length: ", lm_length_pred if not self.config.predict_log_length else torch.exp(lm_length_pred))
    
        
        if self.led.decoder.embed_positions.sinpostype != self.config.sinpostype:
            print(f"Warning: model.led.decoder.embed_positions.sinpostype is being changed from {self.led.decoder.embed_positions.sinpostype} to  {self.config.sinpostype}")
            self.led.decoder.embed_positions.sinpostype = self.config.sinpostype
        
        desired_length = lm_length_pred if not self.config.predict_log_length else torch.exp(lm_length_pred)
        
        if labels is not None:
            # training time
            rand_gen = random.random() # generate a float in (0,1)
            if self.config.length_schedule_sampling_prob > 0:
                print("length_schedule_sampling_prob: ", self.config.length_schedule_sampling_prob)
            
            if rand_gen < self.config.length_schedule_sampling_prob:
                desired_length = length
            else:
                # desired_length is already predicted length
                pass
        
        new_desired_length = desired_length
        if self.config.add_perturbation:
            perturb = torch.randn(desired_length.shape) * 2
            new_desired_length = desired_length + perturb.to(device=desired_length.device)
            print("new_desired_length: ", new_desired_length)

        outputs = self.led(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            global_attention_mask=global_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            length=new_desired_length,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        

        masked_lm_loss, length_loss = None, None
        if labels is not None:
            # training time

            if self.config.predict_log_length:
                length = torch.log(length.float())
            
            if self.config.learnable_multitask_learning:
                loss_fct = CrossEntropyLoss()
                masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
                lm_loss_weight = torch.exp(-2* self.sigma_generation) / 2
                
                mse = MSELoss()
                length_loss =  torch.sqrt(mse(lm_length_pred, length.float()))
                lm_length_weight = torch.exp(- 2 * self.sigma_length)
                masked_lm_loss  = lm_loss_weight * masked_lm_loss + lm_length_weight * length_loss + lm_loss_weight + lm_length_weight

                print(self.sigma_length, self.sigma_generation)

            else:
                if not self.config.length_loss_only:                
                    loss_fct = CrossEntropyLoss()
                    masked_lm_loss = self.config.generation_loss_coef * loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
                    print("Crosss entropy: ", masked_lm_loss)
                if not self.config.generation_loss_only:
                    mse = MSELoss()

                    length_loss = self.config.length_loss_coef * torch.sqrt(mse(lm_length_pred, length.float()))
                    
                    
                    if masked_lm_loss is None:
                        masked_lm_loss = length_loss
                    else:
                        masked_lm_loss += length_loss
                    print("length loss: ", length_loss)

            print("total loss: ", masked_lm_loss)
            # print("length pred: ", lm_length_pred)

            

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return LEDSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            encoder_global_attentions=outputs.encoder_global_attentions,
        )


    def prepare_inputs_for_generation(
        self, decoder_input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        result =  {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }
        kwargs = {k.replace("decoder_", ""):v for k, v in kwargs.items()}
        result.update(kwargs)

        return result

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


class LEDForConditionalGenerationV2(LEDPreTrainedModel):
    base_model_prefix = "led"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]

    def __init__(self, config: LEDConfig):
        super().__init__(config)
        self.led = LEDModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.led.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.led.shared.num_embeddings, bias=False)
        length_hidden_size = 250
        # self.lm_length = nn.Linear(config.d_model, 1 , bias=True)
        self.lm_length = nn.Sequential(
            nn.Linear(config.d_model, length_hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(length_hidden_size, 1, bias=True)
        )

        self.sigma_generation = torch.nn.Parameter(torch.tensor(.1, requires_grad=True))
        self.sigma_length = torch.nn.Parameter(torch.tensor(.1, requires_grad=True))

        self.init_weights()

    def get_encoder(self):
        return self.led.get_encoder()

    def get_decoder(self):
        return self.led.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        global_attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        length=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        Conditional generation example::

            >>> from transformers import LEDTokenizer, LEDForConditionalGeneration
            >>> tokenizer = LEDTokenizer.from_pretrained('allenai/led-base-16384')
            >>> TXT = "My friends are <mask> but they eat too many carbs."

            >>> model = LEDForConditionalGeneration.from_pretrained('allenai/led-base-16384')
            >>> input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
            >>> logits = model(input_ids).logits

            >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
            >>> probs = logits[0, masked_index].softmax(dim=0)
            >>> values, predictions = probs.topk(5)

            >>> tokenizer.decode(predictions).split()
        """
        # print("Cond gen input", input_ids)
        # print("Cond gen decoder input ids", decoder_input_ids)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        if encoder_outputs is None:
            encoder_outputs = self.led.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
        # t = encoder_outputs[0]
        # avg_encoder = (t * torch.ne(input_ids, self.led.decoder.padding_idx).int().unsqueeze(2).repeat(1, 1, self.config.d_model)).sum(1) / torch.ne(input_ids, self.led.decoder.padding_idx).int().sum(-1).unsqueeze(1).repeat(1,self.config.d_model)
        
        cls_encoder = encoder_outputs[0][:,0,:]
        lm_length_pred = self.lm_length(cls_encoder)
        if labels is not None:
            print("length: ", lm_length_pred if not self.config.predict_log_length else torch.exp(lm_length_pred))
        
        if self.led.decoder.embed_positions.sinpostype != self.config.sinpostype:
            print(f"Warning: model.led.decoder.embed_positions.sinpostype is being changed from {self.led.decoder.embed_positions.sinpostype} to  {self.config.sinpostype}")
            self.led.decoder.embed_positions.sinpostype = self.config.sinpostype
        
        desired_length = lm_length_pred if not self.config.predict_log_length else torch.exp(lm_length_pred)
        
        if labels is not None:
            # training time
            rand_gen = random.random() # generate a float in (0,1)
            if self.config.length_schedule_sampling_prob > 0:
                print("length_schedule_sampling_prob: ", self.config.length_schedule_sampling_prob)
            
            if rand_gen < self.config.length_schedule_sampling_prob:
                desired_length = length
            else:
                # desired_length is already predicted length
                pass

        outputs = self.led(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            global_attention_mask=global_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            length=desired_length,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        

        masked_lm_loss, length_loss = None, None
        if labels is not None:

            if self.config.predict_log_length:
                length = torch.log(length.float())
            
            if self.config.learnable_multitask_learning:
                loss_fct = CrossEntropyLoss()
                masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
                lm_loss_weight = torch.exp(-2* self.sigma_generation) / 2
                
                mse = MSELoss()
                length_loss =  torch.sqrt(mse(lm_length_pred, length.float()))
                lm_length_weight = torch.exp(- 2 * self.sigma_length)
                masked_lm_loss  = lm_loss_weight * masked_lm_loss + lm_length_weight * length_loss + lm_loss_weight + lm_length_weight

                print(self.sigma_length, self.sigma_generation)

            else:
                if not self.config.length_loss_only:                
                    loss_fct = CrossEntropyLoss()
                    masked_lm_loss = self.config.generation_loss_coef * loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
                    print("Crosss entropy: ", masked_lm_loss)
                if not self.config.generation_loss_only:
                    mse = MSELoss()

                    length_loss = self.config.length_loss_coef * torch.sqrt(mse(lm_length_pred, length.float()))
                    
                    
                    if masked_lm_loss is None:
                        masked_lm_loss = length_loss
                    else:
                        masked_lm_loss += length_loss
                    print("length loss: ", length_loss)

            print("total loss: ", masked_lm_loss)
            # print("length pred: ", lm_length_pred)

            

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return LEDSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            encoder_global_attentions=outputs.encoder_global_attentions,
        )


    def prepare_inputs_for_generation(
        self, decoder_input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        result =  {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }
        kwargs = {k.replace("decoder_", ""):v for k, v in kwargs.items()}
        result.update(kwargs)

        return result

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


class ConditionalLengthPredicton(nn.Module):

    def __init__(self, d_model, length_hidden_size=200):
        super().__init__()
        self.lm_length_dominant = nn.Sequential(
            nn.Linear(d_model, length_hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(length_hidden_size, 1, bias=True)
        )

        self.lm_length_reference = nn.Sequential(
            nn.Linear(d_model, length_hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(length_hidden_size, 1, bias=True)
        )
    
    def forward(self,x, y):
        return torch.where(y == 0, self.lm_length_reference(x) , self.lm_length_dominant(x))



class LEDForConditionalGenerationMultipleLengthPredictor(LEDPreTrainedModel):
    base_model_prefix = "led"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]  

    def __init__(self, config: LEDConfig):
        super().__init__(config)
        self.led = LEDModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.led.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.led.shared.num_embeddings, bias=False)
        length_hidden_size = 200
        self.lm_length = ConditionalLengthPredicton(
            d_model = config.d_model, 
            length_hidden_size=length_hidden_size
        )
        self.sigma_generation = torch.nn.Parameter(torch.tensor(.1, requires_grad=True))
        self.sigma_length = torch.nn.Parameter(torch.tensor(.1, requires_grad=True))

        self.init_weights()

    def get_encoder(self):
        return self.led.get_encoder()

    def get_decoder(self):
        return self.led.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        global_attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        length=None,
        citation_types = None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        Conditional generation example::

            >>> from transformers import LEDTokenizer, LEDForConditionalGeneration
            >>> tokenizer = LEDTokenizer.from_pretrained('allenai/led-base-16384')
            >>> TXT = "My friends are <mask> but they eat too many carbs."

            >>> model = LEDForConditionalGeneration.from_pretrained('allenai/led-base-16384')
            >>> input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
            >>> logits = model(input_ids).logits

            >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
            >>> probs = logits[0, masked_index].softmax(dim=0)
            >>> values, predictions = probs.topk(5)

            >>> tokenizer.decode(predictions).split()
        """
        # print("Cond gen input", input_ids)
        # print("Cond gen decoder input ids", decoder_input_ids)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        if encoder_outputs is None:
            encoder_outputs = self.led.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
        # t = encoder_outputs[0]
        # avg_encoder = (t * torch.ne(input_ids, self.led.decoder.padding_idx).int().unsqueeze(2).repeat(1, 1, self.config.d_model)).sum(1) / torch.ne(input_ids, self.led.decoder.padding_idx).int().sum(-1).unsqueeze(1).repeat(1,self.config.d_model)
        
        cls_encoder = encoder_outputs[0][:,0,:]
        lm_length_pred = self.lm_length(cls_encoder, torch.unsqueeze(citation_types, dim=-1))
        if labels is not None:
            print("length: ", lm_length_pred if not self.config.predict_log_length else torch.exp(lm_length_pred))
        
        if self.led.decoder.embed_positions.sinpostype != self.config.sinpostype:
            print(f"Warning: model.led.decoder.embed_positions.sinpostype is being changed from {self.led.decoder.embed_positions.sinpostype} to  {self.config.sinpostype}")
            self.led.decoder.embed_positions.sinpostype = self.config.sinpostype
        
        desired_length = lm_length_pred if not self.config.predict_log_length else torch.exp(lm_length_pred)
        
        if labels is not None:
            # training time
            rand_gen = random.random() # generate a float in (0,1)
            if self.config.length_schedule_sampling_prob > 0:
                print("length_schedule_sampling_prob: ", self.config.length_schedule_sampling_prob)
            
            if rand_gen < self.config.length_schedule_sampling_prob:
                desired_length = length
            else:
                # desired_length is already predicted length
                pass

        outputs = self.led(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            global_attention_mask=global_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            length=desired_length,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        

        masked_lm_loss, length_loss = None, None
        if labels is not None:

            if self.config.predict_log_length:
                length = torch.log(length.float())
            
            if self.config.learnable_multitask_learning:
                loss_fct = CrossEntropyLoss()
                masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
                lm_loss_weight = torch.exp(-2* self.sigma_generation) / 2
                
                mse = MSELoss()
                length_loss =  torch.sqrt(mse(lm_length_pred, length.float()))
                lm_length_weight = torch.exp(- 2 * self.sigma_length)
                masked_lm_loss  = lm_loss_weight * masked_lm_loss + lm_length_weight * length_loss + lm_loss_weight + lm_length_weight

                print(self.sigma_length, self.sigma_generation)

            else:
                if not self.config.length_loss_only:                
                    loss_fct = CrossEntropyLoss()
                    masked_lm_loss = self.config.generation_loss_coef * loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
                    print("Crosss entropy: ", masked_lm_loss)
                if not self.config.generation_loss_only:
                    mse = MSELoss()

                    length_loss = self.config.length_loss_coef * torch.sqrt(mse(lm_length_pred, length.float()))
                    
                    
                    if masked_lm_loss is None:
                        masked_lm_loss = length_loss
                    else:
                        masked_lm_loss += length_loss
                    print("length loss: ", length_loss)

            print("total loss: ", masked_lm_loss)
            # print("length pred: ", lm_length_pred)

            

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return LEDSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            encoder_global_attentions=outputs.encoder_global_attentions,
        )


    def prepare_inputs_for_generation(
        self, decoder_input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        result =  {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }
        kwargs = {k.replace("decoder_", ""):v for k, v in kwargs.items()}
        result.update(kwargs)

        return result

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


class LEDForConditionalGenerationLengthClassificationReinforcement(LEDPreTrainedModel):
    base_model_prefix = "led"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]  

    def __init__(self, config: LEDConfig):
        super().__init__(config)
        self.led = LEDModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.led.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.led.shared.num_embeddings, bias=False)
        length_hidden_size = 200
        self.length_bucket_count = 6
        # self.length_class_map = {0: 10, 1:20, 2: 30, 3:40, 4:50, 5:60}
        # self.length_class_map_indexed = torch.nn.Parameter(torch.tensor([10., 20., 30., 40., 50., 60.]))

        self.lm_length = nn.Sequential(
            nn.Linear(config.d_model, length_hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(length_hidden_size, self.length_bucket_count, bias=True)
        )

        self.init_weights()

    def get_encoder(self):
        return self.led.get_encoder()

    def get_decoder(self):
        return self.led.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        global_attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        length_labels=None,
        baseline_loss=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        Conditional generation example::

            >>> from transformers import LEDTokenizer, LEDForConditionalGeneration
            >>> tokenizer = LEDTokenizer.from_pretrained('allenai/led-base-16384')
            >>> TXT = "My friends are <mask> but they eat too many carbs."

            >>> model = LEDForConditionalGeneration.from_pretrained('allenai/led-base-16384')
            >>> input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
            >>> logits = model(input_ids).logits

            >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
            >>> probs = logits[0, masked_index].softmax(dim=0)
            >>> values, predictions = probs.topk(5)

            >>> tokenizer.decode(predictions).split()
        """
        # print("Cond gen input", input_ids)
        # print("Cond gen decoder input ids", decoder_input_ids)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        if encoder_outputs is None:
            encoder_outputs = self.led.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
        # t = encoder_outputs[0]
        # avg_encoder = (t * torch.ne(input_ids, self.led.decoder.padding_idx).int().unsqueeze(2).repeat(1, 1, self.config.d_model)).sum(1) / torch.ne(input_ids, self.led.decoder.padding_idx).int().sum(-1).unsqueeze(1).repeat(1,self.config.d_model)
        
        cls_encoder = encoder_outputs[0][:,0,:]
        lm_length_logits = self.lm_length(cls_encoder)
        
        if self.led.decoder.embed_positions.sinpostype != self.config.sinpostype:
            print(f"Warning: model.led.decoder.embed_positions.sinpostype is being changed from {self.led.decoder.embed_positions.sinpostype} to  {self.config.sinpostype}")
            self.led.decoder.embed_positions.sinpostype = self.config.sinpostype
        
        desired_length_label = torch.argmax(lm_length_logits, dim=-1)
        desired_length = self.length_class_map_indexed[desired_length_label]
        
        if labels is not None:
            # training time
            desired_length = self.length_class_map_indexed[length_labels]
        else:
            # test time --> desired_length is already predicted length
            pass

        outputs = self.led(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            global_attention_mask=global_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            length=desired_length,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss, length_loss, rl_loss = None, 0, 0
        if labels is not None:
            
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = self.config.generation_loss_coef * loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            print("Crosss entropy: ", masked_lm_loss)
            

            length_loss_fct = CrossEntropyLoss()
            print("lm_length_logits shape: ", lm_length_logits.shape)
            print("length_labels shape: ", length_labels.shape)
            length_loss = self.config.length_loss_coef * length_loss_fct(lm_length_logits.view(-1, self.length_bucket_count), length_labels.view(-1))
            
            # reinforce
            if baseline_loss:
                m = Categorical(logits=lm_length_logits)
                reward = baseline_loss.sum().detach().float().cpu() - masked_lm_loss.sum().detach().float().cpu()
                samples = torch.argmax(lm_length_logits, dim=-1)
                rl_loss = - m.log_prob(samples).sum() * reward * self.config.rl_loss_coef
                if self.config.reward_negative:
                    rl_loss = - rl_loss
                print("rl loss", rl_loss)
            
   
            masked_lm_loss = masked_lm_loss +  length_loss + rl_loss
            print("length loss: ", length_loss)
            print("total loss: ", masked_lm_loss)

            

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return LEDSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            encoder_global_attentions=outputs.encoder_global_attentions,
        )


    def prepare_inputs_for_generation(
        self, decoder_input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        result =  {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }
        kwargs = {k.replace("decoder_", ""):v for k, v in kwargs.items()}
        result.update(kwargs)

        return result

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past



class LEDForConditionalGenerationLengthClassification(LEDPreTrainedModel):
    base_model_prefix = "led"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]  

    def __init__(self, config: LEDConfig):
        super().__init__(config)
        self.led = LEDModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.led.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.led.shared.num_embeddings, bias=False)
        length_hidden_size = 200
        self.length_bucket_count = 6
        # self.length_class_map = {0: 10, 1:20, 2: 30, 3:40, 4:50, 5:60}
        # self.length_class_map_indexed = torch.nn.Parameter(torch.tensor([10., 20., 30., 40., 50., 60.]))

        self.lm_length = nn.Sequential(
            nn.Linear(config.d_model, length_hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(length_hidden_size, self.length_bucket_count, bias=True)
        )

        self.init_weights()

    def get_encoder(self):
        return self.led.get_encoder()

    def get_decoder(self):
        return self.led.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        global_attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        length_labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        Conditional generation example::

            >>> from transformers import LEDTokenizer, LEDForConditionalGeneration
            >>> tokenizer = LEDTokenizer.from_pretrained('allenai/led-base-16384')
            >>> TXT = "My friends are <mask> but they eat too many carbs."

            >>> model = LEDForConditionalGeneration.from_pretrained('allenai/led-base-16384')
            >>> input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
            >>> logits = model(input_ids).logits

            >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
            >>> probs = logits[0, masked_index].softmax(dim=0)
            >>> values, predictions = probs.topk(5)

            >>> tokenizer.decode(predictions).split()
        """
        # print("Cond gen input", input_ids)
        # print("Cond gen decoder input ids", decoder_input_ids)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        if encoder_outputs is None:
            encoder_outputs = self.led.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
        # t = encoder_outputs[0]
        # avg_encoder = (t * torch.ne(input_ids, self.led.decoder.padding_idx).int().unsqueeze(2).repeat(1, 1, self.config.d_model)).sum(1) / torch.ne(input_ids, self.led.decoder.padding_idx).int().sum(-1).unsqueeze(1).repeat(1,self.config.d_model)
        
        cls_encoder = encoder_outputs[0][:,0,:]
        lm_length_logits = self.lm_length(cls_encoder)
        
        if self.led.decoder.embed_positions.sinpostype != self.config.sinpostype:
            print(f"Warning: model.led.decoder.embed_positions.sinpostype is being changed from {self.led.decoder.embed_positions.sinpostype} to  {self.config.sinpostype}")
            self.led.decoder.embed_positions.sinpostype = self.config.sinpostype
        
        desired_length_label = torch.argmax(lm_length_logits, dim=-1)
        desired_length = self.length_class_map_indexed[desired_length_label]
        
        if labels is not None:
            # training time
            desired_length = self.length_class_map_indexed[length_labels]
        else:
            # test time --> desired_length is already predicted length
            pass

        outputs = self.led(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            global_attention_mask=global_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            length=desired_length,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss, length_loss = None, None
        if labels is not None:
            
            if not self.config.length_loss_only:                
                loss_fct = CrossEntropyLoss()
                masked_lm_loss = self.config.generation_loss_coef * loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
                print("Crosss entropy: ", masked_lm_loss)
            if not self.config.generation_loss_only:
                length_loss_fct = CrossEntropyLoss()
                print("lm_length_logits shape: ", lm_length_logits.shape)
                print("length_labels shape: ", length_labels.shape)
                length_loss = self.config.length_loss_coef * length_loss_fct(lm_length_logits.view(-1, self.length_bucket_count), length_labels.view(-1))
                
                
                if masked_lm_loss is None:
                    masked_lm_loss = length_loss
                else:
                    masked_lm_loss += length_loss
                print("length loss: ", length_loss)

            print("total loss: ", masked_lm_loss)

            

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return LEDSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            encoder_global_attentions=outputs.encoder_global_attentions,
        )


    def prepare_inputs_for_generation(
        self, decoder_input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        result =  {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }
        kwargs = {k.replace("decoder_", ""):v for k, v in kwargs.items()}
        result.update(kwargs)

        return result

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past





class LEDDecoder(LEDPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`LEDDecoderLayer`

    Args:
        config: LEDConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: LEDConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_decoder_position_embeddings

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        # self.embed_positions = LEDLearnedPositionalEmbedding(
        #     self.max_target_positions,
        #     config.d_model,
        #     self.padding_idx,
        # )
        self.embed_positions = SinusoidalPositionalEmbedding(
            embedding_dim=config.d_model,
            padding_idx=self.padding_idx,
            init_size=self.max_target_positions + 1 + self.padding_idx
        )

        self.layers = nn.ModuleList([LEDDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.init_weights()

    
    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs):
        """
        Implement in subclasses of [`PreTrainedModel`] for custom behavior to prepare inputs in the generate method.
        """
        return {"input_ids": input_ids}

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        length=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.LEDTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            global_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to decide the attention given on each token, local attention or global attention. Tokens with
                global attention attends to all other tokens, and all other tokens attend to them. This is important
                for task-specific finetuning because it makes the model more flexible at representing the task. For
                example, for classification, the <s> token should be given global attention. For QA, all question
                tokens should also have global attention. Please refer to the `Longformer paper
                <https://arxiv.org/abs/2004.05150>`__ for more details. Mask values selected in ``[0, 1]``:

                - 0 for local attention (a sliding window attention),
                - 1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, encoder_sequence_length)`, `optional`):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]` of length :obj:`config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up
                decoding.

                If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None and combined_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            combined_attention_mask = combined_attention_mask + _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        # print("Encoder hidden state size: ", encoder_hidden_states.shape)
        positions = self.embed_positions(input_ids, length=length)
        # positions = self.embed_positions(input_shape, past_key_values_length)


        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False):
                if use_cache:
                    raise ValueError(
                        "When using `gradient_checkpointing`, make sure that `use_cache=False` and `config.use_cache=False`."
                    )

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    combined_attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=combined_attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )



class LEDDecoderPosition(LEDPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`LEDDecoderLayer`

    Args:
        config: LEDConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: LEDConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_decoder_position_embeddings

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.token_type_embeddings = nn.Embedding(3, config.d_model)

        self.embed_positions = SinusoidalPositionalEmbeddingPosition(
            embedding_dim=config.d_model,
            padding_idx=self.padding_idx,
            init_size=self.max_target_positions + 1 + self.padding_idx
        )

        self.layers = nn.ModuleList([LEDDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.init_weights()

    
    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs):
        """
        Implement in subclasses of [`PreTrainedModel`] for custom behavior to prepare inputs in the generate method.
        """
        return {"input_ids": input_ids}

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        length_positions=None,
        segment_ids = None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.LEDTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            global_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to decide the attention given on each token, local attention or global attention. Tokens with
                global attention attends to all other tokens, and all other tokens attend to them. This is important
                for task-specific finetuning because it makes the model more flexible at representing the task. For
                example, for classification, the <s> token should be given global attention. For QA, all question
                tokens should also have global attention. Please refer to the `Longformer paper
                <https://arxiv.org/abs/2004.05150>`__ for more details. Mask values selected in ``[0, 1]``:

                - 0 for local attention (a sliding window attention),
                - 1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, encoder_sequence_length)`, `optional`):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]` of length :obj:`config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up
                decoding.

                If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            segment_embeds:
                Segments ids indicating the segments. for CORWA paragraph generation task, there are three possible segments 
                - prefix span, span, suffix span 
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None and combined_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            combined_attention_mask = combined_attention_mask + _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        # print("Encoder hidden state size: ", encoder_hidden_states.shape)
        positions = self.embed_positions(input_ids, minuspos=length_positions)
        segment_embeds = self.token_type_embeddings(segment_ids)
        # positions = self.embed_positions(input_shape, past_key_values_length)


        hidden_states = inputs_embeds + positions + segment_embeds
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False):
                if use_cache:
                    raise ValueError(
                        "When using `gradient_checkpointing`, make sure that `use_cache=False` and `config.use_cache=False`."
                    )

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    combined_attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=combined_attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )
