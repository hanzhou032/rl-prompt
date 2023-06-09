import sys
sys.path.append('..')
import hydra
from typing import Optional, Tuple, List
import numpy as np
import torch
from transformers import (AutoTokenizer,
                          GPT2LMHeadModel,
                          AutoModelForMaskedLM)
from transformers import T5Tokenizer, T5ForConditionalGeneration

SUPPORTED_LEFT_TO_RIGHT_LMS = ['distilgpt2', 'gpt2', 'gpt2-medium',
                               'gpt2-large', 'gpt2-xl']
SUPPORTED_MASK_LMS = ['distilroberta-base', 'roberta-base', 'roberta-large']


class PromptedClassificationEvaluator:
    def __init__(
        self,
        task_lm: str,
        is_mask_lm: Optional[bool],
        num_classes: int,
        verbalizers: List[str],
        template: Optional[str],
        prompt: str,
        dataset: str
    ):
        super().__init__()
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.task_lm = task_lm
        print("Task LM:", self.task_lm)
        if is_mask_lm is None: 
            # If False, then treat as left-to-right LM
            self.is_mask_lm = True if 'bert' in self.task_lm else False
        else:
            self.is_mask_lm = is_mask_lm  
        if self.is_mask_lm:
            assert self.task_lm in SUPPORTED_MASK_LMS
            self._tokenizer = AutoTokenizer.from_pretrained(self.task_lm,
                                truncation_side="left")
            self._generator = (AutoModelForMaskedLM
                               .from_pretrained(self.task_lm)
                               .to(self.device))
        else:
            self._generator = (T5ForConditionalGeneration.from_pretrained(self.task_lm).to(
                self.device))
            self._tokenizer = AutoTokenizer.from_pretrained(self.task_lm, use_fast=False)
        self.num_classes = num_classes
        self.verbalizers = verbalizers

        self.verbalizer_ids = [self._tokenizer.convert_tokens_to_ids(v)
                               for v in self.verbalizers]
        if template is None:
            self.template = self.load_default_template()  # prompt templates
        else:
            self.template = template
        self.prompt = prompt

    # Adapted from
    # https://huggingface.co/docs/transformers/v4.21.1/en/task_summary#masked-language-modeling
    def _get_mask_token_index(self, input_ids: torch.Tensor) -> np.ndarray:
        mask_token_index = torch.where(
            input_ids == self._tokenizer.mask_token_id)[1]
        return mask_token_index

    def load_default_template(self) -> List[str]:
        if self.is_mask_lm:
            template = "{sentence_1} {prompt} <mask> ."
        else:
            # Template for left-to-right LMs like GPT-2
            # template = "{sentence_1} {prompt}"
            template_dict = {
                "xnli": [
                        " {prompt}. In this task, the goal is to predict textual entailment with 'yes' 'maybe' 'no'. sentence A implies sentence B entailment: yes; sentence A is neutral to sentence B entailment: maybe; sentence A contradicts sentence B entailment: no. Sentence A: {sentence_1}, Sentence B: {sentence_2}, Entailment: ", # 0.6984
                        "{sentence_1} {sentence_2} {prompt}: ", # 0.503
                        " {prompt} {sentence_1} {sentence_2} Entailment: ",# 0.565
                        "Review:\n{sentence_1}\nIs this movie review sentence negative or positive?\n ",
                        " sentence A implies sentence B entailment: yes; sentence A is neutral to sentence B entailment: maybe; sentence A contradicts sentence B entailment: no. {sentence_1} {sentence_2} Entailment: ",
                        " {prompt}. {sentence_1}. Based on the paragraph above, can we conclude that {sentence_2}? Options: yes neither no. Answer: ",# 0.6477
                        " {prompt}. Read the following and determine if the hypothesis can be inferred from the promise: Premise: {sentence_1}. Hypothesis: {sentence_2}? Options: Yes, Maybe, No. Answer: ", # 0.6469
                        " {prompt} erreichbar terminate Exist dislike NOT. In this task, the goal is to predict textual entailment with 'yes' 'maybe' 'no'. sentence A implies sentence B entailment: yes; sentence A is neutral to sentence B entailment: maybe; sentence A contradicts sentence B entailment: no. Sentence A: {sentence_1}, Sentence B: {sentence_2}, Entailment: ",
                        " {prompt} acela Oracle existed Different localitate. In this task, the goal is to predict textual entailment with 'yes' 'maybe' 'no'. sentence A implies sentence B entailment: yes; sentence A is neutral to sentence B entailment: maybe; sentence A contradicts sentence B entailment: no. Sentence A: {sentence_1}, Sentence B: {sentence_2}, Entailment: "
                ],
                "mnli": [
                        " {prompt}. In this task, the goal is to predict textual entailment with 'yes' 'maybe' 'no'. sentence A implies sentence B entailment: yes; sentence A is neutral to sentence B entailment: maybe; sentence A contradicts sentence B entailment: no. Sentence A: {sentence_1}, Sentence B: {sentence_2}, Entailment: ", # 0.6984
                        "{prompt}{sentence_1}\n\nDoes it follow that \"{sentence_2}\"?\nyes, neither, no. ", # flan template
                ],
                "anli": [
                        " {prompt} {sentence_1} {sentence_2} Entailment: ",
                        " {prompt}. In this task, the goal is to predict textual entailment with 'yes' 'maybe' 'no'. sentence A implies sentence B entailment: yes; sentence A is neutral to sentence B entailment: maybe; sentence A contradicts sentence B entailment: no. Sentence A: {sentence_1}, Sentence B: {sentence_2}, Entailment: ", 
                        # flan
                        "{sentence_1}\n\nBased on the paragraph above can we conclude that \"{sentence_2}\"?\n\nyes, maybe, no: ",
                        "{sentence_1}\n\nBased on that paragraph can we conclude that this sentence is true?\n{sentence_2}\n\nyes, maybe, no: ",
                        "{sentence_1}\n\nCan we draw the following conclusion?\n{sentence_2}\n\nyes, maybe, no: ",
                        "{sentence_1}\nDoes this next sentence follow, given the preceding text?\n{sentence_2}\n\nyes, maybe, no: ",
                        "{sentence_1}\nCan we infer the following?\n{sentence_2}\n\nyes, maybe, no: ",
                        "Read the following paragraph and determine if the hypothesis is true:\n\n{sentence_1}\n\nHypothesis: {sentence_2}n\nyes, maybe, no: ",
                        "Read the text and determine if the sentence is true:\n\n{sentence_1}\n\nSentence: {sentence_2}n\nyes, maybe, no: ",
                        "Can we draw the following hypothesis from the context? \n\nContext:\n\n{sentence_1}\n\nHypothesis: {sentence_2}n\nyes, maybe, no:, ",
                        "Determine if the sentence is true based on the text below:\n{sentence_2}\n\n{sentence_1}\nyes, maybe, no:, ",
                        "Generate a context and a hypothesis. Context: {sentence_1}\n\nHypothesis: {sentence_2}"
                ],
                "rte": [
                        " {prompt}. Sentence 1: {sentence_1}, Sentence 2: {sentence_2}, Textual Entailment: ",
                        " {prompt}. {sentence_1}\n\nBased on the paragraph above can we conclude that {sentence_2}?\n\n yes, no",
                ],
                "sst2": [
                        " {prompt}. Sentence: {sentence_1}, Sentiment: ",
                        " {prompt}. In this task, the goal is to predict the sentiment of the sentence. The task is to classify ”great” if the sentence is positive or as ”terrible” if the sentence is negative. Sentence: {sentence_1}, Sentiment: ",
                        " {prompt}. In this task, the goal is to predict the sentiment of the sentence. The task is to classify ”positive” if the sentence is positive or as ”negative” if the sentence is negative. Sentence: {sentence_1}, Sentiment: ",
                        "Review:\n{sentence_1}\nIs this movie review sentence negative or positive?\n ",
                ],
                "mrpc": [
                        " {prompt}. Sentence 1: {sentence_1}, Sentence 2: {sentence_2}, Semantically Equivalent: ",
                        " {prompt}. In this task, the goal is to determine two sentences mean the same thing or not. Sentence 1 mean the same thing as Sentence 2, Paraphrase: No; Sentence 1 mean the different thing as Sentence 2, Paraphrase: Yes; Sentence 1: {sentence_1}, Sentence 2: {sentence_2}, Paraphrase: ",
                        " {prompt}. Here are two sentences:\n{sentence_1}\n{sentence_2}\nDo they have the same meaning?\n ",
                ],
                "qnli": [
                        " {prompt}. Question: {sentence_1}, Sentence: {sentence_2}, Entailment: ",
                        " {prompt}. Does the sentence {sentence_1} answer the question {sentence_2}, yes no, ", # flan template
                ],
                "qqp": [
                        " {prompt}. Sentence 1: {sentence_1}, Sentence 2: {sentence_2}, Semantically Equivalent: ",
                        " {prompt}. Sentence 1: {sentence_1}, Sentence 2: {sentence_2}, Paraphrase: ",
                        " {prompt}. {sentence_1}\n{sentence_2}\nWould you say that these questions are the same?\n ",

                ],
                "boolq": [
                        " {prompt}. Passage: {sentence_2}, Question: {sentence_1}: ",
                ],
                "indonlp/NusaX-senti": [
                        " {prompt}. Sentence: {sentence_1}, Sentiment: ",
                ],
                "yelp_polarity": [
                        " {prompt}. Sentence: {sentence_1}, Sentiment: ",
                ],
                "ag_news": [
                        " {prompt}. Classify the news articles into the categories of World, Sports, Business, and Technology. {sentence_1}: ",
                        "{prompt}\n\n{sentence_1}\n\nWhich topic is this article about?\nWorld, Sports, Business, Technology, "
                ],
                "SetFit/sst5": [
                        " {prompt}. Sentence: {sentence_1}, Sentiment: ",
                        " {prompt}. In this task, you are given sentences from movie reviews. Based on the given review, classify it to one of the five classes: (1) terrible, (2) bad, (3) okay, (4) good, and (5) great. Sentence: {sentence_1}, Sentiment: ",
                        ]
            }
        template = template_dict[self.dataset][0]
        return template

    @torch.no_grad()
    def _get_logits(
        self,
        texts: List[str]
    ) -> torch.Tensor:
        # for MLM, add mask token
        batch_size = len(texts)
        encoded_inputs = self._tokenizer(texts, padding='longest',
                                         truncation=True, return_tensors="pt",
                                         add_special_tokens=True)
        decoder_input_ids = (torch.ones((batch_size, 1))*torch.tensor(self._tokenizer.pad_token_id)).int()
        if self.is_mask_lm:
            # self.ensure_exactly_one_mask_token(encoded_inputs) TODO
            token_logits = self._generator(
                **encoded_inputs.to(self.device)).logits
            mask_token_indices = \
                self._get_mask_token_index(encoded_inputs['input_ids'])
            out_logits = token_logits[range(batch_size), mask_token_indices, :]
        else:
            # token_logits = self._generator(
            #     **encoded_inputs.to(self.device)).logits
            # input_lengths = encoded_inputs['attention_mask'].sum(dim=1)
            # out_logits = token_logits[range(batch_size), input_lengths - 1, :]
            token_logits = self._generator(input_ids=encoded_inputs['input_ids'].to(self.device), decoder_input_ids=decoder_input_ids.to(self.device)).logits
            token_logits = token_logits[:,0,:]

        return token_logits

    def _format_prompts(
        self,
        prompts: List[str],
        source_strs: List[str],
        source_2_strs: List[str]
    ) -> List[str]:
        return [self.template.format(sentence_1=s_1, sentence_2=s_2, prompt=prompt)
                for s_1, s_2, prompt in zip(source_strs, source_2_strs, prompts)]

    def forward(
        self,
        dataloader
    ) -> float:
        num_of_examples = dataloader.dataset.__len__()
        correct_sum = 0
        for i, batch in enumerate(dataloader):
            inputs = batch['source_texts']  # List
            inputs_2 = batch['source_2_texts']  # List
            targets = batch['class_labels']  # Tensor
            batch_size = targets.size(0)
            current_prompts = [self.prompt for _ in range(batch_size)]
            formatted_templates = self._format_prompts(current_prompts, inputs, inputs_2)
            if i == 0:
                print(formatted_templates)
            # print(formatted_templates)
            all_logits = self._get_logits(formatted_templates)
            class_probs = torch.softmax(all_logits[:, self.verbalizer_ids], -1)
            # Get labels
            predicted_labels = torch.argmax(class_probs, dim=-1)
            label_agreement = torch.where(
                targets.cuda() == predicted_labels, 1, 0)
            # Compute accuracy
            correct_sum += label_agreement.sum()
        accuracy = correct_sum/num_of_examples
        return accuracy
