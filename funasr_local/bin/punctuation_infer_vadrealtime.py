#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import sys
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import Any
from typing import List

import numpy as np
import torch
from typeguard import check_argument_types

from funasr_local.datasets.preprocessor import CodeMixTokenizerCommonPreprocessor
from funasr_local.utils.cli_utils import get_commandline_args
from funasr_local.tasks.punctuation import PunctuationTask
from funasr_local.torch_utils.device_funcs import to_device
from funasr_local.torch_utils.forward_adaptor import ForwardAdaptor
from funasr_local.torch_utils.set_all_random_seed import set_all_random_seed
from funasr_local.utils import config_argparse
from funasr_local.utils.types import str2triple_str
from funasr_local.utils.types import str_or_none
from funasr_local.datasets.preprocessor import split_to_mini_sentence


class Text2Punc:

    def __init__(
        self,
        train_config: Optional[str],
        model_file: Optional[str],
        device: str = "cpu",
        dtype: str = "float32",
    ):
        #  Build Model
        model, train_args = PunctuationTask.build_model_from_file(train_config, model_file, device)
        self.device = device
        # Wrape model to make model.nll() data-parallel
        self.wrapped_model = ForwardAdaptor(model, "inference")
        self.wrapped_model.to(dtype=getattr(torch, dtype)).to(device=device).eval()
        # logging.info(f"Model:\n{model}")
        self.punc_list = train_args.punc_list
        self.period = 0
        for i in range(len(self.punc_list)):
            if self.punc_list[i] == ",":
                self.punc_list[i] = "，"
            elif self.punc_list[i] == "?":
                self.punc_list[i] = "？"
            elif self.punc_list[i] == "。":
                self.period = i
        self.preprocessor = CodeMixTokenizerCommonPreprocessor(
            train=False,
            token_type=train_args.token_type,
            token_list=train_args.token_list,
            bpemodel=train_args.bpemodel,
            text_cleaner=train_args.cleaner,
            g2p_type=train_args.g2p,
            text_name="text",
            non_linguistic_symbols=train_args.non_linguistic_symbols,
        )
        print("start decoding!!!")

    @torch.no_grad()
    def __call__(self, text: Union[list, str], cache: list, split_size=20):
        if cache is not None and len(cache) > 0:
            precache = "".join(cache)
        else:
            precache = ""
            cache = []
        data = {"text": precache + text}
        result = self.preprocessor(data=data, uid="12938712838719")
        split_text = self.preprocessor.pop_split_text_data(result)
        mini_sentences = split_to_mini_sentence(split_text, split_size)
        mini_sentences_id = split_to_mini_sentence(data["text"], split_size)
        assert len(mini_sentences) == len(mini_sentences_id)
        cache_sent = []
        cache_sent_id = torch.from_numpy(np.array([], dtype='int32'))
        sentence_punc_list = []
        sentence_words_list= []
        cache_pop_trigger_limit = 200
        skip_num = 0
        for mini_sentence_i in range(len(mini_sentences)):
            mini_sentence = mini_sentences[mini_sentence_i]
            mini_sentence_id = mini_sentences_id[mini_sentence_i]
            mini_sentence = cache_sent + mini_sentence
            mini_sentence_id = np.concatenate((cache_sent_id, mini_sentence_id), axis=0)
            data = {
                "text": torch.unsqueeze(torch.from_numpy(mini_sentence_id), 0),
                "text_lengths": torch.from_numpy(np.array([len(mini_sentence_id)], dtype='int32')),
                "vad_indexes": torch.from_numpy(np.array([len(cache)], dtype='int32')),
            }
            data = to_device(data, self.device)
            y, _ = self.wrapped_model(**data)
            _, indices = y.view(-1, y.shape[-1]).topk(1, dim=1)
            punctuations = indices
            if indices.size()[0] != 1:
                punctuations = torch.squeeze(indices)
            assert punctuations.size()[0] == len(mini_sentence)

            # Search for the last Period/QuestionMark as cache
            if mini_sentence_i < len(mini_sentences) - 1:
                sentenceEnd = -1
                last_comma_index = -1
                for i in range(len(punctuations) - 2, 1, -1):
                    if self.punc_list[punctuations[i]] == "。" or self.punc_list[punctuations[i]] == "？":
                        sentenceEnd = i
                        break
                    if last_comma_index < 0 and self.punc_list[punctuations[i]] == "，":
                        last_comma_index = i
    
                if sentenceEnd < 0 and len(mini_sentence) > cache_pop_trigger_limit and last_comma_index >= 0:
                    # The sentence it too long, cut off at a comma.
                    sentenceEnd = last_comma_index
                    punctuations[sentenceEnd] = self.period
                cache_sent = mini_sentence[sentenceEnd + 1:]
                cache_sent_id = mini_sentence_id[sentenceEnd + 1:]
                mini_sentence = mini_sentence[0:sentenceEnd + 1]
                punctuations = punctuations[0:sentenceEnd + 1]

            punctuations_np = punctuations.cpu().numpy()
            sentence_punc_list += [self.punc_list[int(x)] for x in punctuations_np]
            sentence_words_list += mini_sentence

        assert len(sentence_punc_list) == len(sentence_words_list)
        words_with_punc = []
        sentence_punc_list_out = []
        for i in range(0, len(sentence_words_list)):
            if i > 0:
                if len(sentence_words_list[i][0].encode()) == 1 and len(sentence_words_list[i - 1][-1].encode()) == 1:
                    sentence_words_list[i] = " " + sentence_words_list[i]
            if skip_num < len(cache):
                skip_num += 1
            else:
                words_with_punc.append(sentence_words_list[i])
            if skip_num >= len(cache):
                sentence_punc_list_out.append(sentence_punc_list[i])
                if sentence_punc_list[i] != "_":
                    words_with_punc.append(sentence_punc_list[i])
        sentence_out = "".join(words_with_punc)

        sentenceEnd = -1
        for i in range(len(sentence_punc_list) - 2, 1, -1):
            if sentence_punc_list[i] == "。" or sentence_punc_list[i] == "？":
               sentenceEnd = i
               break
        cache_out = sentence_words_list[sentenceEnd + 1 :]
        if sentence_out[-1] in self.punc_list:
            sentence_out = sentence_out[:-1]
            sentence_punc_list_out[-1] = "_"
        return sentence_out, sentence_punc_list_out, cache_out


def inference(
    batch_size: int,
    dtype: str,
    ngpu: int,
    seed: int,
    num_workers: int,
    output_dir: str,
    log_level: Union[int, str],
    train_config: Optional[str],
    model_file: Optional[str],
    key_file: Optional[str] = None,
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]] = None,
    raw_inputs: Union[List[Any], bytes, str] = None,
    cache: List[Any] = None,
    param_dict: dict = None,
    **kwargs,
):
    inference_pipeline = inference_modelscope(
        output_dir=output_dir,
        batch_size=batch_size,
        dtype=dtype,
        ngpu=ngpu,
        seed=seed,
        num_workers=num_workers,
        log_level=log_level,
        key_file=key_file,
        train_config=train_config,
        model_file=model_file,
        param_dict=param_dict,
        **kwargs,
    )
    return inference_pipeline(data_path_and_name_and_type, raw_inputs, cache)


def inference_modelscope(
    batch_size: int,
    dtype: str,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: Union[int, str],
    #cache: list,
    key_file: Optional[str],
    train_config: Optional[str],
    model_file: Optional[str],
    output_dir: Optional[str] = None,
    param_dict: dict = None,
    **kwargs,
):
    assert check_argument_types()
    ncpu = kwargs.get("ncpu", 1)
    torch.set_num_threads(ncpu)

    if ngpu >= 1 and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)
    text2punc = Text2Punc(train_config, model_file, device)

    def _forward(
        data_path_and_name_and_type,
        raw_inputs: Union[List[Any], bytes, str] = None,
        output_dir_v2: Optional[str] = None,
        cache: List[Any] = None,
        param_dict: dict = None,
    ):
        results = []
        split_size = 10
        cache_in = param_dict["cache"]
        if raw_inputs != None:
            line = raw_inputs.strip()
            key = "demo"
            if line == "":
                item = {'key': key, 'value': ""}
                results.append(item)
                return results
            result, _, cache = text2punc(line, cache_in)
            param_dict["cache"] = cache
            item = {'key': key, 'value': result}
            results.append(item)
            return results

        return results

    return _forward


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Punctuation inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument("--output_dir", type=str, required=False)
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument("--data_path_and_name_and_type", type=str2triple_str, action="append", required=False)
    group.add_argument("--raw_inputs", type=str, required=False)
    group.add_argument("--cache", type=list, required=False)
    group.add_argument("--param_dict", type=dict, required=False)
    group.add_argument("--key_file", type=str_or_none)

    group = parser.add_argument_group("The model configuration related")
    group.add_argument("--train_config", type=str)
    group.add_argument("--model_file", type=str)

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    # kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
