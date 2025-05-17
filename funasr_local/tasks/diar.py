"""
Author: Speech Lab, Alibaba Group, China
SOND: Speaker Overlap-aware Neural Diarization for Multi-party Meeting Analysis
https://arxiv.org/abs/2211.10243
TOLD: A Novel Two-Stage Overlap-Aware Framework for Speaker Diarization
https://arxiv.org/abs/2303.05397
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
import yaml
from typeguard import check_argument_types
from typeguard import check_return_type

from funasr_local.datasets.collate_fn import CommonCollateFn
from funasr_local.datasets.preprocessor import CommonPreprocessor
from funasr_local.layers.abs_normalize import AbsNormalize
from funasr_local.layers.global_mvn import GlobalMVN
from funasr_local.layers.label_aggregation import LabelAggregate
from funasr_local.layers.utterance_mvn import UtteranceMVN
from funasr_local.models.e2e_diar_sond import DiarSondModel
from funasr_local.models.e2e_diar_eend_ola import DiarEENDOLAModel
from funasr_local.models.encoder.abs_encoder import AbsEncoder
from funasr_local.models.encoder.conformer_encoder import ConformerEncoder
from funasr_local.models.encoder.data2vec_encoder import Data2VecEncoder
from funasr_local.models.encoder.ecapa_tdnn_encoder import ECAPA_TDNN
from funasr_local.models.encoder.opennmt_encoders.ci_scorers import DotScorer, CosScorer
from funasr_local.models.encoder.opennmt_encoders.conv_encoder import ConvEncoder
from funasr_local.models.encoder.opennmt_encoders.fsmn_encoder import FsmnEncoder
from funasr_local.models.encoder.opennmt_encoders.self_attention_encoder import SelfAttentionEncoder
from funasr_local.models.encoder.resnet34_encoder import ResNet34Diar, ResNet34SpL2RegDiar
from funasr_local.models.encoder.rnn_encoder import RNNEncoder
from funasr_local.models.encoder.sanm_encoder import SANMEncoder, SANMEncoderChunkOpt
from funasr_local.models.encoder.transformer_encoder import TransformerEncoder
from funasr_local.models.frontend.abs_frontend import AbsFrontend
from funasr_local.models.frontend.default import DefaultFrontend
from funasr_local.models.frontend.fused import FusedFrontends
from funasr_local.models.frontend.s3prl import S3prlFrontend
from funasr_local.models.frontend.wav_frontend import WavFrontend
from funasr_local.models.frontend.wav_frontend import WavFrontendMel23
from funasr_local.models.frontend.windowing import SlidingWindow
from funasr_local.models.specaug.abs_specaug import AbsSpecAug
from funasr_local.models.specaug.specaug import SpecAug
from funasr_local.models.specaug.specaug import SpecAugLFR
from funasr_local.modules.eend_ola.encoder import EENDOLATransformerEncoder
from funasr_local.modules.eend_ola.encoder_decoder_attractor import EncoderDecoderAttractor
from funasr_local.tasks.abs_task import AbsTask
from funasr_local.torch_utils.initialize import initialize
from funasr_local.train.abs_espnet_model import AbsESPnetModel
from funasr_local.train.class_choices import ClassChoices
from funasr_local.train.trainer import Trainer
from funasr_local.utils.types import float_or_none
from funasr_local.utils.types import int_or_none
from funasr_local.utils.types import str2bool
from funasr_local.utils.types import str_or_none

frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        default=DefaultFrontend,
        sliding_window=SlidingWindow,
        s3prl=S3prlFrontend,
        fused=FusedFrontends,
        wav_frontend=WavFrontend,
        wav_frontend_mel23=WavFrontendMel23,
    ),
    type_check=AbsFrontend,
    default="default",
)
specaug_choices = ClassChoices(
    name="specaug",
    classes=dict(
        specaug=SpecAug,
        specaug_lfr=SpecAugLFR,
    ),
    type_check=AbsSpecAug,
    default=None,
    optional=True,
)
normalize_choices = ClassChoices(
    "normalize",
    classes=dict(
        global_mvn=GlobalMVN,
        utterance_mvn=UtteranceMVN,
    ),
    type_check=AbsNormalize,
    default=None,
    optional=True,
)
label_aggregator_choices = ClassChoices(
    "label_aggregator",
    classes=dict(
        label_aggregator=LabelAggregate
    ),
    type_check=torch.nn.Module,
    default=None,
    optional=True,
)
model_choices = ClassChoices(
    "model",
    classes=dict(
        sond=DiarSondModel,
        eend_ola=DiarEENDOLAModel,
    ),
    type_check=AbsESPnetModel,
    default="sond",
)
encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        conformer=ConformerEncoder,
        transformer=TransformerEncoder,
        rnn=RNNEncoder,
        sanm=SANMEncoder,
        san=SelfAttentionEncoder,
        fsmn=FsmnEncoder,
        conv=ConvEncoder,
        resnet34=ResNet34Diar,
        resnet34_sp_l2reg=ResNet34SpL2RegDiar,
        sanm_chunk_opt=SANMEncoderChunkOpt,
        data2vec_encoder=Data2VecEncoder,
        ecapa_tdnn=ECAPA_TDNN,
        eend_ola_transformer=EENDOLATransformerEncoder,
    ),
    type_check=torch.nn.Module,
    default="resnet34",
)
speaker_encoder_choices = ClassChoices(
    "speaker_encoder",
    classes=dict(
        conformer=ConformerEncoder,
        transformer=TransformerEncoder,
        rnn=RNNEncoder,
        sanm=SANMEncoder,
        san=SelfAttentionEncoder,
        fsmn=FsmnEncoder,
        conv=ConvEncoder,
        sanm_chunk_opt=SANMEncoderChunkOpt,
        data2vec_encoder=Data2VecEncoder,
    ),
    type_check=AbsEncoder,
    default=None,
    optional=True
)
cd_scorer_choices = ClassChoices(
    "cd_scorer",
    classes=dict(
        san=SelfAttentionEncoder,
    ),
    type_check=AbsEncoder,
    default=None,
    optional=True,
)
ci_scorer_choices = ClassChoices(
    "ci_scorer",
    classes=dict(
        dot=DotScorer,
        cosine=CosScorer,
        conv=ConvEncoder,
    ),
    type_check=torch.nn.Module,
    default=None,
    optional=True,
)
# decoder is used for output (e.g. post_net in SOND)
decoder_choices = ClassChoices(
    "decoder",
    classes=dict(
        rnn=RNNEncoder,
        fsmn=FsmnEncoder,
    ),
    type_check=torch.nn.Module,
    default="fsmn",
)
# encoder_decoder_attractor is used for EEND-OLA
encoder_decoder_attractor_choices = ClassChoices(
    "encoder_decoder_attractor",
    classes=dict(
        eda=EncoderDecoderAttractor,
    ),
    type_check=torch.nn.Module,
    default="eda",
)


class DiarTask(AbsTask):
    # If you need more than 1 optimizer, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --frontend and --frontend_conf
        frontend_choices,
        # --specaug and --specaug_conf
        specaug_choices,
        # --normalize and --normalize_conf
        normalize_choices,
        # --label_aggregator and --label_aggregator_conf
        label_aggregator_choices,
        # --model and --model_conf
        model_choices,
        # --encoder and --encoder_conf
        encoder_choices,
        # --speaker_encoder and --speaker_encoder_conf
        speaker_encoder_choices,
        # --cd_scorer and cd_scorer_conf
        cd_scorer_choices,
        # --ci_scorer and ci_scorer_conf
        ci_scorer_choices,
        # --decoder and --decoder_conf
        decoder_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        # required = parser.get_default("required")
        # required += ["token_list"]

        group.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token",
        )
        group.add_argument(
            "--split_with_space",
            type=str2bool,
            default=True,
            help="whether to split text using <space>",
        )
        group.add_argument(
            "--seg_dict_file",
            type=str,
            default=None,
            help="seg_dict_file for text processing",
        )
        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )

        group.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
        )
        group.add_argument(
            "--token_type",
            type=str,
            default="char",
            choices=["char"],
            help="The text will be tokenized in the specified level token",
        )
        parser.add_argument(
            "--speech_volume_normalize",
            type=float_or_none,
            default=None,
            help="Scale the maximum amplitude to the given value.",
        )
        parser.add_argument(
            "--rir_scp",
            type=str_or_none,
            default=None,
            help="The file path of rir scp file.",
        )
        parser.add_argument(
            "--rir_apply_prob",
            type=float,
            default=1.0,
            help="THe probability for applying RIR convolution.",
        )
        parser.add_argument(
            "--cmvn_file",
            type=str_or_none,
            default=None,
            help="The file path of noise scp file.",
        )
        parser.add_argument(
            "--noise_scp",
            type=str_or_none,
            default=None,
            help="The file path of noise scp file.",
        )
        parser.add_argument(
            "--noise_apply_prob",
            type=float,
            default=1.0,
            help="The probability applying Noise adding.",
        )
        parser.add_argument(
            "--noise_db_range",
            type=str,
            default="13_15",
            help="The range of noise decibel level.",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_collate_fn(
            cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()
        # NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

    @classmethod
    def build_preprocess_fn(
            cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        if args.use_preprocessor:
            retval = CommonPreprocessor(
                train=train,
                token_type=args.token_type,
                token_list=args.token_list,
                bpemodel=None,
                non_linguistic_symbols=None,
                text_cleaner=None,
                g2p_type=None,
                split_with_space=args.split_with_space if hasattr(args, "split_with_space") else False,
                seg_dict_file=args.seg_dict_file if hasattr(args, "seg_dict_file") else None,
                # NOTE(kamo): Check attribute existence for backward compatibility
                rir_scp=args.rir_scp if hasattr(args, "rir_scp") else None,
                rir_apply_prob=args.rir_apply_prob
                if hasattr(args, "rir_apply_prob")
                else 1.0,
                noise_scp=args.noise_scp if hasattr(args, "noise_scp") else None,
                noise_apply_prob=args.noise_apply_prob
                if hasattr(args, "noise_apply_prob")
                else 1.0,
                noise_db_range=args.noise_db_range
                if hasattr(args, "noise_db_range")
                else "13_15",
                speech_volume_normalize=args.speech_volume_normalize
                if hasattr(args, "rir_scp")
                else None,
            )
        else:
            retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
            cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("speech", "profile", "binary_labels")
        else:
            # Recognition mode
            retval = ("speech", "profile")
        return retval

    @classmethod
    def optional_data_names(
            cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ()
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace):
        assert check_argument_types()
        if isinstance(args.token_list, str):
            with open(args.token_list, encoding="utf-8") as f:
                token_list = [line.rstrip() for line in f]

            # Overwriting token_list to keep it as "portable".
            args.token_list = list(token_list)
        elif isinstance(args.token_list, (tuple, list)):
            token_list = list(args.token_list)
        else:
            raise RuntimeError("token_list must be str or list")
        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size}")

        # 1. frontend
        if args.input_size is None:
            # Extract features in the model
            frontend_class = frontend_choices.get_class(args.frontend)
            if args.frontend == 'wav_frontend':
                frontend = frontend_class(cmvn_file=args.cmvn_file, **args.frontend_conf)
            else:
                frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()
        else:
            # Give features from data-loader
            args.frontend = None
            args.frontend_conf = {}
            frontend = None
            input_size = args.input_size

        # 2. Data augmentation for spectrogram
        if args.specaug is not None:
            specaug_class = specaug_choices.get_class(args.specaug)
            specaug = specaug_class(**args.specaug_conf)
        else:
            specaug = None

        # 3. Normalization layer
        if args.normalize is not None:
            normalize_class = normalize_choices.get_class(args.normalize)
            normalize = normalize_class(**args.normalize_conf)
        else:
            normalize = None

        # 4. Encoder
        encoder_class = encoder_choices.get_class(args.encoder)
        encoder = encoder_class(input_size=input_size, **args.encoder_conf)

        # 5. speaker encoder
        if getattr(args, "speaker_encoder", None) is not None:
            speaker_encoder_class = speaker_encoder_choices.get_class(args.speaker_encoder)
            speaker_encoder = speaker_encoder_class(**args.speaker_encoder_conf)
        else:
            speaker_encoder = None

        # 6. CI & CD scorer
        if getattr(args, "ci_scorer", None) is not None:
            ci_scorer_class = ci_scorer_choices.get_class(args.ci_scorer)
            ci_scorer = ci_scorer_class(**args.ci_scorer_conf)
        else:
            ci_scorer = None

        if getattr(args, "cd_scorer", None) is not None:
            cd_scorer_class = cd_scorer_choices.get_class(args.cd_scorer)
            cd_scorer = cd_scorer_class(**args.cd_scorer_conf)
        else:
            cd_scorer = None

        # 7. Decoder
        decoder_class = decoder_choices.get_class(args.decoder)
        decoder = decoder_class(**args.decoder_conf)

        if getattr(args, "label_aggregator", None) is not None:
            label_aggregator_class = label_aggregator_choices.get_class(args.label_aggregator)
            label_aggregator = label_aggregator_class(**args.label_aggregator_conf)
        else:
            label_aggregator = None

        # 9. Build model
        model_class = model_choices.get_class(args.model)
        model = model_class(
            vocab_size=vocab_size,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            label_aggregator=label_aggregator,
            encoder=encoder,
            speaker_encoder=speaker_encoder,
            ci_scorer=ci_scorer,
            cd_scorer=cd_scorer,
            decoder=decoder,
            token_list=token_list,
            **args.model_conf,
        )

        # 10. Initialize
        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model

    # ~~~~~~~~~ The methods below are mainly used for inference ~~~~~~~~~
    @classmethod
    def build_model_from_file(
            cls,
            config_file: Union[Path, str] = None,
            model_file: Union[Path, str] = None,
            cmvn_file: Union[Path, str] = None,
            device: Union[str, torch.device] = "cpu",
    ):
        """Build model from the files.

        This method is used for inference or fine-tuning.

        Args:
            config_file: The yaml file saved when training.
            model_file: The model file saved when training.
            cmvn_file: The cmvn file for front-end
            device: Device type, "cpu", "cuda", or "cuda:N".

        """
        assert check_argument_types()
        if config_file is None:
            assert model_file is not None, (
                "The argument 'model_file' must be provided "
                "if the argument 'config_file' is not specified."
            )
            config_file = Path(model_file).parent / "config.yaml"
        else:
            config_file = Path(config_file)

        with config_file.open("r", encoding="utf-8") as f:
            args = yaml.safe_load(f)
        if cmvn_file is not None:
            args["cmvn_file"] = cmvn_file
        args = argparse.Namespace(**args)
        model = cls.build_model(args)
        if not isinstance(model, AbsESPnetModel):
            raise RuntimeError(
                f"model must inherit {AbsESPnetModel.__name__}, but got {type(model)}"
            )
        model.to(device)
        model_dict = dict()
        model_name_pth = None
        if model_file is not None:
            logging.info("model_file is {}".format(model_file))
            if device == "cuda":
                device = f"cuda:{torch.cuda.current_device()}"
            model_dir = os.path.dirname(model_file)
            model_name = os.path.basename(model_file)
            if "model.ckpt-" in model_name or ".bin" in model_name:
                if ".bin" in model_name:
                    model_name_pth = os.path.join(model_dir, model_name.replace('.bin', '.pb'))
                else:
                    model_name_pth = os.path.join(model_dir, "{}.pb".format(model_name))
                if os.path.exists(model_name_pth):
                    logging.info("model_file is load from pth: {}".format(model_name_pth))
                    model_dict = torch.load(model_name_pth, map_location=device)
                else:
                    model_dict = cls.convert_tf2torch(model, model_file)
                model.load_state_dict(model_dict)
            else:
                model_dict = torch.load(model_file, map_location=device)
        model_dict = cls.fileter_model_dict(model_dict, model.state_dict())
        model.load_state_dict(model_dict)
        if model_name_pth is not None and not os.path.exists(model_name_pth):
            torch.save(model_dict, model_name_pth)
            logging.info("model_file is saved to pth: {}".format(model_name_pth))

        return model, args

    @classmethod
    def fileter_model_dict(cls, src_dict: dict, dest_dict: dict):
        from collections import OrderedDict
        new_dict = OrderedDict()
        for key, value in src_dict.items():
            if key in dest_dict:
                new_dict[key] = value
            else:
                logging.info("{} is no longer needed in this model.".format(key))
        for key, value in dest_dict.items():
            if key not in new_dict:
                logging.warning("{} is missed in checkpoint.".format(key))
        return new_dict

    @classmethod
    def convert_tf2torch(
            cls,
            model,
            ckpt,
    ):
        logging.info("start convert tf model to torch model")
        from funasr_local.modules.streaming_utils.load_fr_tf import load_tf_dict
        var_dict_tf = load_tf_dict(ckpt)
        var_dict_torch = model.state_dict()
        var_dict_torch_update = dict()
        # speech encoder
        if model.encoder is not None:
            var_dict_torch_update_local = model.encoder.convert_tf2torch(var_dict_tf, var_dict_torch)
            var_dict_torch_update.update(var_dict_torch_update_local)
        # speaker encoder
        if model.speaker_encoder is not None:
            var_dict_torch_update_local = model.speaker_encoder.convert_tf2torch(var_dict_tf, var_dict_torch)
            var_dict_torch_update.update(var_dict_torch_update_local)
        # cd scorer
        if model.cd_scorer is not None:
            var_dict_torch_update_local = model.cd_scorer.convert_tf2torch(var_dict_tf, var_dict_torch)
            var_dict_torch_update.update(var_dict_torch_update_local)
        # ci scorer
        if model.ci_scorer is not None:
            var_dict_torch_update_local = model.ci_scorer.convert_tf2torch(var_dict_tf, var_dict_torch)
            var_dict_torch_update.update(var_dict_torch_update_local)
        # decoder
        if model.decoder is not None:
            var_dict_torch_update_local = model.decoder.convert_tf2torch(var_dict_tf, var_dict_torch)
            var_dict_torch_update.update(var_dict_torch_update_local)

        return var_dict_torch_update


class EENDOLADiarTask(AbsTask):
    # If you need more than 1 optimizer, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --frontend and --frontend_conf
        frontend_choices,
        # --specaug and --specaug_conf
        model_choices,
        # --encoder and --encoder_conf
        encoder_choices,
        # --speaker_encoder and --speaker_encoder_conf
        encoder_decoder_attractor_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        # required = parser.get_default("required")
        # required += ["token_list"]

        group.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token",
        )
        group.add_argument(
            "--split_with_space",
            type=str2bool,
            default=True,
            help="whether to split text using <space>",
        )
        group.add_argument(
            "--seg_dict_file",
            type=str,
            default=None,
            help="seg_dict_file for text processing",
        )
        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )

        group.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
        )
        group.add_argument(
            "--token_type",
            type=str,
            default="char",
            choices=["char"],
            help="The text will be tokenized in the specified level token",
        )
        parser.add_argument(
            "--speech_volume_normalize",
            type=float_or_none,
            default=None,
            help="Scale the maximum amplitude to the given value.",
        )
        parser.add_argument(
            "--rir_scp",
            type=str_or_none,
            default=None,
            help="The file path of rir scp file.",
        )
        parser.add_argument(
            "--rir_apply_prob",
            type=float,
            default=1.0,
            help="THe probability for applying RIR convolution.",
        )
        parser.add_argument(
            "--cmvn_file",
            type=str_or_none,
            default=None,
            help="The file path of noise scp file.",
        )
        parser.add_argument(
            "--noise_scp",
            type=str_or_none,
            default=None,
            help="The file path of noise scp file.",
        )
        parser.add_argument(
            "--noise_apply_prob",
            type=float,
            default=1.0,
            help="The probability applying Noise adding.",
        )
        parser.add_argument(
            "--noise_db_range",
            type=str,
            default="13_15",
            help="The range of noise decibel level.",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_collate_fn(
            cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()
        # NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

    @classmethod
    def build_preprocess_fn(
            cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        # if args.use_preprocessor:
        #     retval = CommonPreprocessor(
        #         train=train,
        #         token_type=args.token_type,
        #         token_list=args.token_list,
        #         bpemodel=None,
        #         non_linguistic_symbols=None,
        #         text_cleaner=None,
        #         g2p_type=None,
        #         split_with_space=args.split_with_space if hasattr(args, "split_with_space") else False,
        #         seg_dict_file=args.seg_dict_file if hasattr(args, "seg_dict_file") else None,
        #         # NOTE(kamo): Check attribute existence for backward compatibility
        #         rir_scp=args.rir_scp if hasattr(args, "rir_scp") else None,
        #         rir_apply_prob=args.rir_apply_prob
        #         if hasattr(args, "rir_apply_prob")
        #         else 1.0,
        #         noise_scp=args.noise_scp if hasattr(args, "noise_scp") else None,
        #         noise_apply_prob=args.noise_apply_prob
        #         if hasattr(args, "noise_apply_prob")
        #         else 1.0,
        #         noise_db_range=args.noise_db_range
        #         if hasattr(args, "noise_db_range")
        #         else "13_15",
        #         speech_volume_normalize=args.speech_volume_normalize
        #         if hasattr(args, "rir_scp")
        #         else None,
        #     )
        # else:
        #     retval = None
        # assert check_return_type(retval)
        return None

    @classmethod
    def required_data_names(
            cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("speech", )
        else:
            # Recognition mode
            retval = ("speech", )
        return retval

    @classmethod
    def optional_data_names(
            cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ()
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace):
        assert check_argument_types()

        # 1. frontend
        if args.input_size is None or args.frontend == "wav_frontend_mel23":
            # Extract features in the model
            frontend_class = frontend_choices.get_class(args.frontend)
            if args.frontend == 'wav_frontend':
                frontend = frontend_class(cmvn_file=args.cmvn_file, **args.frontend_conf)
            else:
                frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()
        else:
            # Give features from data-loader
            args.frontend = None
            args.frontend_conf = {}
            frontend = None
            input_size = args.input_size

        # 2. Encoder
        encoder_class = encoder_choices.get_class(args.encoder)
        encoder = encoder_class(**args.encoder_conf)

        # 3. EncoderDecoderAttractor
        encoder_decoder_attractor_class = encoder_decoder_attractor_choices.get_class(args.encoder_decoder_attractor)
        encoder_decoder_attractor = encoder_decoder_attractor_class(**args.encoder_decoder_attractor_conf)

        # 9. Build model
        model_class = model_choices.get_class(args.model)
        model = model_class(
            frontend=frontend,
            encoder=encoder,
            encoder_decoder_attractor=encoder_decoder_attractor,
            **args.model_conf,
        )

        # 10. Initialize
        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model

    # ~~~~~~~~~ The methods below are mainly used for inference ~~~~~~~~~
    @classmethod
    def build_model_from_file(
            cls,
            config_file: Union[Path, str] = None,
            model_file: Union[Path, str] = None,
            cmvn_file: Union[Path, str] = None,
            device: str = "cpu",
    ):
        """Build model from the files.

        This method is used for inference or fine-tuning.

        Args:
            config_file: The yaml file saved when training.
            model_file: The model file saved when training.
            cmvn_file: The cmvn file for front-end
            device: Device type, "cpu", "cuda", or "cuda:N".

        """
        assert check_argument_types()
        if config_file is None:
            assert model_file is not None, (
                "The argument 'model_file' must be provided "
                "if the argument 'config_file' is not specified."
            )
            config_file = Path(model_file).parent / "config.yaml"
        else:
            config_file = Path(config_file)

        with config_file.open("r", encoding="utf-8") as f:
            args = yaml.safe_load(f)
        args = argparse.Namespace(**args)
        model = cls.build_model(args)
        if not isinstance(model, AbsESPnetModel):
            raise RuntimeError(
                f"model must inherit {AbsESPnetModel.__name__}, but got {type(model)}"
            )
        if model_file is not None:
            if device == "cuda":
                device = f"cuda:{torch.cuda.current_device()}"
            checkpoint = torch.load(model_file, map_location=device)
            if "state_dict" in checkpoint.keys():
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
        model.to(device)
        return model, args
