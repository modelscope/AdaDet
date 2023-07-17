# Copyright (c) Alibaba, Inc. and its affiliates.
from modelscope.pipelines import pipeline
from modelscope.utils.constant import ThirdParty
from modelscope.utils.logger import get_logger
from .infer import INFERS, Infer

logger = get_logger()


@INFERS.register_module('adadet_infer', 'ModelScopePipeline')
class ModelScopePipeline(Infer):

    def __init__(self, task, model_id, **kwargs) -> None:
        super().__init__()
        self.pipeline = pipeline(
            task,
            model=model_id,
            third_party=getattr(ThirdParty, 'ADADET', 'adadet'),
            **kwargs)

    def __call__(self, input, *args, **kwargs):
        return self.pipeline(input, *args, **kwargs)
