# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from adadet.deploy.deploy import deploy
from adadet.utils.config import Config

from modelscope.utils.test_utils import test_level


class FaceRetouchDeploy(unittest.TestCase):

    def setUp(self) -> None:
        self.input_image = './test/data/images/face_retouch_5.png'
        self.config_path = 'configs/deploy/human/face_retouch_deploy.yaml'
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_face_retouch_deploy_image(self):
        cfg = Config.from_file(self.config_path)
        cfg.adadet_deploy.rules.is_video = False
        face_retouch_deploy = deploy(cfg)
        res = face_retouch_deploy(self.input_image)
        print(f'res shape={res.shape}')


if __name__ == '__main__':
    unittest.main()
