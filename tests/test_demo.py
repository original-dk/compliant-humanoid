import unittest
import mujoco as mj
from compliant_humanoid.augmentation import XML, CLIP
from compliant_humanoid.motion import load_clip


class DemoTest(unittest.TestCase):
    def test_project_assets(self):
        model = mj.MjModel.from_xml_path(str(XML))
        motion, features = load_clip(CLIP, model)
        self.assertEqual((model.nq, model.nv), (36, 35))
        self.assertEqual(motion.shape, (201, 36))
        self.assertEqual(features.shape, (201, 38))
