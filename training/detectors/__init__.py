import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

from metrics.registry import DETECTOR

# Function to safely import detectors
def safe_import(module_name, class_name):
    try:
        module = __import__(module_name, fromlist=[class_name], globals=globals(), level=1)
        return getattr(module, class_name)
    except Exception as e:
        print(f"Skipping {class_name}: {e}")
        return None

# Core detector we need for the demo
from .xception_detector import XceptionDetector

# Wrap everything else in try-except to avoid app crashes
try: from .facexray_detector import FaceXrayDetector
except Exception: pass
try: from .efficientnetb4_detector import EfficientDetector
except Exception: pass
try: from .resnet34_detector import ResnetDetector
except Exception: pass
try: from .f3net_detector import F3netDetector
except Exception: pass
try: from .meso4_detector import Meso4Detector
except Exception: pass
try: from .meso4Inception_detector import Meso4InceptionDetector
except Exception: pass
try: from .spsl_detector import SpslDetector
except Exception: pass
try: from .core_detector import CoreDetector
except Exception: pass
try: from .capsule_net_detector import CapsuleNetDetector
except Exception: pass
try: from .srm_detector import SRMDetector
except Exception: pass
try: from .ucf_detector import UCFDetector
except Exception: pass
try: from .recce_detector import RecceDetector
except Exception: pass
try: from .fwa_detector import FWADetector
except Exception: pass
try: from .ffd_detector import FFDDetector
except Exception: pass
try: from .videomae_detector import VideoMAEDetector
except Exception: pass
try: from .clip_detector import CLIPDetector
except Exception: pass
try: from .timesformer_detector import TimeSformerDetector
except Exception: pass
try: from .xclip_detector import XCLIPDetector
except Exception: pass
try: from .sbi_detector import SBIDetector
except Exception: pass
try: from .ftcn_detector import FTCNDetector
except Exception: pass
try: from .i3d_detector import I3DDetector
except Exception: pass
try: from .altfreezing_detector import AltFreezingDetector
except Exception: pass
try: from .stil_detector import STILDetector
except Exception: pass
try: from .lsda_detector import LSDADetector
except Exception: pass
try: from .sladd_detector import SLADDXceptionDetector
except Exception: pass
try: from .pcl_xception_detector import PCLXceptionDetector
except Exception: pass
try: from .iid_detector import IIDDetector
except Exception: pass
try: from .lrl_detector import LRLDetector
except Exception: pass
try: from .rfm_detector import RFMDetector
except Exception: pass
try: from .uia_vit_detector import UIAViTDetector
except Exception: pass
try: from .multi_attention_detector import MultiAttentionDetector
except Exception: pass
try: from .sia_detector import SIADetector
except Exception: pass
try: from .tall_detector import TALLDetector
except Exception: pass
try: from .effort_detector import EffortDetector
except Exception: pass
