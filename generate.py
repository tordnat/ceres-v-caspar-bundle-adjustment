from dataclasses import dataclass
from pathlib import Path
import numpy as np
import symforce

symforce.set_epsilon_to_number(float(10 * np.finfo(np.float32).eps))
import symforce.symbolic as sf
from symforce import typing as T
from symforce.experimental.caspar import CasparLibrary
from symforce.experimental.caspar import memory as mem
from symforce.codegen import codegen_util


# class PinholeCamera:
#   cam_T_world: sf.Pose3
#   calibration: sf.V3

@dataclass
class PinholeIdeal():
  cam_T_world: sf.Pose3


class Point(sf.V3):
  pass

class Pixel(sf.V2):
  pass

caslib = CasparLibrary()

@caslib.add_factor
def simple_pinhole(
    cam: T.Annotated[PinholeIdeal, mem.Tunable],
    point: T.Annotated[Point, mem.Tunable],
    pixel: T.Annotated[Pixel, mem.Constant],
)->sf.V2:
    focal_length = 1
    point_cam = cam.cam_T_world * point
    depth = point_cam[2]
    point_ideal_camera_coords = sf.V2(point_cam[:2])/(depth + sf.epsilon() * sf.sign_no_zero(depth))
    pixel_projected = focal_length  * point_ideal_camera_coords
    reprojection_error = pixel_projected - pixel
    return reprojection_error


@caslib.add_factor
def position_prior(
  cam1: T.Annotated[PinholeIdeal, mem.Tunable],
  position_anchor: T.Annotated[sf.V3, mem.Constant],
)->sf.V3:
  cam1_position = sf.V3(cam1.cam_T_world.t)
  return position_anchor - cam1_position


@caslib.add_factor
def distance_prior(
  cam1: T.Annotated[PinholeIdeal, mem.Tunable],
  cam2: T.Annotated[PinholeIdeal, mem.Tunable],
  dist: T.Annotated[float, mem.Constant],
)->float:
  pose1_x = cam1.cam_T_world.t[0]
  pose2_x = cam2.cam_T_world.t[0]
  cam_dist = pose2_x - pose1_x
  return dist - cam_dist


out_dir = Path(__file__).resolve().parent / "generated"
caslib.generate(out_dir)
caslib.compile(out_dir)