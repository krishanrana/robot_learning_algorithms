
from pyrep.errors import IKError
import numpy as np

class cartesian_position_controller():
    def __init__(self, panda):
        self.panda = panda
        self.target_p = np.zeros(3)
        self.target_r = np.zeros(4)
    
    def set_target(self, target):
        self.target_p = target.get_position()
        target.set_orientation([0,3.14,0]) # Set orientation to be upright
        self.target_r = target.get_quaternion()

    def compute_action(self, gain=0.03):
        current_p = self.panda.get_tip().get_position()
        e_pos = (current_p - self.target_p) * gain

        current_r = self.panda.get_tip().get_quaternion()
        e_rot = (current_r - self.target_r) * gain

        try:
            q_vel = np.array(self.panda.solve_ik_via_jacobian(e_pos, quaternion=e_rot))
        except IKError:
            # So let's swap to an alternative IK method...
            # This returns 'max_configs' number of joint positions
            print("Trying sampling...")
            q_vel = np.array(self.panda.solve_ik_via_sampling(e_pos, quaternion=e_rot)[0])

        v = np.append(q_vel, 1.0)
        return v

    def recompute_action(self, current_q, target_q, gain=0.03):
        err = current_q - target_q
        v = -err * gain
        v = np.append(v, 1.0)
        return v
