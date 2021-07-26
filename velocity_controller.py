from pyrep.errors import IKError
import numpy as np

class joint_velocity_controller():
    def __init__(self, panda):
        self.panda = panda
        self.target_q = np.zeros(7)
    
    def set_target(self, target):
        pos = target.get_position()
        target.set_orientation([0,3.14,0]) # Set orientation to be upright
        quat = target.get_quaternion()

        try:
            self.target_q = np.array(self.panda.solve_ik_via_jacobian(pos, quaternion=quat))
        except IKError:
            # So let's swap to an alternative IK method...
            # This returns 'max_configs' number of joint positions
            print("Trying sampling...")
            self.target_q = np.array(self.panda.solve_ik_via_sampling(pos, quaternion=quat)[0])


    def compute_action(self, gain=0.03):
        current_q = self.panda.get_joint_positions()
        err = current_q - self.target_q
        v = -err * gain
        return v

