
class RRMC():
    def __init__(self):
        self.panda = rb.models.DH.Panda()

    def fkine(self):
        # Tip pose in world coordinate frame
        wTe = SE3(env._task.robot.arm.get_tip().get_position())*SE3.RPY(env._task.robot.arm.get_tip().get_orientation(), order='zyx')
        return wTe
    
    def target_pose(self):
        # Target pose in world coordinate frame
        wTt = SE3(env._task.target.get_position())*SE3.RPY(env._task.target.get_orientation(), order='zyx')
        print(env._task.target.get_orientation())
        return wTt
    
    def p_servo(self, gain=1):
        
        wTe = self.fkine()
        #print("wTe: ", wTe.t)
        wTt = self.target_pose()
        #print("wTt: ", wTt.t)
    
        # Pose difference
        eTt = wTe.inv() * wTt
        # Translational velocity error
        ev = eTt.t
        # Angular velocity error
        ew = eTt.rpy() * np.pi/180
        #ew = np.zeros(3)
        # Form error vector
        e = np.r_[ev, ew]
        #print("e: ", e)
        v = gain * e
        #print("v: ", v)
        return v
    
    def compute_action(self, gain=0.3):
        
        try:
            v = self.p_servo(gain)
            #v[3:] *= 10
            q = env._task.robot.arm.get_joint_positions()
            #print(q)
            #print(np.round(self.panda.jacobe(q), 2))
            action = np.linalg.pinv(self.panda.jacobe(q)) @ v
            #print("action: ", action)

        except np.linalg.LinAlgError:
            action = np.zeros(env_task.action_size)
            print('Fail')
        return action