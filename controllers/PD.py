import numpy as np

def PD(q_ref, q, q_dot, Kp=1, Kd=1, torque_limit=5):
    # Output torques
    torques = Kp * (q_ref - q) - Kd * q_dot

    # clip to limit the maximal value that torques can have
    torques = np.clip(torques, -torque_limit, torque_limit)

    return torques
