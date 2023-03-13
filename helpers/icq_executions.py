import numpy as np
from icq_methods import create_and_execute

def execute_classifier_split_input_weight_normal_sigma_q(vector_x, 
                                               vector_w, 
                                               sigma_q_params=[1,1,1]):
    """
        Executes classifier with the new approach, where weights are loaded in SigmaE and inputs are loaded in rho_env. 
        Uses sigma_q_param[0]*sigmaX + sigma_q_param[1]*sigmaY + sigma_q_param[2]*sigmaZ as sigmaQ.
    """
    return create_and_execute(vector_x, 
                                vector_w, 
                                normalize_x=False, 
                                normalize_w=False, 
                                split_input_weight=True, 
                                sigma_q_params=sigma_q_params,
                                use_polar_coordinates_on_sigma_q=False)

def execute_classifier_split_input_weight_polar_sigma_q(vector_x, 
                                               vector_w, 
                                               sigma_q_params=[1,np.pi/4,np.pi/4]):
    """
        Executes classifier with the new approach, where weights are loaded in SigmaE and inputs are loaded in rho_env. 
        Uses rx*sigmaX + ry*sigmaY + rz*sigmaz as sigmaQ, where:
        - r = param[0] (usually = 1)
        - rx = r * sin(param[1]) * cos(param[2])
        - ry = r * sin(param[1]) * sin(param[2])
        - rz = r * cos(param[1])
    """
    return create_and_execute(vector_x, 
                                vector_w, 
                                normalize_x=False, 
                                normalize_w=False, 
                                split_input_weight=True, 
                                sigma_q_params=sigma_q_params,
                                use_polar_coordinates_on_sigma_q=True)

def execute_classifier_original_normal_sigma_q(vector_x, 
                                               vector_w, 
                                               sigma_q_params=[1,1,1]):
    """
        Executes classifier with the original approach, where SigmaE uses both weights and inputs. However, we initialize the weights randomly and does not uses batches to update our model. 
        Uses sigma_q_param[0]*sigmaX + sigma_q_param[1]*sigmaY + sigma_q_param[2]*sigmaZ as sigmaQ.
    """
    return create_and_execute(vector_x, 
                                vector_w, 
                                normalize_x=False, 
                                normalize_w=False, 
                                split_input_weight=False, 
                                sigma_q_params=sigma_q_params,
                                use_polar_coordinates_on_sigma_q=False)

def execute_classifier_original_polar_sigma_q(vector_x, 
                                               vector_w, 
                                               sigma_q_params=[1,np.pi/4,np.pi/4]):
    """
        Executes classifier with the original approach, where SigmaE uses both weights and inputs. However, we initialize the weights randomly and does not uses batches to update our model. 
        Uses rx*sigmaX + ry*sigmaY + rz*sigmaz as sigmaQ, where:
        - r = param[0] (usually = 1)
        - rx = r * sin(param[1]) * cos(param[2])
        - ry = r * sin(param[1]) * sin(param[2])
        - rz = r * cos(param[1])
    """
    return create_and_execute(vector_x, 
                                vector_w, 
                                normalize_x=False, 
                                normalize_w=False, 
                                split_input_weight=False, 
                                sigma_q_params=sigma_q_params,
                                use_polar_coordinates_on_sigma_q=True)