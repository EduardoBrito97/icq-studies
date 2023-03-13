import numpy as np
from scipy.linalg import expm as expMatrix

def get_sigmaE(vector_x, vector_w):
    """
        Multiplies the input (vector_x) by the weights (vector_w), resulting in a diagonal matrix. 
        It discards any imaginary part vector_x and vector_w might have.
        Equivalent of Equation #17 in the Article.
    """
    n = len(vector_x)
    sigmaE = np.zeros((n,n))
    for i in range(n):
        sigmaE[i,i] = np.real(vector_x[i])*np.real(vector_w[i])

    return sigmaE

def get_weighted_sigmaQ(param):
    """
        returns param[0]*sigmaX + param[1]*sigmaY + param[2]*sigmaZ to get sigmaQ.
        - sigmaX comes from Equation #7 = [0, 1   1, 0]
        - sigmaY comes from Equation #8 = [0, -i  i, 0]
        - sigmaZ comes from Equation #9 = [1, 0   0, -1]
        Equivalent of Equation #16 in the Article.
    """
    sigmaQ = np.zeros((2,2))
    sigmaX = np.array([[0,1], [1,0]])
    sigmaY = np.array([[0,-1j], [1j,0]])
    sigmaZ = np.array([[1,0], [0,-1]])
    sigmaQ = param[0]*sigmaX + param[1]*sigmaY + param[2]*sigmaZ

    return sigmaQ

def get_sigmaQ_from_polar_coord(param):
    """
        param should be an array that pulls:
        - r = param[0]
        - theta = param[1]
        - phi = param[2]

        returns (identity + (rx * sigmaX) + (ry * sigmaY) + (rz * sigmaZ))/2 to get sigmaQ.
        - sigmaX comes from Equation #7 = [0, 1   1, 0]
        - sigmaY comes from Equation #8 = [0, -i  i, 0]
        - sigmaZ comes from Equation #9 = [1, 0   0, -1]

        where:
        - rx = r * sin(theta) * cos(phi)
        - ry = r * sin(theta) * sin(phi)
        - rz = r * cos(theta)
        
        It's an improved version of Equation #16 from the article, since we need it to sum up to 1.
    """
    # First we retrieve the params
    r = param[0]
    theta = param[1]
    phi = param[2]

    # Then we find out what are our rx, ry and rz
    rx = r * np.sin(theta) * np.cos(phi)
    ry = r * np.sin(theta) * np.sin(phi)
    rz = r * np.cos(theta)

    # Latest part is define sigmaX, sigmaY and sigmaZ from Equations #7, #8 and #9 respectively
    sigmaX = np.array([[0,1], [1,0]])
    sigmaY = np.array([[0,-1j], [1j,0]])
    sigmaZ = np.array([[1,0], [0,-1]])

    # Plus the identity which is needed
    identity = np.array([[1, 0], [0, 1]])

    # Now we return the calculation
    return (identity + (rx * sigmaX) + (ry * sigmaY) + (rz * sigmaZ))/2

def get_U_operator(sigmaQ, sigmaE):
    """
        Makes the exponential matrix of tensor product between sigmaQ and sigmaE and multiplies it by j. 
        
        Equivalent of Equation #15 in the Article.
    """
    sigmaQ[np.isnan(sigmaQ)] = 0
    sigmaE[np.isnan(sigmaE)] = 0
    return np.matrix(expMatrix(1j*np.kron(sigmaQ, sigmaE)))

def get_p(psi):
    """
        Creates a matrix out of psi and multiply it against its inverse, resulting in a column vector in the form [[alfa]. [beta]].
        
        Does the operation |psi><psi| from Equation #18 or #19 in the Article.
    """
    psi = np.matrix(psi)
    return psi * psi.getH()

def normalize(x):
    v_norm = x / (np.linalg.norm(x) + 1e-16)
    return v_norm

def create_and_execute(vector_x, 
                        vector_w, 
                        normalize_x=False, 
                        normalize_w=False, 
                        split_input_weight=True, 
                        sigma_q_params=[1,1,1],
                        use_polar_coordinates_on_sigma_q = False):
    """
        Applies the a modified version of ICQ classifier using only the math behind the Quantum Classifier described in Interactive Quantum Classifier Inspired by Quantum Open System Theory article. 
        
        It differs from the original ICQ by adding a new component to Sigma Q: sigmaH, which corresponds to a Haddamard's gate. Another difference is that we load the input in the environment instead of having a combination of weights and inputs in sigmaE.

        After doing so, it gets the result of Equation #20 and returns Z as the predicted class and the probability of being the class 1.
        
        Works only for binary classifications, therefore, if the probability of class 0 is needed, it can be 1 - probability of being class 1.

        To have the original ICQ Classifier, you can have:
        normalize_x = False
        normalize_w = False
        split_input_weight = False
        sigma_q_params = [1, 1, 1]
    """

    if normalize_x:
        vector_x = normalize(vector_x)
    if normalize_w:
        vector_w = normalize(vector_w)  

    if (use_polar_coordinates_on_sigma_q):
        # Eq #16, but using polar coordinates so |sigmaQ| gets to be 1
        sigmaQ = get_sigmaQ_from_polar_coord(sigma_q_params)
    else:
        # Eq #16
        sigmaQ = get_weighted_sigmaQ(sigma_q_params)

    # Eq #17
    N = len(vector_x)

    # Equivalent to Eq #15
    if split_input_weight:
        # We can either keep only weights
        sigmaE = np.zeros((N,N), dtype= np.complex128)
        for i in range(N):
            sigmaE[i,i] = vector_w[i]
    else:
        # Or keep both as the original ICQ article
        sigmaE = get_sigmaE(vector_x, vector_w)
        
    U_operator = get_U_operator(sigmaQ, sigmaE)

    # Eq #18 applied on a Quantum state equivalent of Hadamard(|0>) = 1/sqrt(2) * (|0> + |1>) 
    p_cog = get_p([[1/np.sqrt(2)],[1/np.sqrt(2)]])

    # Eq #19 applied on a Quantum state equivalent of Hadamard(|00...0>) = 1/sqrt(N) * (|00...0> + ... + |11...1>)
    if split_input_weight:
        # We can either have Hadamard applied to each instance attribute...
        vector_x_norm = (np.linalg.norm(vector_x) + 1e-16)
        p_env = get_p([[vector_x_i*(1/vector_x_norm)] for vector_x_i in vector_x])
    else:
        # ... or have as the original ICQ: Hadamard applied to a |00...0> gate
        p_env = get_p([[1/np.sqrt(N)] for _ in range(N)])

    # Extracting p_cog and p_env kron
    p_cog_env = np.kron(p_cog, p_env)

    # First part of Equation #20 in the Article
    quantum_operation = np.array(U_operator * p_cog_env * U_operator.getH())

    # Second part of Equation #20 in the Article
    p_cog_new = np.trace(quantum_operation.reshape([2,N,2,N]), axis1=1, axis2=3)

    # As the result is a diagonal matrix, the probability of being class 0 will be on position 0,0
    p_cog_new_00_2 = p_cog_new[0,0]

    # ... and the probability of being class 1 will be on position 1,1
    p_cog_new_11_2 = p_cog_new[1,1]
    if (p_cog_new_00_2 >= p_cog_new_11_2):
        z = 0
    else:
        z = 1
    return z, p_cog_new_11_2, U_operator

def update_weights(weights, y, z, x, p, n):
  """
    Updates the weights. Equation #34 in the Article.
    
    y is the expected classification [0, 1];
    z is the actual classification [0, 1];
    x is the attribute vector;
    p is the probability of the class 1 (0, 1), powered to 2 (p²);
    n is the learning rate.
  """
  # Eq 33
  loss_derivative_on_weight = (1-p)*x

  # Eq 34
  weights = weights-n*(z-y)*loss_derivative_on_weight
  weights[np.isnan(weights)] = 0
  return weights
