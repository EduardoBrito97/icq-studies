import numpy as np
from scipy.linalg import expm as expMatrix

def get_sigmaE(vectorX, vectorW):
    """
        Multiplies the input (vectorX) by the weights (vectorW), resulting in a diagonal matrix. 
        It discards any imaginary part vectorX and vectorW might have.
        Equivalent of Equation #17 in the Article.
    """
    n = len(vectorX)
    sigmaE = np.zeros((n,n))
    for i in range(n):
        sigmaE[i,i] = np.real(vectorX[i])*np.real(vectorW[i])

    return sigmaE

def get_sigmaQ(n):
    """
        Sums sigmaX, sigmaY and sigmaZ to get sigmaQ.
        - sigmaX comes from Equation #7 = [0, 1   1, 0]
        - sigmaY comes from Equation #8 = [0, -i  i, 0]
        - sigmaZ comes from Equation #9 = [1, 0   0, -1]
        Equivalent of Equation #16 in the Article.
    """
    sigmaQ = np.zeros((n,n))
    sigmaX = np.array([[0,1], [1,0]])
    sigmaY = np.array([[0,-1j], [1j,0]])
    sigmaZ = np.array([[1,0], [0,-1]])
    sigmaQ = sigmaX + sigmaY + sigmaZ

    return sigmaQ

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
        Creates a matrix out of psi and multiply it against its inverse, 
        resulting in a column vector in the form [[alfa]. [beta]].
        Does the operation |psi><psi| from Equation #18 or #19 in the Article.
    """
    psi = np.matrix(psi)
    return psi * psi.getH()

def create_and_execute_classifier(vectorX, vectorW):
    """
        Applies the ICQ classifier using only the math behind the Quantum Classifier 
        described in Interactive Quantum Classifier Inspired by Quantum Open System Theory
        article. 
        After doing so, it gets the result of Equation #20 and returns Z as the predicted class and
        the probability of being the class 1.
        Works only for binary classifications, therefore, if the probability of class 0 is needed, it can
        be 1 - probability of being class 1.
    """

    # Eq #16
    sigmaQ = get_sigmaQ(2)

    # Eq #17
    sigmaE = get_sigmaE(vectorX, vectorW)

    # Eq #15
    U_operator = get_U_operator(sigmaQ, sigmaE)

    # Eq #18 applied on a Quantum state equivalent of Hadamard(|0>) = 1/sqrt(2) * (|0> + |1>) 
    p_cog = get_p([[1/np.sqrt(2)],[1/np.sqrt(2)]])

    # As we must have 1 row per attribute of the input, we need env to be as big as one instance of our input
    N = len(vectorX)

    # Eq #19 applied on a Quantum state equivalent of Hadamard(|000000...>) = 1/sqrt(N) * (|000000...> + ... + |11111111....>) 
    p_env = get_p([[1/np.sqrt(N)] for i in range(N)])

    # First part of Equation #20 in the Article
    quantum_operation = np.array(U_operator * (np.kron(p_cog, p_env)) * U_operator.getH())

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
    return z, p_cog_new_11_2

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
