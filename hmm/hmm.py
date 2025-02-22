import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p




    def forward(self, input_observation_states: np.ndarray) -> float:
        """ 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """       

        ## Edge Cases 
        if not np.isclose(np.sum(self.prior_p), 1.0):
             raise ValueError("Prior probabilities need to sum to 1")
        if not np.allclose(np.sum(self.transition_p, axis=1), 1.0):
             raise ValueError("Transition probabilities need to sum to 1")
        if not np.allclose(np.sum(self.emission_p, axis=1), 1.0):
             raise ValueError("Emission probabilities need to sum to 1")
        if not len(self.prior_p) == len(self.hidden_states):
            raise ValueError("Prior probabilities need to be the same length as hidden states")
   
       
        T = len(input_observation_states)
        # Number of hidden states
        N = len(self.hidden_states)
        
        # Initialize forward matrix (alpha)
        # Shape: (T x N) - T time steps, N hidden states
        alpha = np.zeros((T, N))
        
        # Step 1: Initialization
        # Calculate alpha for t=0 (
        first_observation = input_observation_states[0]
        first_observation_idx = self.observation_states_dict[first_observation]
        for i in range(N):
            # α₁(i) = πᵢ * bᵢ(o₁)
            alpha[0, i] = self.prior_p[i] * self.emission_p[i, first_observation_idx]
        
        # Step 2: Recursion
        # Calculate alpha for remaining time steps
        for t in range(1, T):
            current_observation = input_observation_states[t]
            current_observation_idx = self.observation_states_dict[current_observation]
            
            for j in range(N):  # Current state
                # Sum over all possible previous states
                sum_term = 0
                for i in range(N):  # Previous state
                    # α_t(j) = [Σᵢ α_(t-1)(i) * aᵢⱼ] * bⱼ(oₜ)
                    sum_term += alpha[t-1, i] * self.transition_p[i, j]
                
                # Multiply by emission probability
                alpha[t, j] = sum_term * self.emission_p[j, current_observation_idx]
        
        # Step 3: Termination
        # Sum over all possible final states to get total probability
        final_probability = np.sum(alpha[T-1])
        
        return final_probability

 
        
    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states    
        """
        ## Edge Cases 
        if not np.isclose(np.sum(self.prior_p), 1.0):
             raise ValueError("Prior probabilities need to sum to 1")
        if not np.allclose(np.sum(self.transition_p, axis=1), 1.0):
             raise ValueError("Transition probabilities need to sum to 1")
        if not np.allclose(np.sum(self.emission_p, axis=1), 1.0):
             raise ValueError("Emission probabilities need to sum to 1")
        if not len(self.prior_p) == len(self.hidden_states):
            raise ValueError("Prior probabilities need to be the same length as hidden states")


        # Step 1. Initialize variables
        T = len(decode_observation_states)  
        N = len(self.hidden_states)       
        
        # Store probabilities of most likely path to each state at each time step
        viterbi_table = np.zeros((T, N))
        backpointer = np.zeros((T, N), dtype=int)
        
        # Initialize first time step (t=0)
        first_observation = decode_observation_states[0]
        first_observation_idx = self.observation_states_dict[first_observation]
        for i in range(N):
            # δ₁(i) = πᵢ * bᵢ(o₁)
            viterbi_table[0, i] = self.prior_p[i] * self.emission_p[i, first_observation_idx]
            backpointer[0, i] = 0
        
        # Step 2. Calculate Probabilities
        # For each time step after t=0
        for t in range(1, T):
            current_observation = decode_observation_states[t]
            current_observation_idx = self.observation_states_dict[current_observation]
            
            for j in range(N):  # For each current state
                # Calculate probability for each possible previous state
                probabilities = (viterbi_table[t-1] * 
                            self.transition_p[:, j] * 
                            self.emission_p[j, current_observation_idx])
                
                # Store the highest probability
                viterbi_table[t, j] = np.max(probabilities)
                # Store the index of the previous state that gave highest probability
                backpointer[t, j] = np.argmax(probabilities)
        
        # Step 3. Traceback
        # Find most likely final state
        best_path = np.zeros(T, dtype=int)
        best_path[T-1] = np.argmax(viterbi_table[T-1])
        
        # Trace back through the backpointers
        for t in range(T-2, -1, -1):
            best_path[t] = backpointer[t+1, best_path[t+1]]
        
        # Step 4. Return best hidden state sequence
        # Convert state indices to state names
        best_hidden_state_sequence = [self.hidden_states[i] for i in best_path]
        
        return best_hidden_state_sequence