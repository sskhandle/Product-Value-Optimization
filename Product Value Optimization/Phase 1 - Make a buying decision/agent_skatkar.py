
from agents import Agent

class Agent_skatkar(Agent):
    def will_buy(self, value, price, prob):
 
#           ****ONE SOLUTION TO THE PROBLEM THAT FINDS A HIGHER PROFIT THAN THE GIVEN ALGORITHMS****
#----------------------------------------------------------------------------------------------------------------- 
            # if price/value < 0.25:
                # return (prob > 0.3)
            
            # elif price/value < 0.5:
                # return (prob > 0.5)
            
            # else:
                # return (prob > 0.75)
                
#------------------------------------------------------------------------------------------------------------------


#           ****AN OPTIMIZED VERSION OF THE INITIAL SOLUTION****
#------------------------------------------------------------------------------------------------------------------
            return (prob > (price/value))
            
#------------------------------------------------------------------------------------------------------------------
