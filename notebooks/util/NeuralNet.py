# %%
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict,Counter
from random import sample
from scipy.special import expit

# %%
class NeuralNet:
    def __init__(self,nodes,lamb=0.0,alpha=0.1,eps=0.0):
        '''
        Constructor for neural net
        nodes - list detailing number of nodes in each layer
        lamb - regularization
        alpha - learning rate
        eps - cost function stopping condition
        '''
        self.nodes = nodes
        self.lamb = lamb
        self.alpha = alpha
        self.weights = []
        self.eps = eps
        #initialize weights for each layer, include bias
        for i in range(len(nodes)-1):
            self.weights.append(np.random.normal(0,1,(nodes[i]+1,nodes[i+1])).T)
    
    def get_sigmoid(self, x):
        return expit(x)
        #return 1 / (1+np.exp(-x))
    
    def deriv_sigmoid(self, x):
        return x * (1-x)

    def train(self, features, targs, batch_size, test_feat=None, test_targs=None, for_exam=False, get_costs=False):
        '''
        features - training data features
        targs - training data targets
        batch size - # of instances for mini batch
        test_feat - test data features
        test_targs - test data targets
        for_exam - flag to print for back_prop examples
        get_costs - flag to get J values for varying number of samples
        '''
        prev_cost = math.inf
        gradients = [0]*len(self.weights)
        num_inst = len(targs)
        keep_learn = True
        count = 1
        curr_batch = 1
        cost_j = []
        cost_j_count = []

        while(keep_learn):
            J = 0
            for instance,target in zip(features,targs):
                #iterate through layers, vectorize forward pass
                activations = [np.atleast_2d(instance)]
                for i in range(len(self.weights)-1):
                    try:
                        this_a = self.get_sigmoid(self.weights[i].dot(activations[i].T))
                    except:
                        this_a = self.get_sigmoid(self.weights[i].T.dot(activations[i].T))
                    activations.append(np.insert(this_a,0,1))
                try:
                    activations.append(self.get_sigmoid(activations[len(self.weights)-1].dot(self.weights[len(self.weights)-1])))
                except:
                    activations.append(self.get_sigmoid(activations[len(self.weights)-1].dot(self.weights[len(self.weights)-1].T)))
                guess = activations[-1]

                #accumulate sum loss
                target = np.array(target)
                cost = np.sum((np.array(-target)).dot(np.log(guess)) - (np.array(1-target)).dot(np.log(1-guess)))
                J += cost

                #begin backwards propogation
                error = guess - target
                delta_inst = [error]

                #get delta values for all weights on current instance
                for i in range(len(self.weights)-1, 0, -1):
                    try:
                        this_del = (self.weights[i].T.dot(delta_inst[-1])) * self.deriv_sigmoid(activations[i].T)
                    except:
                        this_del = (self.weights[i].dot(delta_inst[-1])) * self.deriv_sigmoid(activations[i].T)
                    delta_inst.append(this_del[1:])

                #reverse delta values
                delta_inst = delta_inst[::-1]

                #accumulate gradients
                for i in range(len(self.weights)-1,-1,-1):
                    try:
                        gradients[i] += (delta_inst[i]*(activations[i].T)).T
                    except:
                        gradients[i] += (np.atleast_2d(delta_inst[i]).T*np.atleast_2d(activations[i].T))

                #print for examples
                if for_exam:
                    print(f'OUTPUTS FOR INSTANCE {count}')
                    print(f'activations: ')
                    for i in range(len(activations)):
                        print(f'a{i+1}: {activations[i]}')
                    print()
                    print(f'prediction: {guess}')
                    print(f'expected: {target}')
                    print(f'cost J: {cost}')
                    print()
                    print('delta for this instance: ')
                    for i in range(len(delta_inst)):
                        print(f'delta {i+2}: {delta_inst[i]}')
                    print()
                    print('gradients for this instance: ')
                    for i in range(len(self.weights)):
                        try:
                            print_del = (delta_inst[i]*(activations[i].T)).T
                        except:
                            print_del = (np.atleast_2d(delta_inst[i]).T*np.atleast_2d(activations[i].T)).T
                        print(f'theta {i+1}: {print_del}')
                    print()

                if curr_batch == batch_size or count == num_inst:
                    #regularize weights and update
                    for i in range(len(self.weights)-1,-1,-1):
                        P = self.lamb * (self.weights[i])
                        #set first column to all 0
                        P[:,0] = 0
                        try:
                            gradients[i] = gradients[i] + P.T
                        except:
                            gradients[i] = gradients[i] + P
                        gradients[i] = gradients[i] / num_inst
                        learn_diff = self.alpha * (gradients[i])
                        try:
                            self.weights[i] = self.weights[i] - learn_diff
                        except:
                            self.weights[i] = self.weights[i] - learn_diff.T
                    curr_batch = 0

                    if get_costs:
                        cost_j.append(self.cost_on_set(test_feat,test_targs))
                        cost_j_count.append(count)

                curr_batch += 1
                count += 1

            J /= num_inst
            curr_s = 0
            for i in range(len(self.weights)):
                curr_s += np.sum(self.weights[i][1:]**2)

            #curr_s = np.sum(self.weights[1:]**2)
            curr_s *= (self.lamb/(2*num_inst))
            new_cost = J + curr_s

            #if improvement in cost is less than epsilon, stop
            if prev_cost - new_cost < self.eps:
                keep_learn = False

            prev_cost = new_cost

            if for_exam:
                print('regularized gradients: ')
                for i in range(len(gradients)):
                    print(f'theta {i+1}: {gradients[i]}')
                keep_learn = False
            if get_costs:
                return cost_j,cost_j_count

    #forward pass on one instance, returns an array where index of max val is the NN's guess and 0 for all else
    #raw - True if wanting the raw outputs, false if wanting outputed in one hot encoding
    def predict(self,instance,raw=True):
        activations = [np.atleast_2d(instance)]
        for i in range(len(self.weights)-1):
            try:
                this_a = self.get_sigmoid(self.weights[i].dot(activations[i].T))
            except:
                this_a = self.get_sigmoid(self.weights[i].T.dot(activations[i].T))
            activations.append(np.insert(this_a,0,1))
        try:
            activations.append(self.get_sigmoid(activations[len(self.weights)-1].dot(self.weights[len(self.weights)-1])))
        except:
            activations.append(self.get_sigmoid(activations[len(self.weights)-1].dot(self.weights[len(self.weights)-1].T)))
        guess = activations[-1]
        pred = [0]*len(guess)
        pred[np.argmax(guess)] = 1
        
        return guess if raw else pred
    
    def cost_on_set(self,instances,targets):
        J = 0
        for instance,target in zip(instances,targets):
            guess = self.predict(instance)
            target = np.array(target)
            cost = np.sum((np.array(-target)).dot(np.log(guess)) - (np.array(1-target)).dot(np.log(1-guess)))
            J += cost
        J /= len(instances)
        curr_s = 0
        for i in range(len(self.weights)):
            curr_s += np.sum(self.weights[i][1:]**2)

        curr_s *= (self.lamb/(2*len(instances)))
        return J + curr_s
    
def test_decision(nn,test_set,vals):
    test_copy = pd.DataFrame(test_set,copy=True)
    to_guess = test_copy.drop('class',axis=1)
    predictions = pd.DataFrame(to_guess.apply(lambda row: nn.predict(row.to_numpy(),raw=False), axis=1),columns=['predicted'])
    predictions['actual'] = test_set.loc[predictions.index,'class']
    prec,rec,f1 = [0,0,0]

    for val in vals:
        is_targ = predictions[predictions.predicted.apply(lambda x: x == val)]
        not_targ = predictions[predictions.predicted.apply(lambda x: x != val)]
        tp = len(is_targ[is_targ['predicted'] == is_targ['actual']])
        fp = len(is_targ[is_targ['predicted'] != is_targ['actual']])
        fn = len(not_targ[not_targ.actual.apply(lambda x: x == val)])
        tn = len(not_targ[not_targ.actual.apply(lambda x: x != val)])
        this_prec = (tp/(tp+fp)) if (tp+fp) > 0 else 0
        this_rec = (tp/(tp+fn)) if (tp+fn) > 0 else 0
        f1 += (this_prec*this_rec*2)/(this_rec+this_prec) if (this_rec+this_prec) > 0 else 0
        prec += this_prec
        rec += this_rec

    avg_f1 = f1/len(vals)
    accuracy = len(predictions[predictions['predicted'] == predictions['actual']])/len(test_set)
    return accuracy,avg_f1

np.random.seed(1)
k = 10
#function to do cross fold validation
def k_fold(fold,vals,nn_arc,lamb,eps,alpha,batch_size,get_j=False):
    fold_metrics = defaultdict(list)
    #iterate through folds, taking turns being test fold
    for i in range(k):
        test_fold = fold[i]
        test_targs = test_fold['class']
        test_feat = test_fold.drop('class',axis=1)
        train_fold = fold[0:i]
        train_fold.extend(fold[i+1:len(fold)])
        train_data = pd.concat(train_fold)
       
        #iterate through architectures
        for arc in nn_arc:
            np_targs = train_data['class'].to_numpy()
            np_inst = train_data.drop('class',axis=1).to_numpy()
            this_nn = NeuralNet(arc,lamb,alpha,eps)
            if get_j:
                return this_nn.train(np_inst,np_targs,batch_size,test_feat.to_numpy(),test_targs.to_numpy(),get_costs=True)
            this_nn.train(np_inst,np_targs,batch_size)
            fold_metrics[str(arc)].append(test_decision(this_nn,test_fold,vals))
            
    return fold_metrics
