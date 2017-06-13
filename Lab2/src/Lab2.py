import numpy as np
import tools2
import matplotlib.pyplot as plt

# load data:
tidigits = np.load('../Datasets/lab2_tidigits.npz', encoding='latin1' )['tidigits']
models = np.load('../Datasets/lab2_models.npz', encoding='latin1' )['models']               # is an 'array'
example = np.load('../Datasets/lab2_example.npz', encoding='latin1' )['example'].item()     # is an 'dict'
X = example['mfcc']


def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """
    log_lik_gmm = 0
    for obs in range(len(log_emlik)):
        log_lik_gmm += tools2.logsumexp(log_emlik[obs, :] + np.log(weights))
    return log_lik_gmm    

def forward(log_emlik, log_startprob, log_transmat):
    """Forward probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """
    
    log_a = np.zeros((log_emlik.shape[0], log_transmat.shape[0]));
    
    for j in range(log_transmat.shape[0]):
        log_a[0, j] = log_startprob[j] + log_emlik[0,j]
    
    for i in range(1, log_emlik.shape[0]):
        for j in range(log_transmat.shape[0]):
            log_a[i, j] = tools2.logsumexp((log_a[i - 1, :] + log_transmat[:, j])) + log_emlik[i, j]
    
    return log_a


def backward(log_emlik, log_startprob, log_transmat):
    """Backward probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """

def viterbi(log_emlik, log_startprob, log_transmat):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """
    log_v = np.zeros((log_emlik.shape[0], log_transmat.shape[0]))
    B = np.zeros((log_emlik.shape[0], log_transmat.shape[0]))
    best_path = np.zeros([log_emlik.shape[0]], dtype = int)
    for j in range(log_transmat.shape[0]):
        log_v[0, j] = log_startprob[j] + log_emlik[0,j]

    for i in range(1, log_emlik.shape[0]):
        for j in range(log_transmat.shape[0]):
            log_v[i, j] = np.max(log_v[i - 1, :] + log_transmat[:, j]) + log_emlik[i, j]
            B[i, j] = np.argmax(log_v[i - 1, :] + log_transmat[:, j])
    
    best_path[log_emlik.shape[0] - 1] = np.argmax(log_v[log_emlik.shape[0] - 1, :])
    viterbi_loglik = log_v[log_emlik.shape[0] - 1, best_path[log_emlik.shape[0] - 1]]
    
    # Backtracking
    for i in range(log_emlik.shape[0] - 2, 1, -1):
        best_path[i] = B[i + 1, best_path[i + 1]]
    
    return log_v, best_path


################################################################################## ----- Quests----- ##################################################################################
def quest_4():
    #--- Quest 4: ---->>> Log-likelihood for each observation xi and each term in a multivariate 
    # HMM
    hmm_log_emlik = tools2.log_multivariate_normal_density_diag(X,  models[0]['hmm']['means'], models[0]['hmm']['covars'])
    hmm_check = example['hmm_obsloglik']

    # GMM
    gmm_log_emlik = tools2.log_multivariate_normal_density_diag(X, models[0]['gmm']['means'], models[0]['gmm']['covars'])
    gmm_check = example['gmm_obsloglik']


    #---- Plots
    fig1 = plt.figure(figsize = (10, 5))
    fig1.canvas.set_window_title('My results')

    plt.subplot(3, 1, 1)
    plt.pcolormesh(X.T, cmap = 'jet')
    plt.title('MFCC')
    plt.xticks([], [])

    plt.subplot(3, 1, 2)
    plt.pcolormesh(hmm_log_emlik.T, cmap = 'jet')
    plt.title('hmm_obsloglik')
    plt.xticks([], [])

    plt.subplot(3, 1, 3)
    plt.pcolormesh(gmm_log_emlik.T, cmap = 'jet')
    plt.title('gmm_obsloglik')
    plt.xticks([], [])

    plt.savefig("../figs/quest_4.png", bbox_inches='tight')

    #---- Check
    fig2 = plt.figure(figsize = (10, 5))
    fig2.canvas.set_window_title('Check')

    plt.subplot(3, 1, 1)
    plt.pcolormesh(X.T, cmap = 'jet')
    plt.title('MFCC')
    plt.xticks([], [])

    plt.subplot(3, 1, 2)
    plt.pcolormesh(hmm_check.T, cmap = 'jet')
    plt.title('hmm_obsloglik')
    plt.xticks([], [])

    plt.subplot(3, 1, 3)
    plt.pcolormesh(gmm_check.T, cmap = 'jet')
    plt.title('gmm_obsloglik')
    plt.xticks([], [])

    plt.savefig("../figs/quest_4_check.png", bbox_inches='tight')
    #plt.show()

    
def quest_5():
    #--- Quest 5: ---->>> Log-likelihood of an observation sequence X = {x0,...,xN−1} given the full GMM model 

    ''' # --->>> check
    gmm_log_emlik = tools2.log_multivariate_normal_density_diag(X, models[0]['gmm']['means'], models[0]['gmm']['covars'])
    log_lik_gmm = gmmloglik(gmm_log_emlik, models[0]['gmm']['weights'])
    check_log_lik_gmm = example['gmm_loglik']
    print(log_lik_gmm)
    print(check_log_lik_gmm)
    '''

    utters = len(tidigits)
    models_len = len(models)
    log_lik_gmm = np.zeros((utters, models_len))

    for utter in range(utters):
        for digit in range(models_len):
            gmm_log_emlik = tools2.log_multivariate_normal_density_diag(tidigits[utter]['mfcc'], models[digit]['gmm']['means'], models[digit]['gmm']['covars'])
            log_lik_gmm[utter, digit] = gmmloglik(gmm_log_emlik, models[digit]['gmm']['weights'])
    
    #---->>> Check for misrecognized utterances
    miss = 0
    print("------ GMM------")
    for utter in range(utters):
        best_score = np.argmax(log_lik_gmm[utter, :])
        print('tid digit, mod digit: ---> ' + str(tidigits[utter]['digit']) + " - " + str(models[best_score]['digit']))             # Uncomment to see the results analytically!
        if models[best_score]['digit'] != tidigits[utter]['digit']:
            miss += 1   
    accuracy = ((utters - miss) / utters) * 100  

    print()
    print("Misrecognized %d out of %d utterances." % (miss, len(tidigits)))
    print("Accuracy = " + str("%.2f" % round(accuracy,2)) + '%')
    print() 
               
def quest_6_1(flag):
    #--- Quest 6.1: ---->>> Forward Algorithm
    
     # --->>> check
    hmm_log_emlik = tools2.log_multivariate_normal_density_diag(X, models[0]['hmm']['means'], models[0]['hmm']['covars'])
    log_a = forward(hmm_log_emlik, np.log(models[0]['hmm']['startprob']), np.log(models[0]['hmm']['transmat']))
    check_log_a = example['hmm_logalpha']
    # print("log_a =", log_a)
    # print("check_log_a =", check_log_a)
    
    fig3 = plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(log_a.T, cmap = 'jet')
    plt.title('log_a')
    plt.xticks([], [])
    plt.gca().invert_yaxis()
    

    plt.subplot(2, 1, 2)
    plt.imshow(example['hmm_logalpha'].T, cmap = 'jet')
    plt.title('ckeck log_a')
    plt.xticks([], [])
    plt.gca().invert_yaxis()

    plt.savefig("../figs/quest_6_1.png", bbox_inches='tight')
    # plt.show()

    #Convert the formula you have derived into log domain
    log_lik_a = tools2.logsumexp(log_a[-1, :])
    check_log_lik_a = example['hmm_loglik']
    if log_lik_a == check_log_lik_a:
        print('True')
    
    
    if flag == 'HMM':
        utters = len(tidigits)
        models_len = len(models)
        log_lik_a = np.zeros((utters, models_len))

        for utter in range(utters):
            for digit in range(models_len):
                hmm_log_emlik = tools2.log_multivariate_normal_density_diag(tidigits[utter]['mfcc'], models[digit]['hmm']['means'], models[digit]['hmm']['covars'])
                log_a = forward(hmm_log_emlik, np.log(models[digit]['hmm']['startprob']), np.log(models[digit]['hmm']['transmat']))
                log_lik_a[utter, digit] = tools2.logsumexp(log_a[-1, :])
        #---->>> Check for misrecognized utterances
        miss = 0
        print("------ HMM a-pass ------")
        for utter in range(utters):
            best_score = np.argmax(log_lik_a[utter, :])
            print('tid digit, mod digit: ---> ' + str(tidigits[utter]['digit']) + " - " + str(models[best_score]['digit']))             # Uncomment to see the results analytically!
            if models[best_score]['digit'] != tidigits[utter]['digit']:
                miss += 1   
        accuracy = ((utters - miss) / utters) * 100   

        print()
        print("Misrecognized %d out of %d utterances." % (miss, len(tidigits)))
        print("Accuracy = " + str("%.2f" % round(accuracy,2)) + '%')
        print() 


    elif flag == "HMM as GMM":
        utters = len(tidigits)
        models_len = len(models)
        log_like_hmm = np.zeros((utters, models_len))

        for utter in range(utters):
            for digit in range(models_len):
                hmm_log_emlik = tools2.log_multivariate_normal_density_diag(tidigits[utter]['mfcc'], models[digit]['hmm']['means'], models[digit]['hmm']['covars'])
                weights_hmm = np.ones(models[digit]['hmm']['startprob'].shape[0]) / models[digit]['hmm']['startprob'].shape[0] 
                log_like_hmm[utter, digit] = gmmloglik(hmm_log_emlik, weights_hmm)
                
        #---->>> Check for misrecognized utterances
        miss = 0
        print("------ HMM as GMM ------")
        for utter in range(utters):
            best_score = np.argmax(log_like_hmm[utter, :])
            print('tid digit, mod digit: ---> ' + str(tidigits[utter]['digit']) + " - " + str(models[best_score]['digit']))             # Uncomment to see the results analytically!
            if models[best_score]['digit'] != tidigits[utter]['digit']:
                miss += 1   
        accuracy = ((utters - miss) / utters) * 100   

        print()
        print("Misrecognized %d out of %d utterances." % (miss, len(tidigits)))
        print("Accuracy = " + str("%.2f" % round(accuracy,2)) + '%')
        print() 
        

def quest_6_2():
    #--- Quest 6.2: ---->>> Viterbi approximation
    ''' # --->>> check
    hmm_log_emlik = tools2.log_multivariate_normal_density_diag(X, models[0]['hmm']['means'], models[0]['hmm']['covars'])
    log_v, best_path = viterbi(hmm_log_emlik, np.log(models[0]['hmm']['startprob']), np.log(models[0]['hmm']['transmat']))
    check_viterbi = example['hmm_vloglik']
    my_viterbi = []
    my_viterbi.append(log_v[-1, -1])
    my_viterbi.append(best_path)
    print('my_viterbi =', my_viterbi)
    print('check_viterbi =', check_viterbi)
    '''

    # ---->>>> Plot the alpha array that you obtained in the previous Section and overlay the best path obtained by Viterbi decoding
    hmm_log_emlik = tools2.log_multivariate_normal_density_diag(X, models[0]['hmm']['means'], models[0]['hmm']['covars'])
    log_a = forward(hmm_log_emlik, np.log(models[0]['hmm']['startprob']), np.log(models[0]['hmm']['transmat']))
    log_v, best_path = viterbi(hmm_log_emlik, np.log(models[0]['hmm']['startprob']), np.log(models[0]['hmm']['transmat']))
    
    fig4 = plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(log_a.T, cmap = 'jet')
    plt.plot(best_path)
    plt.title('hmm_logalpha & Best Path')
    plt.xticks([], [])
    plt.gca().invert_yaxis()

    plt.subplot(2, 1, 2)
    plt.imshow(example['hmm_logalpha'].T, cmap = 'jet')
    plt.title("Check hmm_logalpha & Best Path")
    plt.plot(example['hmm_vloglik'][1])
    plt.xticks([], [])
    plt.gca().invert_yaxis()

    plt.savefig("../figs/quest_6_2.png", bbox_inches='tight')
    
    #plt.show()

    utters = len(tidigits)
    models_len = len(models)
    loglik_v = np.zeros((utters, models_len))
    
   
    for utter in range(utters):
        for digit in range(models_len):
            hmm_log_emlik = tools2.log_multivariate_normal_density_diag(tidigits[utter]['mfcc'], models[digit]['hmm']['means'], models[digit]['hmm']['covars'])
            log_v, best_path = viterbi(hmm_log_emlik, np.log(models[digit]['hmm']['startprob']), np.log(models[digit]['hmm']['transmat']))
            loglik_v[utter, digit] = log_v[-1, -1]
         
    #---->>> Check for misrecognized utterances
    miss = 0
    print("------ Viterbi approximation ------")
    for utter in range(utters):
        best_score = np.argmax(loglik_v[utter, :])
        print('tid digit, mod digit: ---> ' + str(tidigits[utter]['digit']) + " - " + str(models[best_score]['digit']))             # Uncomment to see the results analytically!
        if models[best_score]['digit'] != tidigits[utter]['digit']:
            miss += 1   
    accuracy = ((utters - miss) / utters) * 100   

    print()
    print("Misrecognized %d out of %d utterances." % (miss, len(tidigits)))
    print("Accuracy = " + str("%.2f" % round(accuracy,2)) + '%')
    print() 







################################################################################## ----- Quests----- ##################################################################################


def main():
    #--- Quest 4: ---->>> Multivariate Gaussian Density
    #quest_4()
    
    #--- Quest 5: ---->>> GMM Likelihood and Recognition
    #quest_5()
    
    #--- Quest 6.1: ---->>> Forward Algorithm                              
    #quest_6_1(flag = "HMM")                                                     # Do Not forget to change the flag ---->>>> {'HMM' or 'HMM as GMM'}

    #--- Quest 6.1: ---->>> Viterbi approximation
    quest_6_2()
        
    plt.show()


if __name__ == "__main__":
    main()  
