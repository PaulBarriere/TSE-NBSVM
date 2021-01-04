''' Function vectorize returns:
        - the Vector V defined as the feature vector (ie a vector containing 
        strings which are the list of vocabulary)
        
        - the matrix F where the ith column is f(i) the feature count vector
          for training case i as defined in the paper. F is therefore a matrix 
          of size : len(V) x len(train_x) (let's remind that V is the vocabulary 
          vector and that len(train_x) is the number of reviews in the training 
          example)

          - vectorization which is the CountVectorizer fittid with the data. This
          is used later to transform the test data.


    Function vectorize uses:
        - Train_x_sample: a list of strings (list of reviews)

        - ngrams: a tupple with for example (1,1) meaning that you want only 
                  unigrams and (1,2) meaning that you want both unigrams and 
                  bigrams
'''

def vectorize(Train_x_sample, ngrams):
  vectorization =  CountVectorizer(ngram_range = ngrams)
  vectorization.fit(Train_x_sample)
  V = vectorization.get_feature_names()
  
  F = vectorization.transform(Train_x_sample)
  F = F.toarray().T

  return V, F, vectorization




''' Function get_P returns:
        - the Vector P as defined is the paper. This vector is the sum of two
        element: alpha a smoothing parameter and the sum of f(i) for all i where 
        the review is positive (ie we the reviews where train_y = 1 and sum the
        columns of the matrix F. )
  
    Function vectorize uses:
        - alpha a real number between 0 and 1
        
        - Train_y_sample which is the value of a review (positive / negative) of
        the training dataset. 

        - The matrix F
'''

def get_P(alpha, Train_y_sample, F):
  pos_list = [i for i, value in enumerate(Train_y_sample) if value == 1]
  restrict_F = F[:,pos_list]
  P = alpha + np.sum(restrict_F, axis = 1)
  return P


''' Function get_Q returns:
        - the Vector Q as defined is the paper. This vector is the sum of two
        element: alpha a smoothing parameter and the sum of f(i) for all i where 
        the review is negative (ie we the reviews where train_y = -1 and sum the
        columns of the matrix F. )
  
    Function vectorize uses:
        - alpha a real number between 0 and 1
        
        - Train_y_sample which is the value of a review (positive / negative) of
        the training dataset. 

        - The matrix F
'''

def get_Q(alpha, Train_y_sample, F):
  neg_list = [i for i, value in enumerate(Train_y_sample) if value == 0]
  restrict_F = F[:,neg_list]
  Q = alpha + np.sum(restrict_F, axis = 1)
  return Q


''' Function get_r returns:
        - the vector r which is the log-count ratio defined in the paper. 

    Function get_r uses: 
        - functions get_P and get_Q as defined above

        - alpha a real number between 0 and 1
        
        - Train_y_sample which is the value of a review (positive / negative) of
        the training dataset. 

        - The matrix F
'''

def get_R(alpha, Train_y_sample, F):
  P = get_P(alpha, Train_y_sample, F)
  Q = get_Q(alpha, Train_y_sample, F)
  
  P = P / np.linalg.norm(P,ord=2)
  Q = Q / np.linalg.norm(Q,ord=2)
  R = np.log(P/Q)

  return R



''' --------------------------------------------------------------------------
Multinomial Naive Bayes (MNB)

Function MNB_fit_model returns:
        - W as defined in the paper. In this case W is R. So the function
        compute R.

        - X as defined in the paper. Here it is f or ^f (depending on what you 
        choose)

        - B as defined in the paper (log(N+/N-))

Function MNB_fit_model uses:
        - Train_x_sample: a list of strings (list of reviews)

        - Train_y_sample which is the value of a review (positive / negative) of
        the training dataset. 

        - ngrams : a tupple of two integer. 

        - alpha : smoothing parameter

        - binarized : a boolean parameter. If TRUE then we tranform F (and p,q,
        r) by taking for each element of F 1 if it is strictly positive and 0 
        otherwise. 

'''

def MNB_fit_model(Train_x_sample, Train_y_sample, 
                  ngrams, alpha = 1, binarized=False):
  # Define V and F
  V,F, vectorization = vectorize(Train_x_sample, ngrams)


  # We change F if Binarized == TRUE:
  if binarized == True:
    F = np.where(F > 0,1,0)

  # Define R and so W:
  R = get_R(alpha, Train_y_sample, F)
  W = R

  # Compute B:
  nb_neg = len([i for i, value in enumerate(Train_y_sample) if value == -1])
  nb_pos = len([i for i, value in enumerate(Train_y_sample) if value == 1])
  B = np.log(nb_pos/nb_neg)

  return W, B, binarized, vectorization


''' Function predict returns:
        - Preds a vector of predictions containing -1 or 1. 

Function predict uses:
        - sample: a list of strings (list of reviews) on which to predict if it 
        is a positive or negative review. 

        - the parameters found in the fitting (binarized,W,B)

The aim is to use this function for all cases MNB, SVM and NBSVM

'''

def predict(binarized,W,B,vectorization,sample):
  # Transform the sample in term of V:
  sample = vectorization.transform(sample).toarray().T

  # Compute the prediction
  predictions =  np.sign(np.dot(W.T, sample) + B)
  return predictions

''' Function eval returns:
        - a vector containing the accuracy, precision, recall and F1 score.

Function eval uses:
        - predictions: a list of integer equals to 1 or -1. 

        - true_value: a list of integers equals to 1 or -1. 

The aim is to use this function for all cases MNB, SVM and NBSVM

'''

def eval(predictions, y_test):
  print('accuracy {}'.format(accuracy_score(y_test, predictions)))
  print(classification_report(y_test, predictions))
  return 

''' Function cross_validation_NBSVM returns:
        - an integer which is the average of the accuracy_score of all step of
         cross validation.

Function eval uses:
        - sample_x: a list of reviews. 

        - sample_y: a list of integers equals to 1 or -1. 

        - cv: an integer which is the number of division of our sample we do

        - alpha: a real number which is the parameter of the NBSVM model.


'''

def cross_validation_NBSVM(sample_x, sample_y, cv, alpha):

  # First we shuffle 
  sample = list(zip(sample_x, sample_y))
  random.shuffle(sample)
  new_sample_x, new_sample_y = zip(*sample)

  # Then we divide the sample into cv time and define test and train set:

  lenght = round(len(new_sample_x)/cv)
  accuracy = [] # list to stock the accuraccy
  for i in range(cv):
    if i == cv-1:
      test_x = new_sample_x[lenght*i:]
      test_y = new_sample_y[lenght*i:]
      train_x = new_sample_x[:lenght*i]
      train_y = new_sample_y[:lenght*i]
    else:
      test_x = new_sample_x[lenght*i:lenght*(i+1)]
      test_y = new_sample_y[lenght*i:lenght*(i+1)]
      train_x = new_sample_x[:lenght*i] + new_sample_x[lenght*(i+1):]
      train_y = new_sample_y[:lenght*i] + new_sample_y[lenght*(i+1):]
    
    # we compute the models parameters
    W, B, binarized, vectorization = MNB_fit_model(Train_x_sample = train_x, 
                                                  Train_y_sample = train_y,
                                                  ngrams = (1,2), 
                                                  alpha = alpha, 
                                                  binarized=False)
    
    # We compute the predictions
    y_pred = predict(binarized,W,B,vectorization,test_x)

    # We then stock the accuracy
    accuracy.append(accuracy_score(test_y, y_pred, sample_weight=None))

    return np.mean(accuracy)



''' --------------------------------------------------------------------------
SVM and NBSVM

Function NBSVM_fit_model returns:
        - a model (sklearn)

Function NBSVM_fit_model uses:
        - Train_x_sample: a list of strings (list of reviews)

        - Train_y_sample which is the value of a review (positive / negative) of
        the training dataset. 

        - ngrams : a tupple of two integer. 

        - alpha : smoothing parameter

        - NB : a boolean argument meaning that the model is NBSVM if True and 
        just SVM if False. 

'''

def NB_SVM_fit_model(Train_x_sample, Train_y_sample, 
                  ngrams, NB=True, alpha = 10):
  
  V, F, vectorization = vectorize(Train_x_sample = train_x, ngrams=(1,2))
  # Since binarized = True always here.
  F = np.where(F > 0,1,0)

  # Fit the model
  if NB == False:
    clf = svm.LinearSVC()
    clf.fit(F.T, train_y)
  else:
    R = get_R(alpha=alpha, Train_y_sample = Train_y_sample, F = F)
    # We define first a matrix r with the same number of columns as F and with each 
    # column equal to R.
    r = np.array([list(R),]*len(F[0]))
    r = r.transpose()

    # We do the elementwise produt of R and F.
    product = np.multiply(r,F)

    # We define and fit the model:
    clf = svm.LinearSVC()
    clf.fit(product.T, train_y)

  return clf
