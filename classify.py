#encoding=utf-8
from sklearn import neighbors,svm,tree
from sklearn.naive_bayes import GaussianNB
import numpy as np
import sys

def update_sp_prob(marginal_pb, spcond, result_proba):
    #============================================================================== 
    # This is the key function of this program, it calculates the local prior tuned 
    # probabilities of different categories and adjust the decision accordingly.
    smooth_factor=1.0 # The smooth factor f, this can changed according to application needs
    # The following follows Equation (4) & (7)
    # Equation (4) $P_{\mathcal{N}(u)}(y_i|\vec{x})=\frac{P(\vec{x}|y_i)P_{\mathcal{N}(u)}(y_i)}{\sum_{j=1}^{K}P(\vec{x}|y_j)P_{\mathcal{N}(u)}(y_j)}$
    # Equation (7) $P_{\mathcal{N}(u)}(y_i)=\frac{|\mathcal{N}_{y_i}(u)|+f\times P(y_i)}{|\mathcal{N}(u)|+f}$
    # Just replace $P_{\mathcal{N}(u)}(y_i)$ in Equation (4) with Equation (7)
        spcond[obj,:]+=marginal_pb*smooth_factor
        spcond[obj,:]/=spcond[obj,:].sum()*smooth_factor
    for obj in range(spcond.shape[0]):
        result_proba[obj,:]=result_proba[obj,:]/marginal_pb
        result_proba[obj,:]=result_proba[obj,:]*spcond[obj,:]
        result_proba[obj,:]/=result_proba[obj,:].sum()
    # calculating the decision features for each unseen object
    resnbsp=np.zeros((NR_obj),np.int)
    for obj in range(spcond.shape[0]):
        resnbsp[obj]=np.argmax(result_proba[obj,:])
    # returning results
    return resnbsp,result_proba


if __name__ == "__main__":
    if (len(sys.argv)!=3):
        print sys.argv[0], "sample_sp.csv test.csv "
        sys.exit(0)
    # Read the sample data.
    DATA=np.loadtxt(sys.argv[1], dtype=np.float)
    # Read the information of unseen objects.
    VALDATA=np.loadtxt(sys.argv[2], dtype=np.float)
    
    result=dict()
    result_proba=dict()
    
    # Read the number of categories
    NR_cate=int(VALDATA[0,0])
    # Read the conditioning features of sample data
    X=DATA[:,2:DATA.shape[1]-NR_cate-1]
    # Read the decision feature of sample data
    Y=np.copy(DATA[:,DATA.shape[1]-1].astype(int))
    # Read the conditioning features of unseen objects 
    V=VALDATA[:,2:VALDATA.shape[1]-1-NR_cate]

    #====================================================================================
    # Read the decision feature of unseen objects. This is just for verification of classification results.
    # The following three line can be deleted when used for real life applications.
    VY=VALDATA[:,VALDATA.shape[1]-1].astype(np.int)
    np.savetxt("result_real.csv",VY,fmt="%d")
    NR_obj=VY.shape[0]

    # ===================================================================================
    # Classifing unseen objects using traditional statistical classifiers. Naive Bayesian
    # as an example. It is easy to change the classifier here using scikit-learn.
    NBclf= GaussianNB()
    NBclf.fit(X,Y)
    result['NB']=NBclf.predict(V)
    result_proba['NB']=NBclf.predict_proba(V)
    result_proba['NB_sp']=result_proba['NB'].copy()
    #saving Naive Bayesian classifier results.
    np.savetxt("NB.csv",result['NB'],fmt="%d")
    np.savetxt("prob_NB.csv",result_proba['NB'])
    
    #=====================================================================================
    # Read the count of different categories in the neighborhood in the sample data of each
    # unseen object. Because it is easy to count the number, we ignored related codes and 
    # just use the result here.
    SP=VALDATA[:,VALDATA.shape[1]-1-NR_cate:VALDATA.shape[1]-1]
    
    #Calculating the marginal probability of all different categories
    marginal_pb=np.zeros((NR_cate),np.float)
    for obj in range(X.shape[0]):
        marginal_pb[Y[obj]]+=1.0
    marginal_pb=marginal_pb/marginal_pb.sum()

    #Tuning the local prior probability of different categories for the local classification and Rectifying the final classification results.
    result['NB_sp'],result_proba['NB_sp']= update_sp_prob(marginal_pb,SP,result_proba['NB'])

    #Saving the hard classification results
    np.savetxt("NB_sp.csv",result['NB_sp'],fmt="%d")
    
    #Saving the soft classification results
    np.savetxt("prob_NB_sp.csv",result_proba['NB_sp'])
