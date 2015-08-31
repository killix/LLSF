
function [model_LLSF] = LLSF( X, Y, optmParameter)
% This function is designed to learning the label specific features each label and
% common features which shared by each combination of two labels.
% 
%
%    Syntax
%
%       [model_LLSF] = LLSF( X, train_target,optmPara)
%
%    Input
%       X           - a n by d data matrix, n is the number of instances and d is the number of features 
%       Y           - a n by l label matrix, n is the number of instances and l is the number of labels
%       optmPara    - A struct variable with seven fields, the optimization parameters for LLSF
%                   1) alpha            - weight for label correlation
%                   2) beta             - weight for sparsity
%                   3) gama             - weight for linear regression
%                   4) maxIter          - number of maximum iterations
%                   5) miniLossMargin   - minmum loss margin between two iterations
%                   6  drawConvergence  - whether drawing the convegence line, {1 - yes, 0 - not}
%                   7) outputtempresult - whether outputing the temporal resutls, {1 - yes, 0 - not}
%   Output
%
%       model_LLSF  - a d by l Coefficient matrix
%
%[1] J. Huang, G.-R Li, Q.-M. Huang and X.-D. Wu. Learning Label Specific Features for Multi-Label Classifcation. 
%    In: Proceedings of the International Conference on Data Mining, 2015.

    
    %% optimization parameters
    
    alpha            = optmParameter.alpha;
    beta             = optmParameter.beta;
    gamma            = optmParameter.gamma;
    maxIter          = optmParameter.maxIter;
    miniLossMargin   = optmParameter.minimumLossMargin;
    outputtempresult = optmParameter.outputtempresult;

    num_dim = size(X,2);
    
    XTX = X'*X;
    XTY = X'*Y;
    
    W_s   = (XTX + gamma*eye(num_dim)) \ (XTY);
    W_s_1 = W_s;
    R     = pdist2( Y'+eps, Y'+eps, 'cosine' );


    iter    = 1;
    loss    = 0;
    oldloss = 0;
    
    Lip = sqrt(2*(norm(XTX)^2 + norm(alpha*R)^2));
    %Lip = norm(XTX) + norm(lambda1*R);
    
    bk = 1;
    bk_1 = 1; 
    u_k = 1;
    while iter <= maxIter

       W_s_k  = W_s + (bk_1 - 1)/bk * (W_s - W_s_1);
       Gw_s_k = W_s_k - 1/Lip * u_k* ((XTX*W_s_k - XTY) + alpha * W_s_k*R);
       bk_1   = bk;
       bk     = (1 + sqrt(4*bk^2 + 1))/2;
       W_s_1  = W_s;
       W_s    = softthres(Gw_s_k,beta/Lip);
       
      %%
       specificloss = trace((X*W_s - Y)'*(X*W_s - Y));
       traceW_S     = trace(R*W_s'*W_s);
       sparesW_s    = sum(sum(W_s~=0));
       
       totalloss = specificloss + alpha*traceW_S + beta*sparesW_s;
       
       if outputtempresult==1
            disp(['iteration ---  ',num2str(iter),'th/',num2str(maxIter)]);
            disp(['       specificloss: ',num2str(specificloss)]);
            disp(['trace of Matrix W_s: ',num2str(traceW_S)]);
            disp(['          ||W_s||1 : ',num2str(sparesW_s)]);
            disp(['         total loss: ',num2str(totalloss)]);
       end
       
       if loss == 0
           loss = totalloss;
       elseif loss >= 0
           loss = [loss,totalloss];
       end
       
       if abs(oldloss - totalloss) <= miniLossMargin
           break;
       elseif totalloss <=0
           break;
       else
           oldloss = totalloss;
       end
       
       iter=iter+1;
    end

    if optmParameter.drawConvergence == 1
        x=1:length(loss);
        figure;
        plot(x,loss);
    end
    
    model_LLSF = W_s;

end


%% soft thresholding operator
function W = softthres(W_t,lambda)

    W = max(W_t-lambda,0) - max(-W_t-lambda,0);
    
end
