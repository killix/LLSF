% This function is designed to learning the label specific features and
% common features which shared by all the binary linear regression models
% The function Output the models parameters
% loss=||XW_s - Y||F^2 + lambda1 Tr(RW_s^TW_s) + lambda2||W_s||1 

    
function [model_ML]=LSCF_ML_SF( X, train_target,optmParameter)
    loss=0;
    %% optimization parameters
    alpha=optmParameter.alpha;
    lambda1 = optmParameter.lambda1;
    lambda2 = optmParameter.lambda2;
    maxIter = optmParameter.maxIter;
    minimumLossMargin = optmParameter.minimumLossMargin;
    outputtempresult = optmParameter.outputtempresult;

    num_dim = size(X,2);
    num_class = size(train_target,1);
    
    XTX = X'*X;
    XTY = X'*train_target';
    W_s = (XTX + alpha*eye(num_dim)) \ (XTY);
    % W_s = rand(num_dim,num_class);
    % W_s = zeros(num_dim,num_class);
    
    W_s_1 = W_s;
    
    %R=chowLiuTree(train_target, [0 1]);
    %R=ones(size(R))-R;
    R = pdist2( train_target+eps, train_target+eps, 'cosine' );


    iter=1;
    oldloss = 0;
    
    %Lip = sqrt(2*(norm(XTX)^2 + norm(lambda1*R)^2));
    Lip = norm(XTX) + norm(lambda1*R);
    
    bk_1 = 1; bk = 1;
    u0 = 1;
    theta = 10^(-9);
    enta = 0.9;
    u_ = theta*u0;
    u_k = u0;

    while iter<=maxIter

       W_s_k = W_s + (bk_1 - 1)/bk * (W_s - W_s_1);
       Gw_s_k = W_s_k - 1/Lip * u_k* ((XTX*W_s_k - XTY) + lambda1 * W_s_k*R);
       bk_1 = bk;
       bk = (1 + sqrt(4*bk^2 + 1))/2;
       W_s_1 = W_s;
       W_s = softthres(Gw_s_k,lambda2/Lip);
       
      %%
       specificloss = trace((X*W_s - train_target')'*(X*W_s - train_target'));
       traceW_S = trace(R*W_s'*W_s);
       sparesW_s = sum(sum(W_s~=0));
       
       totalloss = specificloss + lambda1*traceW_S + lambda2*sparesW_s;
       
       if outputtempresult==1
            disp(['iteration ---  ',num2str(iter),'th/',num2str(maxIter)]);
            disp(['       specificloss: ',num2str(specificloss)]);
            disp(['trace of Matrix W_s: ',num2str(traceW_S)]);
            disp(['          ||W_s||1 : ',num2str(sparesW_s)]);
            disp(['         total loss: ',num2str(totalloss)]);
       end
       
       if loss==0
           loss = totalloss;
       elseif loss>=0
           loss = [loss,totalloss];
       end
       
       if abs(oldloss - totalloss) <= minimumLossMargin
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
    
    model_ML.W_s = W_s;

end


%% soft thresholding operator
function W = softthres(W_t,lambda)

    W = max(W_t-lambda,0) - max(-W_t-lambda,0);
    
end
